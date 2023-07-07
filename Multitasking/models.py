import os.path
import torch
from torch.nn import Module, ModuleDict
from torch.nn import Linear
from Multitasking.clip_load import load

from OPENCLIP.factory import create_model_and_transforms
from Multitasking.swin_models import SwinTransformer
import wget


OPEN_MODELS = {
    "open-ViT-L/14": 'hf-hub:laion/Multitasking-ViT-L-14-laion2B-s32B-b82K',
    "open-ViT-H/14": 'hf-hub:laion/Multitasking-ViT-H-14-laion2B-s32B-b79K',
    "open-ViT-G/14": 'hf-hub:laion/Multitasking-ViT-bigG-14-laion2B-39B-b160k',
}


class ClipManager(Module):

    def __init__(self, params):
        super(ClipManager, self).__init__()
        self.params = params
        self.attr_linear = None
        self.classif_linear = None

        if params["clip_model"] in ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
            self.clip, preprocess = load(params["clip_model"], self.params, device=params["device"], jit=False, download_root="./Models")
            self.preprocess = {
                "train": preprocess,
                "eval": preprocess
            }
            self.open_model = False
        else:
            self.clip, preprocess_train, preprocess_val = create_model_and_transforms(OPEN_MODELS[params["clip_model"]], device=params["device"], jit=False, cache_dir="./Models")
            self.preprocess = {
                "train": preprocess_train,
                "eval": preprocess_val
            }
            self.open_model = True

        self.params["emb_dim"] = self.clip.visual.output_dim
        self.params["text_dim"] = self.clip.text_projection.size(0)

        self.additional_vision_models = ModuleDict()
        for key in self.params["config"].keys():
            name, freeze, model_part = self.params["config"][key]
            if name == "swin_vision":
                self.additional_vision_models[key] = SwinTransformer(num_classes=self.params["emb_dim"], img_size=params["input_size"][0], window_size=12, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
                swin_path = "Models/swin_large_patch4_window12_384_22k.pth"
                if not os.path.exists(swin_path):
                    wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth", out="Models")
                weights = torch.load(swin_path, map_location="cpu")["model"]
                del weights["head.weight"]
                del weights["head.bias"]
                self.additional_vision_models[key].load_state_dict(weights, strict=False)
                self.clip.visual = None

        self.params["visual_dim"] = self.clip.visual.proj.size(0) if self.params["config"]["vision"][0] != "swin_vision" else self.additional_vision_models["vision"].head.weight.size(1)

        if params["classif_linear"]:
            self.classif_linear = Linear(self.params["emb_dim"], self.params["num_classes"])

        if params["attr_linear"]:
            self.attr_linear = Linear(self.params["emb_dim"], self.params["num_attributes"])

        self.visual_loc_proj = Linear(self.params["visual_dim"], self.params["emb_dim"])


    @property
    def dtype(self):
        return self.clip.visual.conv1.weight.dtype if self.clip.visual is not None else self.additional_vision_models["vision"].head.weight.dtype

    def normalize_embedding(self, emb):
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_text(self, text, return_after=None):
        pos = text.argmax(dim=-1)
        seq = self.clip.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        seq = seq + self.clip.positional_embedding.type(self.dtype)

        seq = seq.permute(1, 0, 2)  # NLD -> LND
        seq = self.clip.transformer.all_but_last_layer(seq, attn_mask=None if not self.open_model else self.clip.attn_mask)
        seq = seq.permute(1, 0, 2)

        if return_after == "transformer-1":
            return seq, seq, pos

        seq = seq.permute(1, 0, 2)
        seq = self.clip.transformer.last_layer(seq, attn_mask=None if not self.open_model else self.clip.attn_mask)
        seq = seq.permute(1, 0, 2)  # LND -> NLD

        seq = self.clip.ln_final(seq).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = seq[torch.arange(seq.shape[0]), pos]
        if return_after == "transformer":
            return x, seq, pos

        x = x @ self.clip.text_projection
        if return_after == "proj":
            return x, seq, pos

        x = self.normalize_embedding(x)
        return x, seq, pos

    def proj_text(self, x, seq, pos, features_from=None):
        last_tr = proj = norm = False
        if features_from == "transformer-1":
            last_tr = proj = norm = True
        elif features_from == "transformer":
            proj = norm = True
        elif features_from == "proj":
            norm = True

        if last_tr:
            seq = seq.permute(1, 0, 2)
            seq = self.clip.transformer.last_layer(seq)
            seq = seq.permute(1, 0, 2)  # LND -> NLD
            seq = self.clip.ln_final(seq).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = seq[torch.arange(seq.shape[0]), pos]

        if proj:
            x = x @ self.clip.text_projection

        if norm:
            x = self.normalize_embedding(x)

        return x, seq

    def encode_image(self, x, return_after=None):
        x = x.type(self.dtype)

        if "vision" in self.additional_vision_models:
            return self.additional_vision_models["vision"](x, return_after=return_after)

        if self.open_model and self.clip.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.clip.visual.grid_size[0], self.clip.visual.patch_size[0], self.clip.visual.grid_size[1],
                          self.clip.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.clip.visual.grid_size[0] * self.clip.visual.grid_size[1], -1)
            x = self.clip.visual.patchnorm_pre_ln(x)
            x = self.clip.visual.conv1(x)
        else:
            x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)

        if self.open_model:
            x = self.clip.visual.patch_dropout(x)

        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        seq = self.clip.visual.transformer.all_but_last_layer(x)
        seq = seq.permute(1, 0, 2)

        if return_after == "transformer-1":
            return seq, seq

        seq = seq.permute(1, 0, 2)
        seq = self.clip.visual.transformer.last_layer(seq)
        seq = seq.permute(1, 0, 2)  # LND -> NLD

        if self.open_model:
            if self.clip.visual.attn_pool is not None:
                seq = self.clip.visual.attn_pool(seq)
                seq = self.clip.visual.ln_post(seq)
                x = self.clip.visual._global_pool(seq)
            else:
                x = self.clip.visual._global_pool(seq)
                x = self.clip.visual.ln_post(x)
                seq = self.clip.visual.ln_post(seq)
        else:
            seq = self.clip.visual.ln_post(seq)
            x = seq[:, 0, :]

        if return_after == "transformer":
            return x, seq

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        if return_after == "proj":
            return x, seq

        return x, seq

    def proj_vision(self, x, seq, features_from=None):
        if "vision" in self.additional_vision_models:
            return self.additional_vision_models["vision"].proj(x, seq, features_from)

        last_tr = proj = False
        if features_from == "transformer-1":
            last_tr = proj =  True
        elif features_from == "transformer":
            proj = True

        if last_tr:
            seq = seq.permute(1, 0, 2)
            seq = self.clip.visual.transformer.last_layer(seq)
            seq = seq.permute(1, 0, 2)  # LND -> NLD

            if self.open_model:
                if self.clip.visual.attn_pool is not None:
                    seq = self.clip.visual.attn_pool(seq)
                    seq = self.clip.visual.ln_post(seq)
                    x = self.clip.visual._global_pool(seq)
                else:
                    x = self.clip.visual._global_pool(seq)
                    x = self.clip.visual.ln_post(x)
                    seq = self.clip.visual.ln_post(seq)
            else:
                seq = self.clip.visual.ln_post(seq)
                x = seq[:, 0, :]

        if proj:
            x = x @ self.clip.visual.proj

        return x, seq

    def proj_visual_for_loc(self, x):
        return self.visual_loc_proj(x)

    def compute_class_linear_scores(self, vision_embedding):
        return self.classif_linear(vision_embedding)

    def compute_attr_linear_scores(self, vision_embedding):
        return self.attr_linear(vision_embedding)


def check_or_download_model_weights():
    fold = os.path.join("Results", "open-ViT-L-14_swin_clip_text_finetuned", "model")
    os.makedirs(fold, exist_ok=True)
    if len(os.listdir(fold)) == 0:
        print("Downloading pre-trained model weights, this can take a moment...")
        wget.download("https://zenodo.org/record/8124014/files/last_100.pt?download=1", out=fold)
        print("Download completed")