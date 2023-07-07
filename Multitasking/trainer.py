import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from Multitasking.dataset_utils.cub200_dataset import CUBDataset
from Multitasking.metrics import MetricManager
from Multitasking.models import ClipManager
from Multitasking.clip_models import convert_weights
from clip import tokenize as clip_tokenize
from open_clip import tokenize as open_clip_tokenize
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import os
import sys
import copy
import json
from datetime import date


class Trainer:

    def __init__(self, datasets, params):
        self.datasets = datasets
        self.params = params
        self.device = "cuda:{}".format(params["training"]["gpu_index"]) if torch.cuda.is_available() and params["training"]["gpu_index"] != "cpu" else "cpu"
        self.tokenize_fn = open_clip_tokenize if "open-" in self.params["model"]["clip_model"] else clip_tokenize
        self.model = None
        self.optimizer = None
        self.scaler = None

        self.load_model()
        self.last_epoch = 0
        self.last_step = 0
        self.best = None
        self.acc_class_features = None
        self.acc_attr_features = None

        self.output_folder = os.path.join("./Results", self.params["training"]["output_fold_name"])
        self.params_path = self.get_param_path()
        self.dataloaders = self.load_dataloaders()
        self.metric_managers = dict()
        os.makedirs(self.output_folder, exist_ok=True)

        self.eval_features = self.init_eval_features()
        self.fine_tuning_features = self.init_fine_tuning_features()

        self.freeze = {
            "clip_vision": False,
            "swin_vision": False,
            "clip_text": False,
        }

        self.train_ = {
            "vision":  self.params["training"]["train_class_image"] or self.params["training"]["train_attr_image"] or self.params["training"]["train_triplet_class"] or self.params["training"]["train_triplet_attr"] or self.params["training"]["train_attr_linear"] or self.params["training"]["train_class_linear"],
            "class": self.params["training"]["train_class_image"] or self.params["training"]["train_triplet_class"] or self.params["training"]["train_class_attr"],
            "attr": self.params["training"]["train_attr_image"] or self.params["training"]["train_triplet_attr"] or self.params["training"]["train_class_attr"],
        }

        self.loss_fn_ce = CrossEntropyLoss()
        self.loss_fn_bce = BCEWithLogitsLoss()

        self.model_config = self.params["model"]["config"]

        self.use_loc = self.params["training"]["train_loc"] or np.any(["loc_mAP" in self.params["training"]["metric_names"][key] for key in self.params["training"]["metric_names"].keys()])

    def load_dataloaders(self):
        dataloaders = dict()
        for set_name in self.datasets:
            dataloaders[set_name] = DataLoader(dataset=self.datasets[set_name],
                                               batch_size=self.params["training"]["batch_size"][set_name],
                                               shuffle=True if set_name == "train" else False,
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=self.datasets[set_name].collate_fn,
                                               sampler=None,
                                               batch_sampler=None,
                                               num_workers=8,
                                               )
        if "train" in self.datasets.keys():
            dataloaders["train_no_shuffle"] = DataLoader(dataset=self.datasets["train"],
                                                   batch_size=self.params["training"]["batch_size"]["train"],
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   drop_last=False,
                                                   collate_fn=self.datasets[set_name].collate_fn,
                                                   sampler=None,
                                                   batch_sampler=None,
                                                   num_workers=8,
                                                   )
        return dataloaders

    def zero_optimizer(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step_optimizer(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def backward_loss(self, loss, retain_graph=False):
        self.scaler.scale(loss).backward(retain_graph=retain_graph)

    def load_model(self):
        self.params["model"]["num_classes"] = self.get_default_dataset().num_classes
        if isinstance(self.get_default_dataset(), CUBDataset):
            self.params["model"]["num_attributes"] = self.get_default_dataset().num_attr
        self.params["model"]["device"] = self.device
        self.model = ClipManager(self.params["model"])
        for set_name in self.datasets:
            key = "train" if set_name == "train" else "eval"
            self.datasets[set_name].set_preprocess(self.model.preprocess[key])
        self.model.to(self.device)
        if self.params["training"]["use_amp"]:
            convert_weights(self.model)
        else:
            self.model.float()
        self.scaler = GradScaler(enabled=self.params["training"]["use_amp"])
        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = self.params["training"]["optimizer"]["class"](list(self.model.parameters()), **self.params["training"]["optimizer"]["args"])

    def save_weights(self, mode="best"):
        weights_fold = os.path.join(self.output_folder, "model")
        os.makedirs(weights_fold, exist_ok=True)
        to_del = list()
        for filename in os.listdir(weights_fold):
            if mode in filename:
                to_del.append(os.path.join(weights_fold, filename))
        path = os.path.join(weights_fold, "{}_{}.pt".format(mode, self.last_epoch))
        content = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "last_epoch": self.last_epoch,
            "last_step": self.last_step,
            "best": self.best,
        }
        torch.save(content, path)

        for path_to_del in to_del:
            if path_to_del != path:
                os.remove(path_to_del)

    def load_weights(self, mode="best"):
        weights_fold = os.path.join(self.output_folder, "model")
        os.makedirs(weights_fold, exist_ok=True)
        filename = None
        for filename in os.listdir(weights_fold):
            if mode in filename:
                break
        if filename is None:
            print("new training")
        else:
            checkpoint_path = os.path.join(weights_fold, filename)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.last_epoch = checkpoint["last_epoch"]
            self.last_step = checkpoint["last_step"]
            self.best = checkpoint["best"]
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except:
                # in case of removed unused layer in model
                pass
            print("loaded weights - epoch {}".format(self.last_epoch))
        if self.params["training"]["use_amp"]:
            convert_weights(self.model)
        else:
            self.model.float()

    def free_memory(self):
        self.fine_tuning_features = self.init_fine_tuning_features()
        self.eval_features = self.init_eval_features()

    def init_eval_features(self):
        return {
            "image": dict(),
            "image_sequence": dict(),
            "class_text": None,
            "attr_text": None,
            "attr_text_sequence": None,
            "attr_text_eot_pos": None,
        }

    def init_fine_tuning_features(self):
        return {
            "images": dict(),
            "images_sequence": dict(),
            "class_text": None,
            "class_text_sequence": None,
            "class_text_eot_pos": None,
            "attr_text": None,
            "attr_text_sequence": None,
            "attr_text_eot_pos": None,
        }

    def get_param_path(self):
        path = os.path.join(self.output_folder, "params.txt")
        i = 1
        while os.path.isfile(path):
            path = "{}_{}.txt".format(os.path.join(self.output_folder, "params"), i)
            i += 1
        return path

    def save_params(self):
        """
        Output text file containing a summary of all hyperparameters chosen for the training
        """
        def compute_nb_params(module):
            return sum([np.prod(p.size()) for p in list(module.parameters())])

        def class_to_str_dict(my_dict):
            for key in my_dict.keys():
                if callable(my_dict[key]):
                    my_dict[key] = my_dict[key].__name__
                elif isinstance(my_dict[key], np.ndarray):
                    my_dict[key] = my_dict[key].tolist()
                elif isinstance(my_dict[key], dict):
                    my_dict[key] = class_to_str_dict(my_dict[key])
            return my_dict

        params = copy.deepcopy(self.params)
        params = class_to_str_dict(params)
        params["date"] = date.today().strftime("%d/%m/%Y")
        clip_params = compute_nb_params(self.model)
        params["models"] = {
            "model": "{:,}".format(clip_params),
        }

        params["hardware"] = dict()
        if self.device != "cpu":
            index_gpu = torch.cuda.current_device()
            params["hardware"][str(index_gpu)] = "{} {}".format(torch.cuda.get_device_name(index_gpu), torch.cuda.get_device_properties(index_gpu))
        else:
            params["hardware"]["0"] = "CPU"
        params["software"] = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        with open(self.params_path, 'w') as f:
            json.dump(params, f, indent=4)

    def update_freeze_model(self):
        def freeze_text_module(module, all=False):
            if module is None:
                return
            self.freeze_module(module.token_embedding)
            self.freeze_module(module.positional_embedding)
            if all:
                self.freeze_module(module.transformer)
                self.freeze_module(module.text_projection)
                self.freeze_module(module.ln_final)
            else:
                for r in module.transformer.resblocks[:-1]:
                    self.freeze_module(r)

        def freeze_vision_module(module, all=False):
            self.freeze_module(module)
            if not all:
                self.unfreeze_module(module.transformer.resblocks[-1])
                self.unfreeze_module(module.proj)
                self.unfreeze_module(module.ln_post)

        def freeze_swin_module(module, all=False):
            self.freeze_module(module)
            if not all:
                self.unfreeze_module(module.head)
                self.unfreeze_module(module.norm)
                self.unfreeze_module(module.layers[-1].blocks[-1])

        self.unfreeze_module(self.model)
        config = self.params["model"]["config"]

        module_names = np.unique([j[0] for i, j in config.items() if j is not None])
        for mod in module_names:
            self.freeze[mod] = False
            if np.all([j[1] for i, j in config.items() if j[0] == mod]):
                all = np.all(np.array([j[2] for i, j in config.items() if j[0] == mod]) == "all")
                self.freeze[mod] = True
                if mod == "clip_vision":
                    freeze_vision_module(self.model.clip.visual, all)
                elif mod == "swin_vision":
                    freeze_swin_module(self.model.additional_vision_models["vision"], all)
                elif mod == "clip_text":
                    freeze_text_module(self.model.clip, all)

        # Pre-compute latent representation to save computation time when training last layers only
        self.return_after = "transformer-1"

        #  pre-process all data to only train last layers
        with autocast(enabled=self.params["training"]["use_amp"]):
            if self.freeze[self.model_config["vision"][0]] and ("train" not in self.fine_tuning_features["images"] or self.train_["vision"]):
                self.fine_tuning_features["images"]["train"], self.fine_tuning_features["images_sequence"]["train"] = self.compute_all_image_embedding("train", return_after=self.return_after, train=False, device="cpu")
            if self.freeze[self.model_config["vision"][0]] and "valid" not in self.fine_tuning_features["images"]:
                self.fine_tuning_features["images"]["valid"], self.fine_tuning_features["images_sequence"]["valid"] = self.compute_all_image_embedding("valid", return_after=self.return_after, train=False, device="cpu")
            if self.freeze[self.model_config["class"][0]] and self.fine_tuning_features["class_text"] is None:
                self.fine_tuning_features["class_text"], self.fine_tuning_features["class_text_sequence"], self.fine_tuning_features["class_text_eot_pos"] = self.compute_all_classif_embedding(return_after=self.return_after, train=False, device="cpu")

            if self.freeze[self.model_config["attr"][0]] and self.fine_tuning_features["attr_text"] is None:
                self.fine_tuning_features["attr_text"], self.fine_tuning_features["attr_text_sequence"], \
                self.fine_tuning_features["attr_text_eot_pos"] = self.compute_attr_embedding_all(return_after=self.return_after, train=False, device="cpu")

        # add number of trainable weights first epoch in params file
        if self.last_epoch == 1:
            with open(self.params_path, 'r') as f:
                params = json.load(f)
            with open(self.params_path, 'w') as f:
                params["trainable_parameters_first_epoch"] = "{:,}".format(sum([np.prod(p.size()) for p in list(self.model.parameters()) if p.requires_grad]))
                json.dump(params, f, indent=4)

    def set_requires_grad_module(self, module, requires_grad):
        if isinstance(module, torch.nn.Parameter):
            module.requires_grad = requires_grad
        else:
            for params in module.parameters():
                params.requires_grad = requires_grad

    def freeze_module(self, module):
        if module is None:
            return
        self.set_requires_grad_module(module, False)

    def unfreeze_module(self, module):
        self.set_requires_grad_module(module, True)

    def train(self):
        self.writer = SummaryWriter(self.output_folder)
        self.save_params()
        set_name = "train"
        num_epochs = self.params["training"]["max_num_epochs"]
        num_iter_before_update_weights = max(1, self.params["training"]["gradient_acc"] // self.params["training"]["batch_size"]["train"])
        num_iter_before_update_display = 1 if "num_iter_display_value_update" not in self.params["training"] else self.params["training"]["num_iter_display_value_update"]
        dataset = self.datasets[set_name]
        dataloader = self.dataloaders[set_name]
        self.load_weights(mode=self.params["training"]["load_weights"])
        if self.last_epoch == 0:
            self.update_freeze_model()
            self.eval_and_log()
        for num_epoch in range(self.last_epoch+1, num_epochs+1):
            self.last_epoch = num_epoch
            self.metric_managers["train"] = MetricManager(self.params["training"]["metric_names"]["train"], self.output_folder, self.datasets["train"])
            self.update_freeze_model()
            with tqdm(total=len(dataset)) as pbar:
                pbar.set_description("Training - epoch {}".format(num_epoch))
                for num_batch, batch_data in enumerate(dataloader):
                    self.last_step += 1
                    update_weights = self.last_step % num_iter_before_update_weights == 0
                    update_display = num_batch % num_iter_before_update_display == 0 or len(dataloader) == num_batch+1
                    self.model.train()
                    metrics = self.train_batch(batch_data, update_weights=update_weights)
                    metrics["sample_names"] = batch_data["sample_names"]
                    metrics["sample_ids"] = batch_data["sample_ids"]
                    metrics["nb_samples"] = [len(batch_data["sample_names"]), ]
                    self.metric_managers["train"].add_batch_values(metrics)
                    if update_display:
                        display_values = self.metric_managers[set_name].get_display_values()
                        pbar.set_postfix(values=str(display_values))
                    pbar.update(len(batch_data["sample_names"]))
            for key in display_values.keys():
                if display_values[key] is not None:
                    self.writer.add_scalar('{}_{}'.format("train", key), display_values[key], num_epoch)
            self.save_weights("last")
            if num_epoch % self.params["training"]["eval_on_valid_interval"] == 0:
                self.eval_and_log()

    def eval_and_log(self):
        focus_metric_name = self.params["training"]["metric_to_focus"]
        expected_metric_value = self.params["training"]["expected_metric_value"]
        eval_values = self.evaluate_classification("valid", output=self.last_epoch == 0)
        for key in eval_values.keys():
            if eval_values[key] is not None:
                self.writer.add_scalar('{}_{}'.format("valid", key), eval_values[key], self.last_epoch)
        if (self.best is None or
                (eval_values[focus_metric_name] < self.best and expected_metric_value == "low") or
                (eval_values[focus_metric_name] > self.best and expected_metric_value == "high")):
            self.save_weights(mode="best")
            self.best = eval_values[focus_metric_name]

    def train_batch(self, batch_data, update_weights=True):
        images = batch_data["images"].to(self.device)
        batch_size = images.size(0)
        loss_loc_oracle = loss_loc = loss_attr_image = loss_class_image = loss_class_linear = loss_attr_linear = None

        class_names = np.array(self.get_all_class_names())
        gt_image = torch.zeros((batch_size, class_names.size), device=self.device)
        for i in range(batch_size):
            gt_image[i, batch_data["class_ids"][i]] = 1

        with autocast(enabled=self.params["training"]["use_amp"]):
            logit_scale = self.model.clip.logit_scale.exp()

            # Compute IMAGE embedding
            if self.freeze[self.model_config["vision"][0]]:
                image_features = torch.stack([self.fine_tuning_features["images"]["train"][i] for i in batch_data["sample_ids"]], dim=0).to(self.device)
                image_sequence = torch.stack([self.fine_tuning_features["images_sequence"]["train"][i] for i in batch_data["sample_ids"]], dim=0).to(self.device)
                image_features, image_sequence = self.model.proj_vision(image_features, image_sequence, features_from=self.return_after)
            else:
                image_features, image_sequence = self.compute_image_embedding(images)
            image_emb = self.model.normalize_embedding(image_features)

            # Compute CLASSIFICATION embedding
            if self.acc_class_features is None:
                if self.freeze[self.model_config["class"][0]]:
                    class_features, class_sequence = self.get_class_from_freezing()
                else:
                    class_features, class_sequence, _ = self.compute_classif_embedding(class_names, device=self.device, train=True)
                self.acc_class_features = (class_features, class_sequence)
            else:
                class_features, class_sequence = self.acc_class_features

            # Compute ATTRIBUTES embedding
            attr_classes = batch_data["class_ids"].tolist()
            if self.freeze[self.model_config["attr"][0]]:
                if self.acc_attr_features is None:
                    self.acc_attr_features = self.get_attr_from_freezing([0, ])
                attr_features, attr_sequence = self.acc_attr_features
                attr_sequence = torch.repeat_interleave(attr_sequence, batch_size, dim=0)
                attr_features = torch.repeat_interleave(attr_features, batch_size, dim=0)
            else:
                if self.acc_attr_features is None:
                    feat, seq, _, _ = self.compute_attr_embedding([0, ], device=self.device, train=True)
                    self.acc_attr_features = (feat[0], seq[0])
                attr_features, attr_sequence = self.acc_attr_features
                attr_sequence = torch.repeat_interleave(attr_sequence.unsqueeze(0), batch_size, dim=0)
                attr_features = torch.repeat_interleave(attr_features.unsqueeze(0), batch_size, dim=0)

            # Compute similarity scores
            class_image_scores = (logit_scale * image_emb @ class_features.t()).float()
            attr_image_scores = (logit_scale * image_emb.unsqueeze(1) @ attr_features.permute(0, 2, 1)).squeeze(1).float()

            if self.params["training"]["use_negative_attr"]:
                num_attr = attr_image_scores.size(1)//2
                attr_features_for_image = attr_features[:, :num_attr]
                attr_image_scores, neg_attr_image_scores = attr_image_scores[:, :num_attr], attr_image_scores[:, num_attr:]
                attr_sequence = attr_sequence[:, :num_attr]

            logits_loc = logit_scale * self.compute_sim_token_attr_visual(image_sequence, attr_features_for_image)

            # Compute loss for attributes
            gt_attr, loss_mask = self.format_attr_training(batch_data)
            if self.params["training"]["train_attr_image"]:
                if self.params["training"]["use_negative_attr"]:
                    loss_attr_image = self.compute_attr_image_neg_loss(gt_attr, loss_mask, torch.stack([neg_attr_image_scores, attr_image_scores], dim=2))
                    attr_image_scores = torch.softmax(torch.stack([attr_image_scores, neg_attr_image_scores], dim=0), dim=0)[0]
                else:
                    loss_attr_image = self.compute_attr_image_loss(gt_attr, loss_mask, attr_image_scores)
            if self.params["training"]["train_attr_linear"]:
                attr_scores_linear = self.model.compute_attr_linear_scores(image_features)
                loss_attr_linear = self.compute_attr_image_loss(gt_attr, loss_mask, attr_scores_linear)

            # Compute loss for classes
            if self.params["training"]["train_class_image"]:
                loss_class_image = self.loss_fn_ce(class_image_scores, gt_image)
            if self.params["training"]["train_class_linear"]:
                class_scores_linear = self.model.compute_class_linear_scores(image_features)
                loss_class_linear = self.loss_fn_ce(class_scores_linear, gt_image)

            # Compute loss for localization
            gt_loc = torch.flatten(batch_data["attr_location"], start_dim=2, end_dim=3).permute(0, 2, 1)
            if self.params["training"]["train_loc"]:
                loc_mask = ~(torch.sum(gt_loc, dim=1) == 0)
                loss_loc = self.loss_fn_bce(logits_loc.permute(1, 0, 2)[:, loc_mask], gt_loc.permute(1, 0, 2)[:, loc_mask].to(device=self.device, dtype=torch.float))
            if self.params["training"]["train_oracle_loc"]:
                gt_oracle_loc = torch.flatten(batch_data["oracle_attr_location"], start_dim=2, end_dim=3).permute(0, 2, 1)
                loc_mask = ~(torch.sum(gt_oracle_loc, dim=1) == 0)
                loss_loc_oracle = self.loss_fn_bce(logits_loc.permute(1, 0, 2)[:, loc_mask], gt_oracle_loc.permute(1, 0, 2)[:, loc_mask].to(device=self.device, dtype=torch.float))

            # Compute global loss
            losses = list()
            weights = self.params['training']['loss_weights']
            total_weight = 0
            for loss, weight in zip([loss_class_image, loss_attr_image, loss_loc, loss_class_linear, loss_attr_linear, loss_loc_oracle],
                                    [weights["class"], weights["attr"], weights["loc"], weights["proj_class"], weights["proj_attr"], weights["oracle_loc"]]):
                if loss is None:
                    continue
                losses.append(weight*loss)
                total_weight += weight
            losses = torch.stack(losses, dim=0)
            loss = torch.sum(losses)/total_weight

        # Backprop
        if update_weights:
            self.backward_loss(loss)
            if self.params["training"]["use_amp"]:
                self.model.float()
                self.step_optimizer()
                convert_weights(self.model)
            else:
                self.step_optimizer()
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            total_norm = 0
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.writer.add_scalar("train_gradient_norm", total_norm, self.last_step)
            self.zero_optimizer(set_to_none=True)
            self.model.clip.logit_scale.data = torch.clamp(self.model.clip.logit_scale.data, 0, 4.60517)  # e^4.60517 = 100
            self.acc_class_features = None
            self.acc_attr_features = None
        else:
            self.backward_loss(loss, retain_graph=True)

        values = {
            "loss": [loss.cpu().item(), ],
            "loss_class_image": [loss_class_image.cpu().item() if isinstance(loss_class_image, torch.Tensor) else None, ],
            "loss_attr_image": [loss_attr_image.cpu().item() if isinstance(loss_attr_image, torch.Tensor) else None, ],
            "loss_loc": [loss_loc.cpu().item() if isinstance(loss_loc, torch.Tensor) else None, ],
            "loss_loc_oracle": [loss_loc_oracle.cpu().item() if isinstance(loss_loc_oracle, torch.Tensor) else None, ],
            "gt_classif": torch.argmax(gt_image, dim=1).cpu(),
            "gt_attr": batch_data["attr"],
            "preds_loc": logits_loc.detach().cpu(),
            "gt_loc": gt_loc,
            "attr_classes": attr_classes,
            "certainty_masks": batch_data['certainty_masks'],
            "preds_classif": class_image_scores.detach().float().cpu(),
            "preds_attr": attr_image_scores.detach().float().cpu(),
        }

        if loss_class_linear is not None:
            values["loss_class_linear"] = [loss_class_linear.cpu().item(), ]
        if loss_attr_linear is not None:
            values["loss_attr_linear"] = [loss_attr_linear.cpu().item(), ]

        return values

    def format_attr_training(self, batch_data):
        certainty_masks = batch_data['certainty_masks'].to(self.device)
        gt_attr = batch_data["attr"].to(self.device).float()
        certainty_mask = certainty_masks[2]
        loss_mask = certainty_mask.bool()
        gt_attr = gt_attr * certainty_mask
        return gt_attr, loss_mask

    def compute_attr_image_loss(self, gt_attr, loss_mask, logits_attr_image):
        if torch.sum(loss_mask) == 0:
            return None
        return self.loss_fn_bce(logits_attr_image[loss_mask], gt_attr[loss_mask])

    def compute_attr_image_neg_loss(self, gt_attr, loss_mask, logits_attr_image):
        if torch.sum(loss_mask) == 0:
            return None
        return self.loss_fn_ce(logits_attr_image[loss_mask], gt_attr[loss_mask].long())

    def compute_sim_token_attr_visual(self, image_sequence, attr_emb):
        attr_sequence = attr_emb
        if attr_sequence.ndim == 2:
            attr_sequence = attr_sequence.unsqueeze(0)
        image_sequence = self.model.proj_visual_for_loc(image_sequence)
        image_sequence = self.model.normalize_embedding(image_sequence)
        if self.params["model"]["config"]["vision"][0] == "clip_vision":
            image_sequence = image_sequence[:, 1:]  # remove class token patch
        if attr_sequence.size(0) == 1:
            attr_sequence = torch.repeat_interleave(attr_sequence, dim=0, repeats=image_sequence.size(0))
        scores = torch.bmm(image_sequence, attr_sequence.permute(0, 2, 1))  # (B, P, N)
        return scores

    def get_attr_from_freezing(self, class_ids):
        attr_features = torch.stack([self.fine_tuning_features["attr_text"][int(i)] for i in class_ids], dim=0).to(self.device)
        attr_sequence = torch.stack([self.fine_tuning_features["attr_text_sequence"][int(i)] for i in class_ids], dim=0).to(self.device)
        attr_eot = torch.stack([self.fine_tuning_features["attr_text_eot_pos"][int(i)] for i in class_ids], dim=0).to(self.device)
        return self.compute_batch_attr_from_freezing(attr_features, attr_sequence, attr_eot)

    def compute_batch_attr_from_freezing(self, attr_features, attr_sequence, attr_eot):
        B, A, S, C = attr_sequence.size()
        attr_sequence = attr_sequence.view(B * A, S, C)
        if len(attr_features.size()) == 4:
            attr_features = attr_features.view(B*A, S, C)
        else:
            attr_features = attr_features.view(B*A, C)
        attr_eot = attr_eot.view(B*A)
        attr_features, attr_sequence = self.model.proj_text(attr_features, attr_sequence, attr_eot, features_from=self.return_after)
        attr_features = attr_features.view(B, A, -1)
        attr_sequence = attr_sequence.view(B, A, S, -1)
        return attr_features, attr_sequence

    def get_class_from_freezing(self):
        text_features = self.fine_tuning_features["class_text"].to(self.device)
        text_sequences = self.fine_tuning_features["class_text_sequence"].to(self.device)
        text_eots = self.fine_tuning_features["class_text_eot_pos"].to(self.device)
        return self.model.proj_text(text_features, text_sequences, text_eots, features_from=self.return_after)

    def preprocess_features_for_eval(self):
        device_for_attr = self.device if not self.use_loc else "cpu"
        self.eval_features["class_text"], _, _ = self.compute_all_classif_embedding(device=self.device)
        self.eval_features["attr_text"], self.eval_features["attr_text_sequence"], self.eval_features["attr_text_eot_pos"] = self.compute_attr_embedding_all(device=device_for_attr)
        keys = list(self.eval_features["attr_text"].keys())
        for key in keys:
            if key != 0:
                for name in ["attr_text", "attr_text_sequence", "attr_text_eot_pos"]:
                    del self.eval_features[name][key]
            else:
                for name in ["attr_text", "attr_text_sequence", "attr_text_eot_pos"]:
                    self.eval_features[name][key] = self.eval_features[name][key].to(self.device)

    def evaluate_classification(self, set_name, output=False, preprocess=True, output_name=None):
        self.model.eval()
        dataset = self.datasets[set_name]
        dataloader = self.dataloaders[set_name]
        self.metric_managers[set_name] = MetricManager(self.params["training"]["metric_names"][set_name], self.output_folder, self.datasets[set_name])
        num_iter_before_update_display = 1 if "num_iter_display_value_update" not in self.params["training"] else self.params["training"]["num_iter_display_value_update"]
        with torch.no_grad():
            if preprocess:
                self.preprocess_features_for_eval()
            with tqdm(total=len(dataset)) as pbar:
                pbar.set_description("Evaluation on {}".format(set_name))
                for num_batch, batch_data in enumerate(dataloader):
                    update_display = num_batch % num_iter_before_update_display == 0 or len(dataloader) == num_batch + 1
                    metrics = self.evaluate_classification_batch(set_name, batch_data)
                    metrics["sample_names"] = batch_data["sample_names"]
                    metrics["sample_ids"] = batch_data["sample_ids"]
                    metrics["nb_samples"] = [len(batch_data["sample_names"]), ]
                    self.metric_managers[set_name].add_batch_values(metrics)
                    if update_display:
                        display_values = self.metric_managers[set_name].get_display_values()
                        pbar.set_postfix(values=str(display_values))
                    pbar.update(len(batch_data["sample_names"]))
        if output:
            self.output_results(set_name, filename=output_name)
        return display_values

    def evaluate_classification_batch(self, set_name, batch_data):
        with autocast(enabled=self.params["training"]["use_amp"]):
            class_features = self.eval_features["class_text"]
            dtype = class_features.dtype if self.device != "cpu" else torch.float32
            class_features = class_features.to(dtype)
            attr_features = self.eval_features["attr_text"][0]
            num_attr = attr_features.size(0)

            if set_name in self.eval_features["image"]:
                image_features = self.eval_features["image"][set_name].index_select(0, torch.tensor(batch_data["sample_ids"], device=self.eval_features["image"][set_name].device)).to(device=self.device, dtype=dtype)
                if self.use_loc:
                    image_sequence = self.eval_features["image_sequence"][set_name].index_select(0, torch.tensor(batch_data["sample_ids"], device=self.eval_features["image_sequence"][set_name].device)).to(device=self.device, dtype=dtype)
            elif self.freeze[self.model_config["vision"][0]] and set_name in self.fine_tuning_features["images"]:
                image_features = torch.stack([self.fine_tuning_features["images"][set_name][i] for i in batch_data["sample_ids"]], dim=0).to(self.device)
                if self.use_loc:
                    image_sequence = torch.stack([self.fine_tuning_features["images_sequence"][set_name][i] for i in batch_data["sample_ids"]], dim=0).to(self.device)
                    image_features, image_sequence = self.model.proj_vision(image_features, image_sequence, features_from=self.return_after)
            else:
                x = batch_data["images"].to(device=self.device, dtype=dtype)
                image_features, image_sequence = self.compute_image_embedding(x)
                class_features = class_features.to(dtype=dtype)
            image_emb = self.model.normalize_embedding(image_features)

            class_image_scores = 100 * (image_emb @ class_features.T)
            attr_image_scores = 100 * (image_emb @ attr_features.T)

            if self.params["training"]["use_negative_attr"]:
                num_attr = num_attr // 2
                attr_image_scores, neg_attr_image_scores = attr_image_scores[:, :num_attr], attr_image_scores[:, num_attr:]
                attr_features = attr_features[:num_attr]
                attr_scores = torch.softmax(torch.stack([attr_image_scores, neg_attr_image_scores], dim=0), dim=0)[0].float()
            else:
                attr_scores = attr_image_scores.float()

            class_scores = class_image_scores.softmax(dim=-1).float()

            if self.use_loc:
                logits_loc = 100 * self.compute_sim_token_attr_visual(image_sequence, attr_features).cpu()
                gt_loc = torch.flatten(batch_data["attr_location"], start_dim=2, end_dim=3).permute(0, 2, 1)
            else:
                gt_loc = logits_loc = None

            class_linear_scores = self.model.compute_class_linear_scores(image_features).float() if self.model.classif_linear is not None else None
            attr_linear_score = self.model.compute_attr_linear_scores(image_features).float() if self.model.attr_linear is not None else None

        metrics = {
            "gt_classif": batch_data["class_ids"],
            "preds_classif": class_scores.cpu(),
            "preds_attr": attr_scores.cpu(),
            "gt_attr": batch_data["attr"],
            "attr_classes": batch_data["class_ids"],
            "certainty_masks":  batch_data['certainty_masks'],
            "preds_loc": logits_loc,
            "gt_loc": gt_loc,
        }

        if self.model.classif_linear is not None:
            metrics["accuracy_linear"] = class_linear_scores.cpu()
        if self.model.attr_linear is not None:
            metrics["attr_mAP_linear"] = attr_linear_score.cpu()

        return metrics

    def output_results(self, set_name, filename=None):
        filename = "predict_{}_{}.txt".format(set_name, self.last_epoch) if filename is None else filename
        res_file_path = os.path.join(self.output_folder, filename)
        metrics = self.metric_managers[set_name].get_display_values()
        with open(res_file_path, "w") as f:
            for metric_name in metrics.keys():
                f.write("{}: {}\n".format(metric_name, metrics[metric_name]))

    def compute_text_embedding(self, text_inputs, return_after=None):
        return self.model.encode_text(text_inputs, return_after=return_after)

    def compute_image_embedding(self, images, return_after=None):
        return self.model.encode_image(images, return_after=return_after)

    def compute_all_image_embedding(self, set_name, dtype=None, return_after=None, device=None, train=False):
        device = self.device if device is None else device
        dataset = self.datasets[set_name]
        dataloader = self.dataloaders["train_no_shuffle"] if set_name == "train" else self.dataloaders[set_name]
        image_features = list()
        sequences = list()
        with torch.set_grad_enabled(train):
            with tqdm(total=len(dataset)) as pbar:
                pbar.set_description("Computing image embedding for {}".format(set_name))
                for i, batch_data in enumerate(dataloader):
                    features, sequence = self.compute_image_embedding(batch_data["images"].to(device=self.device), return_after=return_after)
                    image_features.append(features.to(device=device, dtype=dtype))
                    if self.use_loc:
                        sequences.append(sequence.to(device=device, dtype=dtype))
                    pbar.update(len(batch_data["sample_names"]))
        return torch.cat(image_features, dim=0), torch.cat(sequences, dim=0) if self.use_loc else None

    def compute_all_classif_embedding(self, dtype=None, device="cpu", return_after=None, train=False):
        class_names = self.get_all_class_names().copy()
        if len(class_names) < 2000:
            features, sequences, eot_pos = self.compute_classif_embedding(class_names, dtype=dtype, device=device, return_after=return_after, train=train)
        else:
            features = list()
            sequences = list()
            eot_pos = list()
            for i in range(0, len(class_names), 2000):
                feat, seq, eot = self.compute_classif_embedding(class_names[i:i+2000], dtype=dtype, device=device, return_after=return_after, train=train)
                features.append(feat)
                sequences.append(seq)
                eot_pos.append(eot)
            features, sequences, eot_pos = torch.cat(features, dim=0), torch.cat(sequences, dim=0), torch.cat(eot_pos, dim=0)
        return features, sequences, eot_pos

    def compute_classif_embedding(self, class_names, dtype=None, device="cpu", return_after=None, train=False):
        with torch.set_grad_enabled(train):
            text_inputs = torch.cat([self.tokenize_fn(self.generate_class_text(class_name)) for class_name in class_names], dim=0).to(self.device)
            text_features, sequence, pos_eot = self.compute_text_embedding(text_inputs, return_after=return_after)
        return text_features.to(dtype=dtype, device=device), sequence.to(device=device), pos_eot.to(device=device)

    def compute_attr_embedding_all(self, dtype=None, device=None, return_after=None, train=False):
        class_ids = np.arange(self.get_num_classes()).tolist()
        return self.compute_attr_embedding(class_ids, dtype=dtype, device=device, return_after=return_after, train=train)

    def compute_attr_embedding(self, class_ids, dtype=None, device="cpu", return_after=None, train=False):
        embed = dict()
        sequences = dict()
        eot = dict()
        with torch.set_grad_enabled(train):
            text_features, sequence, pos_eot = self.compute_attr_embedding_by_class(0, return_after=return_after, dtype=dtype, device=device)
            for class_id in class_ids:
                sequences[class_id] = sequence
                eot[class_id] = pos_eot
                embed[class_id] = text_features
        return embed, sequences, eot

    def compute_attr_embedding_by_class(self, class_id, return_after=None, dtype=None, device=None):
        num_attr = self.get_num_attributes()
        if self.params["training"]["use_negative_attr"]:
            templates = ["{}", "no {}"]
            text_inputs = list()
            class_name = self.get_class_name(class_id)
            for t in templates:
                text_inputs.extend([t.format(self.get_attr_name(attr_id), class_name) for attr_id in range(num_attr)])
            text_features, sequence, pos_eot = self.compute_many_text_embeddings(text_inputs, return_after=return_after, device=device, dtype=dtype)
        else:
            text_inputs = [self.generate_attr_text(attr_id) for attr_id in range(num_attr)]
            text_features, sequence, pos_eot = self.compute_many_text_embeddings(text_inputs, return_after=return_after, device=device, dtype=dtype)

        return text_features, sequence, pos_eot

    def compute_many_text_embeddings(self, texts, return_after=None, device=None, dtype=None):
        max_batch_size = 1024
        text_inputs = torch.cat([self.tokenize_fn(t) for t in texts], dim=0).to(self.device)
        if len(texts) <= max_batch_size:
            text_features, sequence, pos_eot = self.compute_text_embedding(text_inputs, return_after=return_after)
        else:
            text_features = list()
            sequence = list()
            pos_eot = list()
            for i in range(0, len(texts), max_batch_size):
                text = text_inputs[i:i+max_batch_size]
                feat, seq, pos = self.compute_text_embedding(text, return_after=return_after)
                text_features.append(feat.to(dtype=dtype, device=device))
                sequence.append(seq.to(dtype=dtype, device=device))
                pos_eot.append(pos.to(device=device))
            text_features = torch.cat(text_features, dim=0)
            sequence = torch.cat(sequence, dim=0)
            pos_eot = torch.cat(pos_eot, dim=0)
        return text_features, sequence, pos_eot

    def get_class_attr_gt(self):
        return self.get_default_dataset().class_attr_gt

    def get_num_attributes(self):
        return self.get_default_dataset().num_attr

    def get_num_classes(self):
        return self.get_default_dataset().num_classes

    def get_default_dataset(self):
        if "train" in self.datasets:
            return self.datasets["train"]
        else:
            return self.datasets["test"]

    def get_attr_name(self, attr_id):
        return self.get_default_dataset().attr_labels["raw"][attr_id]

    def get_class_name(self, class_id):
        return self.get_default_dataset().class_names[class_id]

    def get_all_class_names(self):
        return list(self.get_default_dataset().class_names.values())

    def get_all_attr_names(self):
        return self.get_default_dataset().attr_labels["raw"]

    def generate_attr_text(self, attr_id):
        return self.get_attr_name(attr_id)

    def generate_class_text(self, class_name):
        return "a photo of a {}".format(class_name)

    def get_attr_features(self, set_name, batch_class_ids, device="cpu"):
        class_ids = batch_class_ids
        attr_labels = self.datasets[set_name].attr_labels["raw"]
        with torch.no_grad():
            list_str = list()
            for _ in class_ids:
                list_str.extend([self.tokenize_fn(self.generate_attr_text(attr_id)) for attr_id in range(len(attr_labels))])
            text_inputs = torch.cat(list_str, dim=0).to(self.device)
            text_features, _, _ = self.compute_text_embedding(text_inputs)
        text_features = text_features.resize(len(batch_class_ids), self.datasets[set_name].num_attr, text_features.size(-1)).to(device=device)
        return text_features

    def get_image_features(self, set_name, device="cpu"):
        dataset = self.datasets[set_name]
        dataloader = self.dataloaders[set_name]
        print("Computing image features")
        image_features = list()
        with torch.no_grad():
            with tqdm(total=len(dataset)) as pbar:
                for i, batch_data in enumerate(dataloader):
                    image_features, _, _ = self.compute_image_embedding(batch_data["images"].to(self.device))
                    image_features.append(image_features)
                    pbar.update(len(batch_data["sample_names"]))
        print("done")
        return torch.cat(image_features).to(device=device)

    def preprocess_text_embedding(self, name=None, save=True, load_if_exist=True, device="cpu"):
        fold_name = os.path.basename(self.output_folder)
        if name is None:
            save_path = os.path.join("Embedding", fold_name, "text_embedding_{}.pt".format(self.last_epoch))
        else:
            save_path = os.path.join("Embedding", name)
        if load_if_exist and os.path.exists(save_path):
            embed = torch.load(save_path, map_location=self.device)
            self.eval_features["class_text"], self.eval_features["attr_text"] = embed["classif_features"], embed["attr_features"]
            for k in self.eval_features["attr_text"].keys():
                self.eval_features["attr_text"][k] = self.eval_features["attr_text"][k].to(device=device)
        else:
            self.eval_features["class_text"], _, _ = self.compute_all_classif_embedding(device=device)
            self.eval_features["attr_text"], self.eval_features["attr_text_sequence"], self.eval_features["attr_text_eot_pos"] = self.compute_attr_embedding_all(device=device)
            if save:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if os.path.isfile(save_path):
                    print("Embedding file already exist, not saving")
                else:
                    torch.save({
                        "classif_features": self.eval_features["class_text"],
                        "attr_features": self.eval_features["attr_text"],
                        "attr_names": self.get_all_attr_names(),
                        "class_names": self.get_all_class_names()
                    }, save_path)

    def preprocess_image_embedding(self, set_name, save_filename=None, dtype=None, device="cpu", save=True, load_if_exist=True):
        save_path = os.path.join("Preprocess", "image_embedding_{}_{}.pt".format(save_filename, set_name))
        if load_if_exist and os.path.exists(save_path):
            self.eval_features["image"][set_name] = torch.load(save_path, map_location=device)
        else:
            self.eval_features["image"][set_name], self.eval_features["image_sequence"][set_name] = self.compute_all_image_embedding(set_name, dtype, device=device, train=False)
            if save:
                torch.save(self.eval_features["image"][set_name], save_path)
