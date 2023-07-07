import os
import copy
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Normalize
from torchvision.transforms import Compose, RandomAdjustSharpness, ToTensor, InterpolationMode
from torch.nn import Module
from tqdm import tqdm
from PIL import Image

from Multitasking.utils import rand, get_tokenizer, randint


class GenericDataset(Dataset):

    def __init__(self, path, set_name, params):
        self.params = params
        self.path = path
        self.set_name = set_name
        self.attr_from = dict()
        self.collate_fn = GenericCollateFunction()
        self.samples = None
        self.preprocess_fn = None
        self.input_size = self.params["input_size"]
        self.norm_fn = None
        self.image_id_class_id_mapping = None
        self.oracle_matching_keys = None
        self.init_dataset()
        self.tokenizer = get_tokenizer(self.params["model_name"])
        self.open_tokenizer = "open" in self.params["model_name"]
        self.dataset_name = os.path.basename(self.path)
        self.model_name = self.params["model_name"].replace("/", "-")
        if self.params["other_vision_encoder"] is None:
            archi_name = self.model_name
        else:
            archi_name = "{}_{}".format(self.model_name, self.params["other_vision_encoder"])
        self.preprocess_foldpath = os.path.join("Preprocess", self.dataset_name, archi_name, self.set_name)
        self.class_names = self.get_class_names()
        self.num_classes = len(self.class_names)

        self.samples = self.load_samples()

        self.image_preprocess_function = Compose([
            ToTensor(),
            ToRGB(),
        ])

        self.da_function = Compose([
            RandomAdjustSharpness(sharpness_factor=1.5, p=1)
        ])

        self.swin_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.text_preprocessing()

    def init_dataset(self):
        pass

    def get_class_names(self):
        pass

    def get_image_by_id(self, img_id):
        pass

    def get_image_ids(self):
        pass

    def update_info_resize(self, sample, init_size, new_size):
        pass

    def update_info_crop(self, sample, init_size, new_size, start_h, start_w):
        pass

    def update_info_hflip(self, sample, size):
        pass

    def before_preprocess_sample(self, sample, preprocess_fn):
        return sample

    def after_preprocess_sample(self, sample):
        return sample

    def text_preprocessing(self):
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])

        # load pseudolabel for localization if oracle (UIO/OFA) is used
        if self.params["load_oracle_segmentation"]:
            sample["oracle_seg"] = self.load_oracle_segmentation(sample)

        # IF CLIP IMAGE ENCODER
        if self.params["other_vision_encoder"] is None:
            self.before_preprocess_sample(sample, self.preprocess_fn)
            sample["image"] = self.preprocess_fn(Image.fromarray(sample["image"]))
            if self.set_name == "train":
                sample = self.apply_custom_da(sample)
            sample = self.after_preprocess_sample(sample)
            sample["image"] = self.norm_fn(sample["image"])
        # IF SWIN IMAGE ENCODER
        else:
            sample["image"] = self.image_preprocess_function(sample["image"])
            sample = self.resize(sample, (510, 510))
            if self.set_name == "train":
                sample = self.apply_custom_da(sample)
                sample = self.random_crop(sample, self.input_size)
            else:
                sample = self.center_crop(sample, self.input_size)
            sample = self.after_preprocess_sample(sample)
            sample["image"] = self.swin_normalize(sample["image"])

        return sample

    def denormalize(self, image):
        if self.params["other_vision_encoder"] is None:
            norm_fn = self.norm_fn
        else:
            norm_fn = self.swin_normalize
        return ((image * torch.tensor(norm_fn.std).view(3, 1, 1) + torch.tensor(norm_fn.mean).view(3, 1, 1)) * 255).to(torch.int)

    def apply_custom_da(self, sample):
        sample["image"] = self.da_function(sample["image"])
        if rand() < 0.1:
            sample["image"] = torchvision.transforms.functional.hflip(sample["image"])
            self.update_info_hflip(sample, sample["image"].shape[1:])
        return sample

    def random_crop(self, sample, crop_size):
        h, w = sample["image"].shape[1:]
        diff_h = h - crop_size[0]
        diff_w = w - crop_size[1]
        assert diff_w >= 0 and diff_h >= 0
        start_h = randint(0, diff_h + 1)
        start_w = randint(0, diff_w + 1)
        return self.crop(sample, start_h, start_w, crop_size[0], crop_size[1])

    def center_crop(self, sample, crop_size):
        h, w = sample["image"].shape[1:]
        diff_h = h - crop_size[0]
        diff_w = w - crop_size[1]
        assert diff_w >= 0 and diff_h >= 0
        start_h = diff_h // 2
        start_w = diff_w // 2
        return self.crop(sample, start_h, start_w, crop_size[0], crop_size[1])

    def crop(self, sample, start_h, start_w, crop_height, crop_width):
        h, w = sample["image"].shape[1:]
        sample["image"] = sample["image"][:, start_h:start_h+crop_height, start_w:start_w+crop_width]
        self.update_info_crop(sample, (h, w), (crop_height, crop_width), start_h, start_w)
        return sample

    def resize(self, sample, new_size):
        h, w = sample["image"].shape[1:]
        sample["image"] = Resize(new_size, InterpolationMode.BILINEAR, antialias=True)(sample["image"])
        self.update_info_resize(sample, (h, w), new_size)
        return sample

    def load_samples(self):
        samples = list()
        image_ids = self.get_image_ids()
        bar = tqdm(image_ids)
        bar.set_description("Loading {} samples".format(self.set_name))
        for i, img_id in enumerate(bar):
            samples.append(self.load_sample(img_id, i))
        return samples

    def load_sample(self, img_id, sample_id):
        img = self.get_image_by_id(img_id)
        class_id = self.image_id_class_id_mapping[img_id]
        sample = {
            "image": img,
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "image_id": img_id,
            "sample_id": sample_id,
            "image_name": img_id,
        }
        return sample

    def set_preprocess(self, preprocess_fn):
        self.preprocess_fn = Compose([tr for tr in preprocess_fn.transforms if not isinstance(tr, Normalize)])
        self.norm_fn = [tr for tr in preprocess_fn.transforms if isinstance(tr, Normalize)][0]

    def load_oracle_segmentation(self, sample):
        items = list()
        for i in self.oracle_matching_keys.values():
            items.extend(i)
        items = list(np.unique(items))

        # load segmentation masks
        image_name = os.path.basename(sample["image_name"]).split(".")[0]
        size, shape = (sample["image"][:, :, 0].size, sample["image"][:, :, 0].shape) if len(sample["image"].shape) == 3 else (sample["image"].size, sample["image"].shape)
        segmentations_maps = dict()
        filepath = os.path.join("Oracle_seg", "merged", self.dataset_name, self.params["oracle"], "{}.npz".format(image_name))
        masks = torch.load(filepath)
        for item in items:
            segmentations_maps[item] = torch.tensor(np.unpackbits(masks[item], count=size).reshape(shape).view(bool), dtype=torch.bool)

        # generate mask per attribute
        masks = list()
        for i in range(self.num_attr):
            attr_name = self.attr_names[i].split("::")[0]
            attr_words = attr_name\
                .replace("has_shape", "birdshape")\
                .replace("-", " ").replace("_", " ").split(" ")
            mask = torch.zeros((sample["image"].shape[:2]), dtype=torch.bool)
            for word in attr_words:
                if word in segmentations_maps.keys():
                    mask = torch.logical_or(mask, segmentations_maps[word])
                if word not in self.oracle_matching_keys:
                    continue
                for key in self.oracle_matching_keys[word]:
                    mask = torch.logical_or(mask, segmentations_maps[key])
            masks.append(mask)
        return torch.stack(masks, dim=0)

    def oracle_mask_to_patch_loc(self, sample):
        image_size = sample["image"].shape[-2:]
        patch_size = self.params["patch_size"]
        num_patches = image_size[0] // patch_size, image_size[1] // patch_size
        masks = sample["oracle_seg"]
        masks = masks.reshape(masks.size(0), num_patches[0], patch_size, num_patches[1], patch_size).permute(0, 1, 3, 2, 4)
        return torch.sum(masks, dim=[3, 4]) > 0.25*patch_size**2


class GenericCollateFunction:

    def __init__(self):
        pass

    def __call__(self, batch_data):
        data = {
            "class_ids": torch.tensor([data["class_id"] for data in batch_data]),
            "class_names": [data["class_name"] for data in batch_data],
            "images": torch.stack([data["image"] for data in batch_data]),
            "sample_ids": [data["sample_id"] for data in batch_data],
            "sample_names": [data["image_name"] for data in batch_data],
        }
        if "oracle_attr_location" in batch_data[0]:
            data["oracle_attr_location"] = torch.stack([data["oracle_attr_location"] for data in batch_data], dim=0)
        return data


# preprocessing
class ToRGB(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor) and x.size(0) in [1, 3]
        if x.size(0) == 1:
            x = torch.cat([x, x, x], dim=0)
        return x

