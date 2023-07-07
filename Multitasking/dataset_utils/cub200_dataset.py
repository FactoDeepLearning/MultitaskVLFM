import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, RandomResizedCrop, InterpolationMode
import torchvision
from Multitasking.dataset_utils.generic_dataset import GenericDataset
import wget
import tarfile

BODYPART_MATCHING = {
    "back": ["back"],
    "bill": ["beak"],
    "belly": ["belly"],
    "breast": ["breast"],
    "crown": ["crown"],
    "eye": ["right eye", "left eye"],
    "forehead": ["forehead"],
    "head": ["forehead", "crown", "throat", "right eye", "left eye", "nape"],
    "leg": ["right leg", "left leg"],
    "nape": ["nape"],
    "primary": ["right wing", "left wing"],
    "tail": ["tail"],
    "throat": ["throat"],
    "under_tail": ["tail"],
    "underparts": ["throat", "breast", "belly"],
    "upper_tail": ["tail"],
    "upperparts": ["nape", "back"],
    "wing": ["right wing", "left wing"],
}


class CUBDataset(GenericDataset):

    def __init__(self, path, set_name, params):
        super(CUBDataset, self).__init__(path, set_name, params)
        self.collate_fn = BirdCollateFunction()

    def init_dataset(self):
        self.image_id_path_mapping = self.get_image_id_path_mapping()
        self.image_id_class_id_mapping = self.get_image_id_class_id_mapping()
        self.image_id_attr_mapping = self.get_image_id_attributes_mapping()
        self.attr_names = self.get_attribute_names()
        self.attr_labels = self.get_attribute_labels()
        self.num_attr = len(self.attr_labels["raw"])
        self.bodypart_centers = self.get_bodypart_center_points()
        self.bodypart_id_name_mapping = self.get_bodypart_id_name_mapping()
        self.bounding_boxes = self.get_bounding_boxes()
        self.oracle_matching_keys = {
            "back": ["back"],
            "bill": ["beak"],
            "belly": ["belly"],
            "breast": ["chest"],
            "crown": ["top of the head"],
            "eye": ["eyes"],
            "forehead": ["forehead"],
            "head": ["head", ],
            "leg": ["legs"],
            "nape": ["neck"],
            "primary": ["wings"],
            "tail": ["tail"],
            "throat": ["throat"],
            "under_tail": ["tail"],
            "underparts": ["throat", "chest", "belly"],
            "upper_tail": ["tail"],
            "upperparts": ["neck", "back"],
            "wing": ["wings", ],
            "birdshape": ["bird", ],
            "wingshape": ["wings", ],
            "size":  ["bird", ],
        }

    def before_preprocess_sample(self, sample, preprocess_fn):
        sample = self.preprocess_located_info(sample, preprocess_fn)
        return sample

    def after_preprocess_sample(self, sample):
        sample["attr_location"] = self.bodypart_centers_and_bb_to_attr_location(sample["bodypart_centers"], sample["bounding_box"], sample["image"].shape[1:])
        if self.params["load_oracle_segmentation"]:
            sample["oracle_attr_location"] = self.oracle_mask_to_patch_loc(sample)
        return sample

    def text_preprocessing(self):
        self.class_attr_gt = self.get_class_attribute_gt()

    def get_class_names(self):
        classes = dict()
        classes_path = os.path.join(self.path, "CUB_200_2011", "classes.txt")
        with open(classes_path) as f:
            lines = f.readlines()
        for line in lines:
            class_id, class_name = line.strip().split(" ")
            class_name = class_name.split(".")[1].replace("_", " ").lower()
            classes[int(class_id)-1] = class_name
        return classes

    def get_attribute_names(self):
        attr = dict()
        attr_path = os.path.join(self.path, "attributes.txt")
        with open(attr_path) as f:
            lines = f.readlines()
        for line in lines:
            attr_id, attr_name = line.strip().split(" ")
            attr[int(attr_id)-1] = attr_name
        return attr

    def get_image_id_attributes_mapping(self):
        attr = dict()
        attr_path = os.path.join(self.path, "CUB_200_2011", "attributes", "image_attribute_labels.txt")
        with open(attr_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.replace("  ", " ")
            try:
                image_id, attribute_id, attribute_value, attribute_certainty, time = line.strip().split(" ")
            except:
                image_id, attribute_id, attribute_value, attribute_certainty, _, time = line.strip().split(" ")
            image_id, attribute_id, attribute_value, attribute_certainty = int(image_id), int(attribute_id)-1, int(attribute_value), int(attribute_certainty)
            content = {
                    "value": attribute_value,
                    "certainty": attribute_certainty
                }
            if image_id in attr:
                attr[image_id][attribute_id] = content
            else:
                attr[image_id] = {
                    attribute_id: content
                }
        return attr

    def get_image_ids(self):
        if self.set_name == "train":
            set_name_id = "1"
        else:
            set_name_id = "0"
        split_path = os.path.join(self.path, "CUB_200_2011", "train_test_split.txt")
        with open(split_path) as f:
            lines = f.readlines()
        ids = [int(l.strip().split(" ")[0]) for l in lines if l.strip().split(" ")[1] == set_name_id]
        return ids

    def get_image_id_class_id_mapping(self):
        map = dict()
        mapping_path = os.path.join(self.path, "CUB_200_2011", "image_class_labels.txt")
        with open(mapping_path) as f:
            lines = f.readlines()
        for line in lines:
            id_img, id_class = line.strip().split(" ")
            map[int(id_img)] = int(id_class)-1
        return map

    def get_image_id_path_mapping(self):
        map = dict()
        mapping_path = os.path.join(self.path, "CUB_200_2011", "images.txt")
        with open(mapping_path) as f:
            lines = f.readlines()
        for line in lines:
            id, path = line.strip().split(" ")
            map[int(id)] = path
        return map

    def get_bodypart_id_name_mapping(self):
        map = dict()
        mapping_path = os.path.join(self.path, "CUB_200_2011", "parts", "parts.txt")
        with open(mapping_path) as f:
            lines = f.readlines()
        for line in lines:
            words = line.strip().split(" ")
            bodypart_id = int(words[0])
            name = " ".join(words[1:])
            map[bodypart_id] = name
        return map

    def get_bodypart_center_points(self):
        map = dict()
        filepath = os.path.join(self.path, "CUB_200_2011", "parts", "part_locs.txt")
        with open(filepath) as f:
            lines = f.readlines()
        for line in lines:
            id_image, id_part, x, y, visible = line.strip().split(" ")
            if int(id_image) not in map.keys():
                map[int(id_image)] = dict()
            map[int(id_image)][int(id_part)] = [bool(int(visible)), int(float(x)), int(float(y))]
        return map

    def get_bounding_boxes(self):
        map = dict()
        filepath = os.path.join(self.path, "CUB_200_2011", "bounding_boxes.txt")
        with open(filepath) as f:
            lines = f.readlines()
        for line in lines:
            id_image, x, y, width, height = line.strip().split(" ")
            id_image, x, y, width, height = int(id_image), int(float(x)), int(float(y)), int(float(width)), int(float(height))
            map[id_image] = [x, y, width, height]
        return map

    def get_image_by_id(self, id):
        return self.get_image_by_path(self.image_id_path_mapping[id])

    def get_image_by_path(self, img_path):
        path = os.path.join(self.path, "CUB_200_2011", "images", img_path)
        with Image.open(path) as pil_img:
            img = np.array(pil_img)
        return img

    def load_sample(self, img_id, sample_id):
        img = self.get_image_by_id(img_id)
        class_id = self.image_id_class_id_mapping[img_id]
        sample = {
            "image": img,
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "image_id": img_id,
            "image_name": self.image_id_path_mapping[img_id],
            "attr": self.get_attr_by_img_id(img_id),
            "certainty_masks": self.get_certainty_mask_by_img_id(img_id),
            "bodypart_centers": self.bodypart_centers[img_id],
            "bounding_box": self.bounding_boxes[img_id],
            "sample_id": sample_id
        }
        return sample

    def get_attr_by_img_id(self, img_id):
        d = self.image_id_attr_mapping[img_id]
        return torch.tensor([d[i]["value"]for i in range(self.num_attr)])

    def get_certainty_mask_by_img_id(self, img_id):
        thresholds = torch.tensor([1, 2, 3, 4])
        d = self.image_id_attr_mapping[img_id]
        return torch.stack([(d[i]["certainty"] >= thresholds).int() for i in range(self.num_attr)]).permute(1, 0)

    def get_class_attribute_gt(self, confidence=3):
        class_attr = dict()
        masks = dict()
        for s in self.samples:
            if s["class_id"] in class_attr:
                class_attr[s["class_id"]] = torch.cat([class_attr[s["class_id"]], s["attr"].unsqueeze(0)], dim=0)
                masks[s["class_id"]] = torch.cat([masks[s["class_id"]], s["certainty_masks"][confidence-1:confidence]], dim=0)
            else:
                class_attr[s["class_id"]] = (s["attr"]*s["certainty_masks"][confidence-1]).unsqueeze(0)
                masks[s["class_id"]] = s["certainty_masks"][confidence-1:confidence]
        class_attr_list = list()
        for i in range(self.num_classes):
            mask_sum = torch.sum(masks[i], dim=0)
            mask_no_value = mask_sum == 0
            class_attr_gt = torch.sum(class_attr[i]*masks[i], dim=0) / mask_sum
            class_attr_gt[mask_no_value] = -1
            class_attr_list.append(class_attr_gt)

        class_attr = torch.stack(class_attr_list, dim=0)
        return class_attr

    def update_info_resize(self, sample, init_size, new_size):
        ratio = (new_size[0]/init_size[0], new_size[1]/init_size[1])
        return self.adapt_location(sample, new_size, ratio=ratio)

    def update_info_crop(self, sample, init_size, new_size, start_h, start_w):
        shift = (start_h, start_w)
        return self.adapt_location(sample, new_size, shift=shift)

    def update_info_hflip(self, sample, size):
        return self.adapt_location(sample, size, hflip=True)

    def adapt_location(self, sample, max_size, ratio=(1, 1), shift=(0, 0), hflip=False):
        assert int(ratio != (1, 1)) + int(shift != (0, 0)) + int(hflip) <= 1  # check only one change maximum at once

        # Update bodypart centers
        bodypart_centers = sample["bodypart_centers"]
        for key in bodypart_centers:
            if bodypart_centers[key][0]:
                x, y = bodypart_centers[key][1:]
                if hflip:
                    new_x, new_y = max_size[1] - x, y
                else:
                    new_x, new_y = int(ratio[1] * x) - shift[1], int(ratio[0] * y) - shift[0]
                if 0 <= new_x <= max_size[1] and 0 <= new_y <= max_size[0]:
                    bodypart_centers[key][1:] = new_x, new_y
                else:
                    bodypart_centers[key] = False, 0, 0
        sample["bodypart_centers"] = bodypart_centers

        # Update bounding boxes
        x_min, y_min, width, height = sample["bounding_box"]
        x_min, x_max = [int(ratio[1] * x) - shift[1] for x in [x_min, x_min + width]]
        y_min, y_max = [int(ratio[0] * y) - shift[0] for y in [y_min, y_min + height]]
        if hflip:
            x_min, x_max = [max_size[1] - x for x in [x_min, x_max]]
        x_min, x_max, y_min, y_max = [max(0, t) for t in [x_min, x_max, y_min, y_max]]
        x_min, x_max = [min(t, max_size[1]) for t in [x_min, x_max]]
        y_min, y_max = [min(t, max_size[0]) for t in [y_min, y_max]]
        sample["bounding_box"] = [x_min, x_max, y_min, y_max]

        # Update oracle masks
        if self.params["load_oracle_segmentation"]:
            if hflip:
                sample["oracle_seg"] = torchvision.transforms.functional.hflip(sample["oracle_seg"])
            elif shift != (0, 0):
                sample["oracle_seg"] = sample["oracle_seg"][:, shift[0]:shift[0]+max_size[0], shift[1]:shift[1]+max_size[1]]
            else:
                sample["oracle_seg"] = Resize(max_size, InterpolationMode.NEAREST)(sample["oracle_seg"])
        return sample

    def preprocess_located_info(self, sample, preprocess_fn):
        image_shape = sample["image"].shape[:2]
        bodypart_centers = sample["bodypart_centers"]

        new_size = [t for t in preprocess_fn.transforms if isinstance(t, Resize) or isinstance(t, RandomResizedCrop)][0].size
        new_size = new_size if isinstance(new_size, int) else new_size[0]
        crop_size = [t for t in preprocess_fn.transforms if isinstance(t, CenterCrop) or isinstance(t, RandomResizedCrop)][0].size
        h, w = image_shape
        ratio_resize = new_size / min(h, w)
        new_h, new_w = int(ratio_resize * h), int(ratio_resize * w)
        diff_h = np.abs(new_h-crop_size[1])
        diff_w = np.abs(new_w-crop_size[0])
        new_h, new_w = crop_size

        for key in bodypart_centers:
            if bodypart_centers[key][0]:
                x, y = bodypart_centers[key][1:]
                new_x, new_y = int(ratio_resize * x), int(ratio_resize * y)
                if diff_h != 0:
                    new_y -= diff_h // 2
                if diff_w != 0:
                    new_x -= diff_w // 2
                new_x, new_y = min(new_x, new_w), min(new_y, new_h)
                bodypart_centers[key][1:] = new_x, new_y
        sample["bodypart_centers"] = bodypart_centers
        x_min, y_min, width, height = sample["bounding_box"]
        x_max, y_max = x_min + width, y_min + height
        x_min, x_max, y_min, y_max = [int(ratio_resize * t) for t in [x_min, x_max, y_min, y_max]]
        if diff_h != 0:
            y_min, y_max = y_min - diff_h // 2, y_max - diff_h // 2
        if diff_w != 0:
            x_min, x_max = x_min - diff_w // 2, x_max - diff_w // 2
        x_min, x_max, y_min, y_max = [max(0, t) for t in [x_min, x_max, y_min, y_max]]
        x_min, x_max = [min(t, new_w) for t in [x_min, x_max]]
        y_min, y_max = [min(t, new_h) for t in [y_min, y_max]]
        sample["bounding_box"] = [x_min, x_max, y_min, y_max]
        return sample

    def bodypart_centers_and_bb_to_attr_location(self, bodypart_centers, bounding_box, img_size):
        patch_size = self.params["patch_size"]
        dist_threshold = 0.75 * patch_size
        reversed_bodypart_id_mapping = {v: k for k, v in self.bodypart_id_name_mapping.items()}
        x_min, x_max, y_min, y_max = [t for t in bounding_box]
        width, height = x_max - x_min, y_max-y_min
        center_x, center_y = x_min + width / 2, y_min + height / 2
        attribute_location = list()
        for attr_id, attr in self.attr_names.items():
            patch_centers = torch.full((img_size[0] // patch_size, img_size[1] // patch_size), fill_value=patch_size, device="cpu", dtype=torch.float)
            patch_centers_x, patch_centers_y = torch.cumsum(patch_centers, dim=1) - patch_size//2, torch.cumsum(patch_centers, dim=0) - patch_size//2
            attr_name, attr_value = attr.split("::")
            if attr_name in ["has_shape", "has_size"]:
                dx = torch.abs(center_x - patch_centers_x) - 0.5 * width
                dx[dx < 0] = 0
                dy = torch.abs(center_y - patch_centers_y) - 0.5 * height
                dy[dy < 0] = 0
                attribute_location.append((torch.norm(torch.stack([dx, dy], dim=0), dim=0) <= dist_threshold).long())
                continue
            attr_words = attr_name.split("_") + attr_value.split("_")
            loc = torch.zeros((img_size[0] // patch_size, img_size[1] // patch_size), device="cpu", dtype=torch.long)
            for bodypart_name, list_parts in BODYPART_MATCHING.items():
                if bodypart_name in attr_words:
                    for part_name in list_parts:
                        bodypart_center = bodypart_centers[reversed_bodypart_id_mapping[part_name]]
                        is_visible, x, y = bodypart_center
                        if is_visible:
                            bodypart_loc = torch.norm(torch.stack([patch_centers_x-x, patch_centers_y-y], dim=0), dim=0) <= dist_threshold
                            loc = torch.logical_or(loc, bodypart_loc)
            attribute_location.append(loc.long())

        return torch.stack(attribute_location, dim=0)

    def get_attribute_labels(self):
        labels = list()
        labels_raw = list()
        for i, attr_name in enumerate(self.attr_names.values()):
            attr, value = attr_name.split("::")
            value = value.replace("_", " ")
            if attr.split("_")[-1] == "color":
                bodypart_name = " ".join(attr.split("_")[1:-1])
                labels.append(self.format_color_caption(bodypart_name, value))
                labels_raw.append(self.format_raw_color_caption(bodypart_name, value))
            elif attr.split("_")[-1] == "shape":
                bodypart_name = " ".join(attr.split("_")[1:-1])
                labels.append(self.format_shape_caption(bodypart_name, value))
                labels_raw.append(self.format_shape_caption(bodypart_name, value, mode="raw"))
            elif attr.split("_")[-1] == "length":
                bodypart_name = " ".join(attr.split("_")[1:-1])
                if value == "about the same as head":
                    labels.append("its {} is about the same length as its head".format(bodypart_name))
                    labels_raw.append("a {} about the same length as the head".format(bodypart_name))
                elif value == "longer than head":
                    labels.append("its {} is longer than its head".format(bodypart_name))
                    labels_raw.append("a {} longer than the head".format(bodypart_name))
                elif value == "shorter than head":
                    labels.append("its {} is shorter than its head".format(bodypart_name))
                    labels_raw.append("a {} shorter than the head".format(bodypart_name))
            elif attr.split("_")[-1] == "pattern":
                bodypart_name = " ".join(attr.split("_")[1:-1])
                labels.append(self.format_pattern_caption(bodypart_name, value, mode=None))
                labels_raw.append(self.format_pattern_caption(bodypart_name, value, mode="raw"))
            elif attr.split("_")[-1] == "size":
                labels.append("it is {}".format(value))
                labels_raw.append("a {} size".format(value))
        return {
                "raw": labels_raw,
                "formatted": labels
        }

    def get_singular_article(self, name):
        if name in ["iridescent", "olive", "orange"] +\
                ["all-purpose", "owl-like", "upland-ground-like", "upright-perching water-like"]:
            return "an"
        elif name in ["black", "blue", "buff", "brown", "green", "grey", "pink", "purple", "red", "rufous", "white", "yellow"]+\
                ["cone", "chicken-like-marsh", "curve", "dagger", "duck-like", "fan", "fork", "gull-like", "hawk-like", "hook",
                 "hummingbird-like", "long-legged-like", "needle", "notch", "perching-like", "pigeon-like", "point", "round", "sandpiper-like",
                 "seabird-hook", "spatulate", "specialize", "square", "swallow-like", "tree-clinging-like"]:
            return "a"
        else:
            raise NotImplementedError("unkwnown word: {}".format(name))

    def format_raw_color_caption(self, bodypart, color):
        caption_mapping_raw = {
            "underparts": "{} underparts",
            "upperparts": "{} upperparts",
            "throat": "@ {} throat",
            "eye": "{} eyes",
            "leg": "{} legs",
            "back": "@ {} back",
            "tail": "@ {} tail",
            "head": "@ {} head",
            "breast": "@ {} breast",
            "forehead": "@ {} forehead",
            "under tail": "@ {} under tail",
            "upper tail": "@ {} upper tail",
            "nape": "@ {} nape",
            "belly": "@ {} belly",
            "primary": "@ {} primary color",
            "bill": "@ {} bill",
            "crown": "@ {} crown",
            "wing": "{} wings"
        }
        caption = caption_mapping_raw[bodypart].format(color)
        caption = caption.replace("@", self.get_singular_article(color))
        return caption

    def format_color_caption(self, bodypart, color):
        caption_mapping = {
            "underparts": "its underparts are {}",
            "upperparts": "its upperparts are {}",
            "throat": "its throat is {}",
            "eye": "its eyes are {}",
            "leg": "its legs are {}",
            "back": "its back is {}",
            "tail": "its tail is {}",
            "under tail": "its under tail is {}",
            "upper tail": "its upper tail is {}",
            "head": "its head is {}",
            "breast": "its breast is {}",
            "forehead": "its forehead is {}",
            "nape": "its nape is {}",
            "belly": "its belly is {}",
            "primary": "its primary color is {}",
            "bill": "its bill is {}",
            "crown": "its crown is {}",
            "wing": "its wings are {}"
        }
        caption = caption_mapping[bodypart].format(color)
        return caption

    def format_shape_caption(self, bodypart, shape, mode=None):
        caption_mapping = {
            "curved (up or down)": "curve",
            "dagger": "dagger",
            "hooked": "hook",
            "needle": "needle",
            "hooked seabird": "seabird-hook",
            "spatulate": "spatulate",
            "all-purpose": "all-purpose",
            "cone": "cone",
            "specialized": "specialize",
            "forked": "fork",
            "rounded": "round",
            "notched": "notch",
            "fan-shaped": "fan",
            "pointed": "point",
            "squared": "square",
        }

        shape = shape.replace(" tail", "").replace("-wings", "")
        if bodypart == "":
            caption = "@ {} shape".format(shape)
        elif bodypart == "wing":
            caption = "{} wings".format(shape)
        else:
            shape = caption_mapping[shape]
            caption = "@ {}-shaped {}".format(shape, bodypart)
        if mode is None:
            caption = "it has {}".format(caption)
        elif mode == "raw":
            pass
        else:
            raise NotImplementedError("Unknown mode: {}".format(mode))
        if "@" in caption:
            caption = caption.replace("@", self.get_singular_article(shape))
        return caption

    def format_pattern_caption(self, bodypart, pattern, mode=None):
        pattern = pattern.replace("_pattern", "")
        pattern_mapping = {
            "solid": "a solid {}",
            "spotted": "a spotted {}",
            "striped": "a striped {}",
            "multi-colored": "a multi-colored {}",
            "malar": "malars",
            "crested": "a crest",
            "masked": "a masked face",
            "unique pattern": "a specific {}",
            "eyebrow": "eyebrows",
            "eyering": "eyerings",
            "plain": "a plain head",
            "eyeline": "eyelines",
            "capped": "a capped {}"
        }

        if mode is None:
            caption = "it has {}".format(pattern_mapping[pattern])
        elif mode == "raw":
            caption = pattern_mapping[pattern]
        else:
            raise NotImplementedError("Unknown mode: {}".format(mode))
        if "{}" in caption:
            caption = caption.format(bodypart)

        if bodypart == "wing":
            caption = caption.replace("wing", "wings").replace("a ", "")
        return caption


class BirdCollateFunction:

    def __init__(self):
        pass

    def __call__(self, batch_data):
        data = {
            "class_ids": torch.tensor([data["class_id"] for data in batch_data]),
            "class_names": [data["class_name"] for data in batch_data],
            "images": torch.stack([data["image"] for data in batch_data]),
            "sample_ids": [data["sample_id"] for data in batch_data],
            "sample_names": [data["image_name"] for data in batch_data],
            "attr": torch.stack([data["attr"] for data in batch_data]),
            "certainty_masks": torch.stack([data["certainty_masks"] for data in batch_data], dim=1),
            "attr_location": torch.stack([data["attr_location"] for data in batch_data], dim=0),
        }
        if "oracle_attr_location" in batch_data[0]:
            data["oracle_attr_location"] = torch.stack([data["oracle_attr_location"] for data in batch_data], dim=0)
        return data


def check_or_download_CUB200_dataset():
    dataset_path = "Datasets/CUB200/"
    tar_path = os.path.join(dataset_path, "CUB_200_2011.tgz")
    fold_path = os.path.join(dataset_path, "CUB_200_2011")
    if not os.path.exists(tar_path):
        os.makedirs(dataset_path, exist_ok=True)
        print("Downloading CUB200 dataset, this can take a moment...")
        wget.download("https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1", out=dataset_path)
        print("Download completed")
    if not os.path.exists(fold_path):
        tar = tarfile.open(tar_path)
        print("Extraction...")
        tar.extractall(path=dataset_path)
        tar.close()
        os.remove(tar_path)
        print("Completed")


def check_or_download_CUB200_oracle_seg():
    fold = "Oracle_seg"
    os.makedirs(fold, exist_ok=True)
    check_path = os.path.join(fold, "merged", "CUB200", "UIO", "Fish_Crow_0002_26072.npz")
    if not os.path.exists(check_path):
        print("Downloading oracle segmentation maps for CUB200 dataset, this can take a moment...")
        wget.download("https://zenodo.org/record/8124265/files/Oracle_segmentation_maps.tar.gz?download=1", out=fold)
        print("Download completed")
        tar_path = os.path.join(fold, "Oracle_segmentation_maps.tar.gz")
        tar = tarfile.open(tar_path)
        print("Extraction...")
        tar.extractall(path=fold)
        tar.close()
        os.remove(tar_path)
        print("Completed")