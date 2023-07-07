from torch.optim import Adam
from Multitasking.dataset_utils.cub200_dataset import CUBDataset, check_or_download_CUB200_dataset, check_or_download_CUB200_oracle_seg
from Multitasking.trainer import Trainer
from Multitasking.utils import seed_all, none_or_str, arg_to_bool
from Multitasking.models import check_or_download_model_weights
import torch
import argparse


seed_all(0)


def main(args):
    clip_model_name = args.model
    dataset_name = "CUB200"

    check_or_download_CUB200_dataset()

    # Get patch size from model name
    patch_size = clip_model_name.split("/")[1] if "/" in clip_model_name else None
    if "@" in patch_size:
        patch_size = patch_size.split("@")[0]
    patch_size = int(patch_size) if patch_size else None

    dataset_params = {
        "patch_size": patch_size if args.image_encoder is None else 32,
        "model_name": clip_model_name,
        "load_oracle_segmentation": args.train_oracle_loc,
        "oracle": args.oracle,
        "other_vision_encoder": args.image_encoder,
        "input_size": (224, 224) if args.image_encoder is None else (384, 384),
    }

    dataset_path = "./Datasets/{}".format(dataset_name)
    dataset_class = CUBDataset

    datasets = {
        "train": dataset_class(dataset_path, "train", dataset_params),
        "valid": dataset_class(dataset_path, "valid", dataset_params),
    }

    params = {
        "training": {
            "output_fold_name": "{}_{}".format(clip_model_name.replace("/", "-"), args.output_name),
            "load_weights": "last",
            "max_num_epochs": args.num_epochs,
            "eval_on_valid_interval": 2,
            "num_iter_display_value_update": 100,
            "gpu_index": "0",
            "use_amp": torch.cuda.is_available(),
            "batch_size": {
                "train": args.batch_size,
                "valid": 4*args.batch_size,
            },
            "gradient_acc": 200,
            "metric_names": {
                "train": ["loss", "loss_class_image", "loss_attr_image", "loss_loc",  "accuracy", "top5", "attr_mAP", "loc_mAP"],
                "valid": ["accuracy", "top5", "attr_mAP", "loc_mAP", ],
            },
            "optimizer": {
                    "class": Adam,
                    "args": {
                        "lr": 1e-5,
                        "amsgrad": False,
                        "betas": (0.9, 0.98),
                        "eps": 1e-6,
                        "weight_decay": 0.2
                    }
                },
            "train_class_image": args.train_class,
            "train_attr_image": args.train_attr,
            "train_loc": args.train_loc,
            "train_oracle_loc": args.train_oracle_loc,
            "train_class_linear": args.train_proj_class,
            "train_attr_linear": args.train_proj_attr,
            "loss_weights": {
                "class": args.weight_class,
                "attr": args.weight_attr,
                "loc": args.weight_loc,
                "proj_class": args.weight_proj_class,
                "proj_attr": args.weight_proj_attr,
                "oracle_loc": args.weight_oracle_loc,
            },
            "metric_to_focus": "accuracy",
            "expected_metric_value": "high",
            "use_negative_attr": args.neg_attributes,
        },
        "model": {
            "clip_model": clip_model_name,
            "classif_linear": False,
            "attr_linear": False,
            "input_size": dataset_params["input_size"],
            "config": {
                #  goal: [transformer_name, freeze, part to freeze]
                "vision": ["clip_vision", args.adapter_image, "backbone"],
                "class": ["clip_text", args.adapter_text, 'backbone'],
                "attr": ["clip_text", args.adapter_text, "backbone"],
            },
        }
    }
    if args.image_encoder is not None:
        params["model"]["config"]["vision"][0] = args.image_encoder

    if params["training"]["train_class_linear"]:
        params["training"]["metric_names"]["train"].append("loss_class_linear")
        params["training"]["metric_names"]["valid"].append("accuracy_linear")
        params["model"]["classif_linear"] = True

    if params["training"]["train_attr_linear"]:
        params["training"]["metric_names"]["train"].append("loss_attr_linear")
        params["training"]["metric_names"]["valid"].append("attr_mAP_linear")
        params["model"]["attr_linear"] = True

    if params["training"]["train_oracle_loc"]:
        params["training"]["metric_names"]["train"].append("loss_loc_oracle")

    trainer = Trainer(datasets, params)

    if args.train:
        trainer.train()

    if args.eval:
        trainer.free_memory()
        trainer.load_weights("last")
        trainer.evaluate_classification("valid", output=True)


parser = argparse.ArgumentParser()
parser.add_argument("--output-name", type=str, default="my_expe")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--oracle", type=str, default="OFA")

parser.add_argument("--model", type=str, default="open-ViT-L/14")
parser.add_argument("--image-encoder", type=none_or_str, default="swin_vision")
parser.add_argument("--load-pretrain", type=arg_to_bool, default=False)

parser.add_argument("--train", type=arg_to_bool, default=True)
parser.add_argument("--eval", type=arg_to_bool, default=True)

parser.add_argument("--train-class", type=arg_to_bool, default=True)
parser.add_argument("--train-attr", type=arg_to_bool, default=True)
parser.add_argument("--train-loc", type=arg_to_bool, default=True)
parser.add_argument("--train-oracle-loc", type=arg_to_bool, default=False)
parser.add_argument("--train-proj-class", type=arg_to_bool, default=True)
parser.add_argument("--train-proj-attr", type=arg_to_bool, default=False)

parser.add_argument("--weight-class", type=int, default=1)
parser.add_argument("--weight-attr", type=int, default=1)
parser.add_argument("--weight-loc", type=int, default=1)
parser.add_argument("--weight-oracle-loc", type=int, default=1)
parser.add_argument("--weight-proj-attr", type=int, default=1)
parser.add_argument("--weight-proj-class", type=int, default=1)

parser.add_argument("--adapter-image", type=arg_to_bool, default=False)
parser.add_argument("--adapter-text", type=arg_to_bool, default=True)

parser.add_argument("--neg-attributes", type=arg_to_bool, default=True)

args = parser.parse_args()

if args.load_pretrain:
    args.output_name = "swin_clip_text_finetuned"
    check_or_download_model_weights()

if args.train_oracle_loc:
    check_or_download_CUB200_oracle_seg()

main(args)
