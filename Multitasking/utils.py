import torch
import numpy as np
import random
import os
from Multitasking.models import OPEN_MODELS
from clip.simple_tokenizer import SimpleTokenizer
from OPENCLIP.factory import get_tokenizer as open_get_tokenizer


def randint(low, high):
    """
    call torch.randint to preserve random among dataloader workers
    """
    return int(torch.randint(low, high, (1, )))


def rand():
    """
    call torch.rand to preserve random among dataloader workers
    """
    return float(torch.rand((1, )))


def pad_images(data, padding_value):
    """
    data: list of numpy array
    """
    x_lengths = [x.shape[0] for x in data]
    y_lengths = [x.shape[1] for x in data]
    longest_x = max(x_lengths)
    longest_y = max(y_lengths)
    padded_data = np.ones((len(data), longest_x, longest_y, data[0].shape[2])) * padding_value
    for i, xy_len in enumerate(zip(x_lengths, y_lengths)):
        x_len, y_len = xy_len
        padded_data[i, :x_len, :y_len, ...] = data[i]
    return padded_data


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_tokenizer(model_name):
    if "open-" in model_name:
        return open_get_tokenizer(OPEN_MODELS[model_name].replace("open-", ""), cache_dir="Tokenizers").tokenizer
    return SimpleTokenizer()


def arg_to_bool(arg):
    return arg.lower() == "true"


def none_or_str(value):
    if value.lower() == 'none':
        return None
    return value