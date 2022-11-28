import torch
from transformers import AlbertTokenizerFast, AutoModel
from retriever import DualEncoder
import sys
import os
import random
import bisect
from datetime import datetime
import numpy as np


def set_seeds(seed: int):
    """Fix the seed for reproducibility.

    Args:
        seed (int): seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_range_excluding(n, k, excluding):
    """Sample elements for a list, excluding certain elements."""
    skips = [j - i for i, j in enumerate(sorted(set(excluding)))]
    sampled = random.sample(range(n - len(skips)), k)
    return [i + bisect.bisect_right(skips, i) for i in sampled]


def strtime(datetime_checkpoint):
    """Formate time."""
    diff = datetime.now() - datetime_checkpoint
    return str(diff).split(".", maxsplit=1)[0]  # Ignore below seconds


class Logger(object):
    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += "+"

    def log(self, string, newline=True, force=False):
        if self.on or force:
            with open(self.log_path, "a") as logf:
                logf.write(string)
                if newline:
                    logf.write("\n")

            sys.stdout.write(string)
            if newline:
                sys.stdout.write("\n")
            sys.stdout.flush()


def load_tokenizer(tokenizer_name):
    if tokenizer_name == "Albert":
        tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")
    else:
        print("Unknwown model name")
    special_tokens_dict = {
        "additional_special_tokens": ["[Ent]", "[/Ent]", "[Inter]"]  # , "'", "|", "@"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def load_model(config, tokenizer, init, device):
    """Load the dual encoder."""
    if config["encoder_name"] == "Albert":
        core_encoder = AutoModel.from_pretrained("albert-base-v2")
    core_encoder.resize_token_embeddings(len(tokenizer))
    model = DualEncoder(core_encoder, "dots" in config)
    if not init:
        state_dict = (
            torch.load(config["init_model_path"])
            if device.type == "cuda"
            else torch.load(config["init_model_path"], map_location=torch.device("cpu"))
        )
        model.load_state_dict(state_dict["sd"])
    return model


def type_contradiction(types_a, types_b):
    """Check for type contradiction between entities."""
    t_a = set(types_a)
    t_b = set(types_b)
    if t_a.issubset(t_b) or t_b.issubset(t_a):
        return False
    return True
