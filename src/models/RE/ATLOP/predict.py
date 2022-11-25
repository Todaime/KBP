"""Script to predict the relations for the processing chain."""

import argparse
import json
import os
import torch


from train import report
from model import DocREModel
from prepro import read_dwie
from transformers import AutoConfig, AutoModel, AutoTokenizer
from apex import amp


PATH_DWIE_RE_ATLOP_PREDS = "data/DWIE/RE/ATLOP/predictions"
PATH_DWIE_RE_ATLOP = "data/DWIE/RE/ATLOP/test"
PATH_DWIE_RE_ATLOP_MODEL = "data/models/RE/ATLOP/atlop.pt"


def main():
    """_summary_"""

    os.makedirs(
        os.path.join(PATH_DWIE_RE_ATLOP_PREDS),
        exist_ok=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params.n_gpu = torch.cuda.device_count()
    params.device = device

    config = AutoConfig.from_pretrained(
        params.config_name if params.config_name else params.model_name_or_path,
        num_labels=params.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        params.model_name_or_path,
    )

    model = AutoModel.from_pretrained(
        params.model_name_or_path,
        from_tf=bool(".ckpt" in params.model_name_or_path),
        config=config,
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = "bert"

    model = DocREModel(config, model, num_labels=params.num_labels)
    model.to(0)
    model = amp.initialize(model, opt_level="O1", verbosity=0)
    model.load_state_dict(torch.load(params.load_path))
    for filename in os.listdir(PATH_DWIE_RE_ATLOP):
        test_features = read_dwie(
            os.path.join(PATH_DWIE_RE_ATLOP, filename),
            tokenizer,
            max_seq_length=params.max_seq_length,
        )

        pred = report(params, model, test_features, filter=params.filter)
        with open(
            os.path.join(PATH_DWIE_RE_ATLOP, filename), "r", encoding="utf-8"
        ) as file:
            data = json.load(file)[0]
        if pred is not None:
            data["relations"] = [
                {"h": rel["h_idx"], "t": rel["t_idx"], "r": rel["r"]} for rel in pred
            ]

        with open(
            os.path.join(
                PATH_DWIE_RE_ATLOP_PREDS,
                filename,
            ),
            "w",
            encoding="UTF-8",
        ) as completed_file:
            json.dump(data, completed_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", default=PATH_DWIE_RE_ATLOP, type=str)
    parser.add_argument("--output_path", default=PATH_DWIE_RE_ATLOP_PREDS, type=str)
    parser.add_argument("--load_path", default=PATH_DWIE_RE_ATLOP_MODEL, type=str)
    parser.add_argument("--filter", action="store_true", default=False)
    parser.add_argument("--config_name", default="")
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int, help="Batch size.")
    parser.add_argument("--num_labels", default=4, type=int)
    parser.add_argument("--num_class", type=int, default=46, help="Number of relation.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased")
    params = parser.parse_args()
    main()
