"""Train a classifier between 2 texts"""

import pickle
from train_retriever import configure_optimizer_simple, count_parameters, set_seeds
from utils import load_tokenizer
from transformers import AlbertForSequenceClassification
import os
import json
import argparse
import torch
import numpy as np
import wandb
from utils import Logger
from torch.utils.data import Dataset, DataLoader

PATH_CONFIG = "data/models/LINK/MERIT/classifier_clean_pretrained_new_weights.json"


def flat_metrics(preds, labels):
    pred_flat = np.argmax(preds.cpu().detach().numpy(), axis=1).flatten()
    labels_flat = labels.cpu().detach().numpy().flatten()
    non_zero_pred = np.nonzero(pred_flat)[0]
    non_zero_labels = np.nonzero(labels_flat)[0]
    pos = len(non_zero_labels)
    tp_fp = len(non_zero_pred)
    tp = len(np.intersect1d(non_zero_pred, non_zero_labels))
    print(pred_flat, labels_flat, pos, tp_fp, tp)
    return pos, tp_fp, tp


def evaluate(val_loader, model, device):
    # return modified hard recall@k, lrap and recall@K
    # hard recall: predict successfully if all labels are predicted
    #  recall: micro over passages
    model.eval()
    tt_tp, tt_tp_fp, tt_pos = 0, 0, 0
    for batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_masks, labels = batch
        kwargs = {"input_ids": input_ids, "attention_mask": input_masks}
        with torch.no_grad():
            output = model(**kwargs)
        pos, tp_fp, tp = flat_metrics(output[0], labels)
        tt_tp += tp
        tt_tp_fp += tp_fp
        tt_pos += pos
    precision = 0 if tt_tp_fp == 0 else tt_tp / tt_tp_fp
    recall = 0 if tt_pos == 0 else tt_tp / tt_pos
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return (precision, recall, f1)


def load_set(tokenizer, config, set_type):
    with open(config["path_classif_samples"] + set_type + ".pickle", "rb") as file:
        samples = pickle.load(file)
    sample_set = []
    for ref_ent, kb_ent, same in samples:
        texts_ids = (
            ref_ent["text"][:-1]
            + [tokenizer.convert_tokens_to_ids("[Inter]")]
            + kb_ent["text"][1:]
        )
        print(ref_ent["ent"], kb_ent["ent"], same)
        assert len(ref_ent["text"]) <= 256
        assert len(kb_ent["text"]) <= 256
        texts_ids = (
            texts_ids + [tokenizer.pad_token_id] * (config["max_len"] - len(texts_ids))
        )[: config["max_len"]]
        mask = ([1] * len(texts_ids) + [0] * (config["max_len"] - len(texts_ids)))[
            : config["max_len"]
        ]
        sample_set.append(
            [
                torch.from_numpy(np.asarray(texts_ids).astype("long")),
                torch.from_numpy(np.asarray(mask).astype("long")),
                torch.tensor(same).long(),
            ]
        )
    return sample_set


class SequencesSet(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


def main(config):
    # configure loggers
    wandb.init(project="Classifier", name="Classifier clean resumed")
    logger = Logger(config["path_log"], on=True)

    # set_seeds(config)

    tokenizer = load_tokenizer("Albert")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlbertForSequenceClassification.from_pretrained(
        "albert-base-v2", num_labels=2
    )
    model.resize_token_embeddings(len(tokenizer))
    print(model.config)
    train_set = load_set(tokenizer, config, "train")
    val_set = load_set(tokenizer, config, "val")
    train_loader = DataLoader(train_set, config["B"], shuffle=True)
    val_loader = DataLoader(val_set, config["B"], shuffle=True)

    if not config["init"]:
        state_dict = (
            torch.load(config["init_model_path"])
            if device.type == "cuda"
            else torch.load(config["init_model_path"], map_location=torch.device("cpu"))
        )
        model.load_state_dict(state_dict["sd"])
        best_val_perf = 0.8
    else:
        best_val_perf = float("-inf")
    # configure optimizer
    (
        optimizer,
        scheduler,
        num_train_steps,
        num_warmup_steps,
    ) = configure_optimizer_simple(config, model, len(train_set))
    model.to(device)
    effective_bsz = config["B"] * config["gradient_accumulation_steps"]

    # train
    logger.log("***** train *****")
    logger.log("# train samples: {:d}".format(len(train_set)))
    logger.log("# val samples: {:d}".format(len(val_set)))
    logger.log("# epochs: {:d}".format(config["epochs"]))
    logger.log(" batch size : {:d}".format(config["B"]))
    logger.log(
        " gradient accumulation steps {:d}"
        "".format(config["gradient_accumulation_steps"])
    )
    logger.log(
        " effective training batch size with accumulation: {:d}"
        "".format(effective_bsz)
    )
    logger.log(" # training steps: {:d}".format(num_train_steps))
    logger.log(" # warmup steps: {:d}".format(num_warmup_steps))
    logger.log(" learning rate: {:g}".format(config["lr"]))
    logger.log(" # parameters: {:d}".format(count_parameters(model)))

    model.train()
    step_num = 0
    start_epoch = 1
    model.zero_grad()
    loss = torch.nn.MSELoss()
    for epoch in range(start_epoch, config["epochs"] + 1):
        logger.log("\nEpoch {:d}".format(epoch))
        for batch in train_loader:
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, labels = batch
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": input_masks,
                "labels": labels,
            }
            output = model(**kwargs)
            loss = output[0]
            batch_loss = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            step_num += 1
            wandb.log({"step": step_num, "epoch": epoch, "loss": batch_loss})

        precision, recall, f1 = evaluate(val_loader, model, device)
        wandb.log(
            {
                "epochs": epoch,
                "f1": f1,
                "recall": recall,
                "precision": precision,
            }
        )
        if f1 >= best_val_perf:
            current_best = f1
            logger.log(
                "------- new best val perf: {:g} --> {:g} "
                "".format(best_val_perf, current_best)
            )
            best_val_perf = current_best
            torch.save(
                {
                    "opt": args,
                    "sd": model.state_dict(),
                    "perf": best_val_perf,
                    "epoch": epoch,
                    "opt_sd": optimizer.state_dict(),
                    "scheduler_sd": scheduler.state_dict(),
                    "step_num": step_num,
                },
                config["model_path"],
            )
        else:
            logger.log("")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=PATH_CONFIG,
        help="Path to the config file for training",
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    # Set environment variables before all else.
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]  # Sets torch.cuda behavior
    main(config)
