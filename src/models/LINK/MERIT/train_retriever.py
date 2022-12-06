"""Train a modele to retrieve similar entities from one identified in an input text."""
import itertools
import os
import argparse
import random
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn

from data_retriever import (
    load_data,
    get_loaders,
    get_embeddings,
    get_hard_negative,
    get_labels,
    get_entity_map,
    get_loader_from_candidates,
)

from transformers import get_constant_schedule

from utils import Logger, load_model, load_tokenizer, strtime
from sklearn.metrics import label_ranking_average_precision_score

import wandb
from apex import amp


PATH_CONFIG = "data/models/LINK/MERIT/config_clean.json"


def set_seeds(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])


def configure_optimizer_simple(config, model, num_train_examples):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    num_train_steps = int(
        num_train_examples
        / config["B"]
        / config["gradient_accumulation_steps"]
        * config["epochs"]
    )
    num_warmup_steps = 0

    scheduler = get_constant_schedule(optimizer)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def evaluate(scores_k, top_k, labels):
    # return modified hard recall@k, lrap and recall@K
    # hard recall: predict successfully if all labels are predicted
    #  recall: micro over passages
    nb_samples = len(labels)
    r_k = 0
    y_trues = []
    num_labels = 0
    num_hits = 0
    nb_simple_hit = 0
    preds = []
    assert len(labels) == len(top_k)
    for i, ent_labels in enumerate(labels):
        pred = top_k[i]
        preds.append(pred)
        r_k += set(ent_labels).issubset(set(pred[:50]))
        y_trues.append(np.in1d(pred[:50], ent_labels))
        num_labels += len(set(ent_labels))
        num_hits += len(set(ent_labels).intersection(set(pred[:50])))
        nb_simple_hit += (
            1 if len(set(ent_labels).intersection(set(pred[:10]))) > 0 else 0
        )
    r_k /= nb_samples
    h_k = num_hits / num_labels
    nb_simple_hit /= nb_samples
    y_trues = np.vstack(y_trues)
    lrap = label_ranking_average_precision_score(y_trues, scores_k)
    return r_k, lrap, h_k, nb_simple_hit


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config):
    wandb.init(project="retriever", name="Retriever Clean Train/Val/Test")
    logger = Logger(config["path_log"], on=True)
    set_seeds(config)

    # configure logger
    if config["init"]:
        best_val_perf = float("-inf")
        best_val_perf_hard = float("-inf")
        best_val_perf_simple = float("-inf")
    else:
        best_val_perf = 0.0
        best_val_perf_hard = 0.0
        best_val_perf_simple = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.log(f"Using device: {str(device)}", force=True)

    # load data and initialize model and dataset

    (
        samples_train,
        samples_val,
        _,
        entities,
        ent_to_indexes,
        nb_train_samples,
    ) = load_data(config["data_dir"], logger)
    logger.log("number of entities {:d}".format(len(entities)))

    # get model and tokenizer
    tokenizer = load_tokenizer(config["model_name"])
    model = load_model(config, tokenizer, config["init"], device)

    # configure optimizer
    (
        optimizer,
        scheduler,
        num_train_steps,
        num_warmup_steps,
    ) = configure_optimizer_simple(config, model, len(samples_train))

    model.to(device)
    # if config["fp16"]:
    # model, optimizer = amp.initialize(
    #    model, optimizer, opt_level=config["fp16_opt_level"]
    # )

    dp = torch.cuda.device_count() > 1
    if dp:
        logger.log(
            "Data parallel across {:d} GPUs {:s}"
            "".format(len(config["gpus"].split(",")), config["gpus"])
        )
        model = nn.DataParallel(model)

    # Get loaders

    entity_map = get_entity_map(entities)
    train_men_loader, val_men_loader, _, entity_loader = get_loaders(
        samples_train,
        samples_val,
        None,
        entities,
        config["max_len"],
        tokenizer,
        config["mention_bsz"],
        config["entity_bsz"],
    )

    train_labels = get_labels(
        samples_train, entity_map, ent_to_indexes, nb_train_samples
    )
    val_labels = get_labels(samples_val, entity_map, ent_to_indexes)

    effective_bsz = config["B"] * config["gradient_accumulation_steps"]
    # train
    logger.log("***** train *****")
    logger.log("# train samples: {:d}".format(len(samples_train)))
    logger.log("# val samples: {:d}".format(len(samples_val)))
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
    tr_loss, logging_loss = 0.0, 0.0
    start_epoch = 1

    model.zero_grad()

    logger.log("get candidates embeddings")
    all_cands_embeds = get_embeddings(entity_loader, model, device)

    for epoch in range(start_epoch, config["epochs"] + 1):
        logger.log("\nEpoch {:d}".format(epoch))
        epoch_start_time = datetime.now()

        if config["num_hards"] < 1:
            logger.log("no need to mine hard negatives")
            candidates = None
        mention_embeds = get_embeddings(train_men_loader, model, device)
        logger.log("mining hard negatives")
        mining_start_time = datetime.now()
        candidates = get_hard_negative(
            mention_embeds,
            all_cands_embeds[:nb_train_samples],
            config["k"],
            config["k"] - config["num_cands"],
            samples_train,
            entities,
        )[0]
        mining_time = strtime(mining_start_time)
        logger.log("mining time for epoch {:3d} " "are {:s}".format(epoch, mining_time))
        train_loader = get_loader_from_candidates(
            samples_train,
            dict(itertools.islice(entities.items(), nb_train_samples)),
            train_labels,
            config["max_len"],
            tokenizer,
            candidates,
            config["num_cands"],
            config["rands_ratio"],
            config["type_loss"],
            True,
            config["B"],
            config["num_pos"],
            config["num_rands"],
            config["num_hards"],
        )
        epoch_train_start_time = datetime.now()
        for step, batch in enumerate(train_loader):
            model.train()
            bsz = batch[0].size(0)
            batch = tuple(t.to(device) for t in batch)
            loss = model(*batch)[0]
            if dp:
                loss = loss.sum() / bsz
            else:
                loss /= bsz
            loss_avg = loss / config["gradient_accumulation_steps"]
            if config["fp16"]:
                with amp.scale_loss(loss_avg, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_avg.backward()
            tr_loss += loss_avg.item()

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                if config["fp16"]:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), config["clip"]
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                step_num += 1

                if step_num % config["logging_steps"] == 0:
                    avg_loss = (tr_loss - logging_loss) / config["logging_steps"]
                    wandb.log(
                        {
                            "step": step_num,
                            "epoch": epoch,
                            "batch": step + 1 / len(train_loader),
                            "avg_loss": avg_loss,
                        }
                    )
                    logging_loss = tr_loss

        logger.log(
            "training time for epoch {:3d} "
            "is {:s}".format(epoch, strtime(epoch_train_start_time))
        )
        all_cands_embeds = get_embeddings(entity_loader, model, device)
        all_mention_embeds = get_embeddings(val_men_loader, model, device)
        top_k, scores_k = get_hard_negative(
            all_mention_embeds,
            all_cands_embeds,
            config["k"],
            0,
            samples_val,
            entities,
        )

        eval_result = evaluate(scores_k, top_k, val_labels)

        wandb.log(
            {
                "epochs": epoch,
                "loss": logging_loss,
                "hard_recall": eval_result[2],
                "lrap": eval_result[1],
                "recall": eval_result[0],
                "simple_recall": eval_result[-1],
                "time": strtime(epoch_start_time),
            }
        )
        if eval_result[2] >= best_val_perf_hard:
            current_best_hard = eval_result[2]
            logger.log(
                "------- new best val hard_perf: {:g} --> {:g} "
                "".format(best_val_perf_hard, current_best_hard)
            )
            best_val_perf_hard = current_best_hard
            torch.save(
                {
                    "opt": args,
                    "sd": model.module.state_dict() if dp else model.state_dict(),
                    "perf": best_val_perf,
                    "epoch": epoch,
                    "opt_sd": optimizer.state_dict(),
                    "scheduler_sd": scheduler.state_dict(),
                    "tr_loss": tr_loss,
                    "step_num": step_num,
                    "logging_loss": logging_loss,
                },
                config["model_path"] + "_hard",
            )
            np.save(config["path_cand_embedding"], all_cands_embeds)
        if eval_result[3] >= best_val_perf_simple:
            current_best_simple = eval_result[3]
            logger.log(
                "------- new best val simple_perf: {:g} --> {:g} "
                "".format(best_val_perf_simple, current_best_simple)
            )
            best_val_perf_simple = current_best_simple
            torch.save(
                {
                    "opt": args,
                    "sd": model.module.state_dict() if dp else model.state_dict(),
                    "perf": best_val_perf,
                    "epoch": epoch,
                    "opt_sd": optimizer.state_dict(),
                    "scheduler_sd": scheduler.state_dict(),
                    "tr_loss": tr_loss,
                    "step_num": step_num,
                    "logging_loss": logging_loss,
                },
                config["model_path"] + "_simple",
            )
            np.save(config["path_cand_embedding"], all_cands_embeds)
        if eval_result[1] >= best_val_perf:
            current_best = eval_result[1]
            logger.log(
                "------- new best val perf: {:g} --> {:g} "
                "".format(best_val_perf, current_best)
            )
            best_val_perf = current_best
            torch.save(
                {
                    "opt": args,
                    "sd": model.module.state_dict() if dp else model.state_dict(),
                    "perf": best_val_perf,
                    "epoch": epoch,
                    "opt_sd": optimizer.state_dict(),
                    "scheduler_sd": scheduler.state_dict(),
                    "tr_loss": tr_loss,
                    "step_num": step_num,
                    "logging_loss": logging_loss,
                },
                config["model_path"],
            )
            np.save(config["path_cand_embedding"], all_cands_embeds)
        else:
            logger.log("")


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
