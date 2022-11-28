"""Get results from trained model;
"""
from collections import defaultdict
import os
import argparse
import pickle
import json
import sys
import inspect
import torch
from tqdm import tqdm

from utils import load_tokenizer, set_seeds, Logger, load_model
from pre_process_data_dwie import extract_entities_info, tokenize_text

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

PATH_DWIE_DATA = "data/DWIE/annotated_texts"
PATH_BENCHMARK_EMBS = "data/models/LINK/MERIT/embs_for_benchmark.pickle"
PATH_RETRIEVER_CONFIG = "data/models/LINK/MERIT/config_retriever.json"
PATH_WARM_CONTENT = "data/models/LINK/MERIT/warm_content.pickle"
PATH_BENCHMARK_RELS = "data/models/LINK/MERIT/benchmark_rels.pickle"
INPUT_LENGHT = 256


def main():
    """Encode entities for benchmarking."""
    logger = Logger(config["model_path"] + ".log", on=True)
    set_seeds(config["seed"])

    # configure logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.log(f"Using device: {str(device)}", force=True)

    # get model and tokenizer
    tokenizer = load_tokenizer(config["model_name"])
    model = load_model(config, tokenizer, config["init"], device)

    cpt = (
        torch.load(config["model_path"])
        if device.type == "cuda"
        else torch.load(config["model_path"], map_location=torch.device("cpu"))
    )
    model.load_state_dict(cpt["sd"])
    model.to(device)
    model.eval()

    embeddings = {}
    train_embeddings = []
    train_ent_map = []
    kb_rels = defaultdict(list)
    with torch.no_grad():
        for filename in tqdm(os.listdir(args.input_path)):
            with open(
                os.path.join(args.input_path, filename), "r", encoding="utf-8"
            ) as file:
                text_content = json.load(file)
            file_embeddings = {}
            entities_info = extract_entities_info(text_content, filename)
            for (ent_name, ent_elems) in entities_info.items():
                ent_embeddings = []
                tokenized_texts, _ = tokenize_text(
                    filename,
                    ent_name,
                    tokenizer,
                    ent_elems,
                    text_content["content"].replace("\xa0", " "),
                    INPUT_LENGHT,
                    0,
                )
                kb_rels[ent_name] += [
                    (pred, name, filename) for pred, name in ent_elems["relations"]
                ]
                if tokenized_texts is not None:
                    for t_text in tokenized_texts:
                        t_ids = (
                            t_text["text"]
                            + [tokenizer.pad_token_id]
                            * (config["max_len"] - len(t_text["text"]))
                        )[: config["max_len"]]
                        t_mask = (
                            [1] * len(t_text["text"])
                            + [0] * (config["max_len"] - len(t_text["text"]))
                        )[: config["max_len"]]
                        kwargs = {
                            "mention_token_ids": torch.tensor(t_ids)
                            .long()
                            .unsqueeze(0)
                            .to(device),
                            "mention_masks": torch.tensor(t_mask)
                            .long()
                            .unsqueeze(0)
                            .to(device),
                        }
                        embed = model(**kwargs)[0].detach().cpu()
                        ent_embeddings.append(
                            (
                                embed.numpy().reshape(-1),
                                t_text["text"],
                                t_text["types"],
                                t_text["relations"],
                            )
                        )
                        if text_content["tags"][1] == "train":
                            train_embeddings.append(embed.numpy().reshape(-1))
                            train_ent_map.append(
                                {
                                    "name": ent_name,
                                    "text_ids": t_text["text"],
                                    "filename": filename,
                                }
                            )
                file_embeddings[ent_elems["id"]] = ent_embeddings
            embeddings[filename] = file_embeddings

    with open(PATH_WARM_CONTENT, "wb") as file:
        pickle.dump((train_embeddings, train_ent_map), file)
    with open(PATH_BENCHMARK_RELS, "wb") as file:
        pickle.dump(kb_rels, file)
    with open(PATH_BENCHMARK_EMBS, "wb") as file:
        pickle.dump(embeddings, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default=PATH_DWIE_DATA,
        help="Path to the dwie dataset",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=PATH_RETRIEVER_CONFIG,
        help="Path to the config file for training",
    )
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="UTF-8") as f:
        config = json.load(f)
    # Set environment variables before all else.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Sets torch.cuda behavior
    main()
