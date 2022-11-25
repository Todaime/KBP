"""Script to build the base with a specified builder."""

from collections import defaultdict
from copy import deepcopy

import os
import json

import logging
import argparse
import pickle
from tqdm import tqdm
from dwie.data import PATH_DWIE, PATH_DWIE_BENCHMARK, PATH_DWIE_DATA
from build_groundtruth_kbs import extract_entities_info

from builders import NerBuilder, NerCorefBuilder, NerCorefREBuilder, Baseline


DEFAULT_PATH_DWIE_SEQUENCE = PATH_DWIE + "sequences10.pickle"
BUILDERS_WITH_RESET = []


def new_entity():
    """Dummy function for Defaultdict"""
    return {"attributes": set(), "relations": set()}


def load_builder(builder_name):
    """Looad the required builder to build the KB."""
    if builder_name == "NER":
        return NerBuilder()
    if builder_name == "NERCoref":
        return NerCorefBuilder()
    if builder_name == "NERCorefRE":
        return NerCorefREBuilder()
    if builder_name == "Baseline":
        return Baseline()
    print("Oups wrong builder")
    return None


def build_init_kb():
    """Build the initial KB for warm-start."""
    init_kb = {
        "texts": {},
        "entities": defaultdict(new_entity),
        "mentions": defaultdict(list),
    }
    for filename in tqdm(os.listdir(PATH_DWIE_DATA)):
        with open(
            os.path.join(PATH_DWIE_DATA, filename), "r", encoding="utf-8"
        ) as file:
            data = json.load(file)

        if data["tags"][1] != "test":

            init_kb["texts"][filename] = []
            entities_info = extract_entities_info(data, filename)

            for (_, info) in entities_info.items():
                if len(info["mentions"]) > 0 and len(info["tags"]) > 0:
                    init_kb["entities"][info["link"]]["attributes"].update(info["tags"])
                    for mention in info["mentions"]:
                        init_kb["mentions"][mention[1].lower()].append(info["link"])
                        init_kb["entities"][info["link"]]["attributes"].add(mention)
                        init_kb["texts"][filename].append(info["link"])
                        init_kb["entities"][info["link"]]["relations"].update(
                            {
                                (relation[0], relation[1], filename)
                                for relation in info["relations"]
                            }
                        )
    return init_kb


def main():
    """Build a KB from a specified builder."""

    with open(params["sequences_path"], "rb") as sequence_file:
        sequences = pickle.load(sequence_file)

    builder = load_builder(params["builder"])

    if params["mode"] == "warm_start":
        init_kb = build_init_kb()
    else:
        init_kb = {
            "texts": {},
            "entities": defaultdict(new_entity),
            "mentions": defaultdict(list),
        }

    for i, sequence in tqdm(enumerate(sequences)):
        built_kb = deepcopy(init_kb)
        os.makedirs(
            os.path.join(
                params["input_path"], params["mode"], params["builder"], str(i)
            ),
            exist_ok=True,
        )
        for j, filename in tqdm(enumerate(sequence)):
            builder.build_kb(built_kb, filename)
            with open(
                os.path.join(
                    params["input_path"],
                    params["mode"],
                    params["builder"],
                    str(i),
                    str(j) + ".pickle",
                ),
                "wb",
            ) as outfile:
                pickle.dump(dict(built_kb), outfile)
        if params["builder"] in BUILDERS_WITH_RESET:
            builder = load_builder(params["builder"])
    logging.info("Ref kbs created and stored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for the end-to-end test bases construction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--sequences_path",
        default=DEFAULT_PATH_DWIE_SEQUENCE,
        help="Path to the ordered sequences",
    )

    parser.add_argument(
        "-i",
        "--input_path",
        default=PATH_DWIE_BENCHMARK + "/input/",
        help="Path where the created databases are stored",
    )

    parser.add_argument(
        "--builder",
        default="Baseline",
        help="Name of the builder to use",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="cold_start",
        choices=["warm_start", "cold_start"],
        help="Build a kb from an existing one or from scratch",
    )

    args = parser.parse_args()
    params = vars(args)
    main()
