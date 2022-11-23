"""
Script to build the groundtruth KB for KBP system evaluation.
"""
import json
import os
import pickle
import random
import logging
import argparse

from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from dwie.data import (
    PATH_DWIE,
    PATH_DWIE_DATA,
    PATH_DWIE_GROUNDTRUTH_KBS,
    PATH_DWIE_INIT_KB,
    PATH_DWIE_TEST_FILES,
    UNWANTED_ENT_TYPES,
)

SEQUENCES_FILE_NAME = "sequences"
DEFAULT_MODE = "cold_start"


def get_test_files() -> list[str]:
    """Retrieve the test files in the DWIE dataset.

    Returns:
        list[str]: List of the files included in the test set.
    """
    if os.path.exists(PATH_DWIE_TEST_FILES):
        return pickle.load(open(PATH_DWIE_TEST_FILES, "rb"))
    test_files = []
    for filename in tqdm(os.listdir(PATH_DWIE_DATA)):
        with open(os.path.join(PATH_DWIE_DATA, filename), encoding="utf-8") as file:
            if json.load(file)["tags"][1] == "test":
                test_files.append(filename)
    pickle.dump(test_files, open(PATH_DWIE_TEST_FILES, "wb"))
    return test_files


def get_test_sequences(
    test_files: list[str], nb_sequence: int, output_path: str
) -> list[list[str]]:
    """_summary_

    Args:
        test_files (list[str]): list of the files in the test set
        nb_sequence (int): number of sequence to generate
        output_path (str): path to save the sequences

    Returns:
        list[list[str]]: sequences of text
    """
    sequences_path = (
        os.path.join(output_path, SEQUENCES_FILE_NAME) + f"{nb_sequence}.pickle"
    )
    if os.path.exists(sequences_path):
        logging.info("Loading the sequences.")
        return pickle.load(open(sequences_path, "rb"))

    logging.info("Generating the sequences.")
    test_size = len(test_files)
    sequences = []
    for _ in range(nb_sequence):
        shuffled_test_files = random.sample(test_files, test_size)
        while shuffled_test_files in sequences:
            shuffled_test_files = random.sample(test_files, test_size)
        sequences.append(shuffled_test_files)

    with open(sequences_path, "wb") as seq_file:
        pickle.dump(sequences, seq_file)

    return sequences


def add_link_to_concept(concepts: list[dict], filename: str) -> defaultdict[int, dict]:
    """Add reference to link entities in the texts.

    Args:
        concepts (list[dict]): identified concepts in the file
        filename (str): name of the proessed dwie file

    Returns:
        defaultdict[int, dict]: entities with their reference link in the KB
    """
    entities_info = defaultdict(
        lambda: {"mentions": set(), "relations": set(), "tags": set()}
    )
    for concept in concepts:
        concept_id = concept["concept"]
        if "link" in concept and concept["link"] is not None:
            entities_info[concept_id]["link"] = concept["link"]
        else:
            entities_info[concept_id]["link"] = str(concept_id) + "_" + filename
    return entities_info


def add_mentions_to_entities(
    entities_info: dict[str, dict], mentions: list[dict], filename: str
):
    """List for each entity their mentions in the text.

    Args:
        entities_info (dict[str, dict]): information about entities in the text
        mentions (list[dict]): every mention in the text
        filename (str): name of the file
    """
    for mention in mentions:
        entities_info[mention["concept"]]["mentions"].add(
            ("mention", mention["text"], filename)
        )


def add_tags_to_entities(
    entities_info: dict[int, dict], concepts: list[dict], filename: str
):
    """Add types to the entities.

    Args:
        entities_info (dict[int, dict]): information about entities in the text
        concepts (list[dict]): concepts with their information
        filename (str): name of the file
    """
    for concept in concepts:
        tags = concept["tags"]
        if tags is not None:
            for tag in tags:
                tag_type, value = tag.split("::")
                if tag_type == "type":
                    if value in UNWANTED_ENT_TYPES:
                        entities_info[concept["concept"]]["tags"] = set()
                        break
                    entities_info[concept["concept"]]["tags"].add(
                        ("type", value, filename)
                    )


def add_relations_to_entities(
    ents_info: dict[int, dict], relations: list[dict], filename
):
    """Add the relations between concepts that are present in the text.

    Args:
        entities_info (dict[int, dict]): information about the entities in the text
        relations (list[dict]): relations between concepts
        filename (str): name of the file
    """
    for rel in relations:
        subj, pred, obj = rel["s"], rel["p"], rel["o"]
        if (
            len(ents_info[subj]["mentions"]) > 0
            and len(ents_info[obj]["tags"]) > 0
            and len(ents_info[obj]["mentions"]) > 0
        ):
            ents_info[subj]["relations"].add(
                (pred, tuple(m[1] for m in ents_info[obj]["mentions"]), filename)
            )


def extract_entities_info(data: dict[str, list], filename: str):
    """Process the annotations in the text to build entities.

    Args:
        data (dict[str, list]): annotations
        filename (str): name of the file

    Returns:
        defaultdict[ing, dict]: entities
    """
    entities_info = add_link_to_concept(data["concepts"], filename)
    add_mentions_to_entities(entities_info, data["mentions"], filename)
    add_tags_to_entities(entities_info, data["concepts"], filename)
    add_relations_to_entities(entities_info, data["relations"], filename)
    return entities_info


def build_groundtruth_kb(ref_kb, filename):
    """Buld a groundtruth kb from text annotations.

    Args:
        ref_kb (_type_): KB to populate
        filename (_type_): file to process
    """
    with open(os.path.join(PATH_DWIE_DATA, filename), "r", encoding="utf-8") as file:
        data = json.load(file)

    entities_info = extract_entities_info(data, filename)
    for info in entities_info.values():
        if len(info["mentions"]) > 0 and len(info["tags"]) > 0:
            ref_kb[info["link"]]["attributes"].update(info["tags"])
            ref_kb[info["link"]]["texts"].add(filename)
            ref_kb[info["link"]]["attributes"].update(info["mentions"])
            ref_kb[info["link"]]["relations"].update(info["relations"])


def get_starting_kb(test_files: list[str]) -> dict:
    """Create the initial KB to populate.

    Args:
        test_files (list[str]): the text included in the test set

    Returns:
        dict: the inital KB to populate
    """
    ref_kb = defaultdict(
        lambda: {"attributes": set(), "relations": set(), "texts": set()}
    )
    if params["mode"] == "warm_start":
        for filename in list(set(os.listdir(PATH_DWIE_DATA)) - set(test_files)):
            build_groundtruth_kb(ref_kb, filename)
        with open(PATH_DWIE_INIT_KB, "wb") as init_file:
            pickle.dump(dict(ref_kb), init_file)
    return ref_kb


def build_groundtruth_kbs(test_files, sequences: list[list[str]], output_path):
    """Build the groundtruth KBs after each test file for each sequence.

    Args:
        test_files (list[str]): name of the test files
        sequences (list[list[str]]): sequence of test files
        output_path (str): path to store the built kbs
    """
    logging.info("Creating the groundtruth kbs")
    init_kb = get_starting_kb(test_files)
    for i, sequence in tqdm(enumerate(sequences)):
        ref_kb = deepcopy(init_kb)
        sequence_path = os.path.join(output_path, params["mode"], str(i))
        os.makedirs(sequence_path, exist_ok=True)
        for j, filename in enumerate(sequence):
            build_groundtruth_kb(ref_kb, filename)
            with open(os.path.join(sequence_path, str(j) + ".pickle"), "wb") as outfile:
                pickle.dump(dict(ref_kb), outfile)
    logging.info("Ref kbs created and stored.")


def main():
    """Create and store groundtruth KBs for different order of test files."""
    test_files = get_test_files()
    sequences = get_test_sequences(
        test_files, params["nb_shuffle"], params["output_path_seq"]
    )
    build_groundtruth_kbs(test_files, sequences, params["output_path_kb"])


if __name__ == "__main__":
    logging.basicConfig(
        filename="data_preprocessing.log", encoding="utf-8", level=logging.DEBUG
    )
    parser = argparse.ArgumentParser(
        description="Arguments to generate the groundtruth KBs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--nb_shuffle",
        help="number of sequence to generate",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--output_path_seq",
        help="Path to save the sequences",
        default=PATH_DWIE,
    )
    parser.add_argument(
        "-o",
        "--output_path_kb",
        help="Path to save the kbs",
        default=PATH_DWIE_GROUNDTRUTH_KBS,
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="Benchmark mode of evaluation",
        default=DEFAULT_MODE,
        choices=["warm_start", "cold_start"],
    )

    args = parser.parse_args()
    params = vars(args)
    main()
