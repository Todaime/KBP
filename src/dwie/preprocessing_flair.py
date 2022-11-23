"""Process the DWIE dataset to build files for the FLAIR NER model training.
"""

import logging
import argparse
import os
import json
import pickle
import spacy

from tqdm import tqdm
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from data import (
    DWIE_NER_TYPES,
    DWIE_NER_ONTOLOGY,
    PATH_DWIE_DATA,
    PATH_DWIE_NER_FLAIR,
    UNWANTED_ENT_TYPES,
    UNWANTED_TAG_TYPES,
)


def get_fine_grained_type(types: list[str]) -> str:
    """_summary_

    Args:
        types (list[str]): _description_

    Returns:
        str: _description_
    """
    selected_type, degree = None, 0
    for ent_type in types:
        if ent_type in UNWANTED_ENT_TYPES:
            return None
        if ent_type in TYPES_FOR_NER and degree <= TYPES_FOR_NER[ent_type][0]:
            (degree, selected_type) = TYPES_FOR_NER[ent_type]
    return selected_type


def get_concepts_types(concepts: list) -> dict:
    """Return the types of each concept.

    Args:
        concepts (list): entities in the doc

    Returns:
        dict: types for each concepts
    """
    ner_types = DWIE_NER_ONTOLOGY.keys()
    concepts_types = {}
    for concept in concepts:
        tags = []
        if concept["tags"] is not None:
            for tag in concept["tags"]:
                tag_type, value = tag.split("::")
                if tag_type == "type":
                    if value in UNWANTED_ENT_TYPES:
                        tags = []
                        break
                    tags.append(value)
        if len(tags) > 0:
            tag_ontology = DWIE_NER_ONTOLOGY[tags[0]]
            selected_tag = tags[0]

            for tag in tags:
                if tag not in tag_ontology:
                    if selected_tag in DWIE_NER_ONTOLOGY[tag]:
                        selected_tag = tag
                        tag_ontology = DWIE_NER_ONTOLOGY[tag]
                    else:
                        print(
                            DWIE_NER_ONTOLOGY[tag],
                            DWIE_NER_ONTOLOGY[selected_tag],
                            tags,
                            concept["text"],
                        )
        # concepts_types[concept["concept"]] = get_fine_grained_type(
        #    [tag.split("::")[1] for tag in concept["tags"] if "type::" in tag]
        # )
    return concepts_types


def is_included_in_other_span(spans: tuple[int, int, str], i: int) -> bool:
    """_summary_

    Args:
        spans (tuple[int, int, str]): _description_
        i (int): _description_

    Returns:
        bool: _description_
    """
    (begin, end, _) = spans[i]
    length = end - begin
    for j, (other_begin, other_end, _) in enumerate(spans):
        other_length = other_end - other_begin
        if (
            i != j
            and length < other_length
            and other_begin <= begin <= end <= other_end
        ):
            return True
    return False


def get_ents_from_mentions(doc, mentions, type_of_concepts):
    """_summary_

    Args:
        doc (_type_): _description_
        mentions (_type_): _description_
        type_of_concepts (_type_): _description_

    Returns:
        _type_: _description_
    """
    spans = []

    for mention in mentions:
        m_type = type_of_concepts[mention["concept"]]
        if m_type is not None:
            m_begin, m_end = mention["begin"], mention["end"]
            char_span = doc.char_span(m_begin, m_end, m_type)
            if char_span is None:
                larger_char_span = doc.char_span(m_begin, m_end + 1, m_type)
                if larger_char_span is not None:
                    spans.append((m_begin, m_end + 1, m_type))
            else:
                spans.append((m_begin, m_end, m_type))

    return [
        doc.char_span(b, e, t)
        for i, (b, e, t) in enumerate(spans)
        if not is_included_in_other_span(spans, i)
    ]


def get_spacy_docs(path_dwie: str) -> dict:
    """Transform raw texts to sentences.

    Args:
        path_dwie (str): path to the DWIE annotated data

    Returns:
        dict: _description_
    """
    nlp = spacy.load("en_core_web_sm")
    spacy_train = []
    spacy_test = []
    for filename in tqdm(os.listdir(path_dwie)):
        with open(os.path.join(path_dwie, filename), encoding="utf-8") as dwie_file:
            data = json.load(dwie_file)
            doc = nlp(data["content"])

            spacy_test.append((doc, filename))

            type_of_concepts = get_concepts_types(data["concepts"])
            #   ents = get_ents_from_mentions(doc, data["mentions"], type_of_concepts)
            # if len(ents) > 0:
            #     doc.set_ents(ents)
            #     spacy_docs[data["tags"][1]].append(doc)
    return spacy_train, spacy_test


def spacy_token_to_flair(token) -> str:
    """_summary_

    Args:
        token (_type_): _description_

    Returns:
        str: _description_
    """
    line = token.text
    if line != "\n":
        line += " " + token.ent_iob_
        if token.ent_iob_ != "O":
            line += "-" + token.ent_type_
        return line.replace("\n", "")
    return ""


def flair_sent_from_spacy(sent) -> list[str]:
    """_summary_

    Args:
        sent (_type_): _description_

    Returns:
        list[str]: _description_
    """
    return [spacy_token_to_flair(token) for token in sent] + [""]


def flair_doc_from_spacy(doc) -> list[str]:
    """_summary_

    Args:
        doc (_type_): _description_

    Returns:
        list[str]: _description_
    """
    sents = []
    for sent in doc.sents:
        sents += flair_sent_from_spacy(sent)
    return sents


def flair_from_spacy(docs: list) -> list[str]:
    """_summary_

    Args:
        docs (list): _description_

    Returns:
        list[str]: _description_
    """
    dataset = []
    for doc in tqdm(docs):
        dataset += flair_doc_from_spacy(doc)
    return dataset


def clean_spaces(data: list[str]) -> list[str]:
    """_summary_

    Args:
        data (list[str]): _description_

    Returns:
        list[str]: _description_
    """
    cleaned_data = []
    for i in range(0, len(data) - 1):
        if not data[i] == data[i + 1] == "":
            cleaned_data.append(data[i])
            if (not data[i] in SENTENCE_NER_END_CHAR) and data[i + 1] == "":
                cleaned_data.append(". O")
    cleaned_data.append(data[-1])
    return cleaned_data


def create_flair_datasets(spacy_docs: dict[list]) -> dict[list[str]]:
    """_summary_

    Args:
        spacy_docs (dict[list]): _description_

    Returns:
        dict[list[str]]: _description_
    """
    return {
        "train": clean_spaces(flair_from_spacy(spacy_docs["train"])),
        "test": clean_spaces(flair_from_spacy(spacy_docs["test"])),
    }


def save_flair_datasets(flair_datasets: dict[list[str]], path_flair: str):
    """_summary_

    Args:
        flair_datasets (dict[list[str]]): _description_
        path_flair (str): _description_
    """
    if not os.path.exists(path_flair):
        os.makedirs(path_flair)
    with open(
        os.path.join(path_flair, "train.txt"), mode="w", encoding="utf-8"
    ) as flair_file:
        for token in flair_datasets["train"]:
            flair_file.write(token + "\n")
    with open(
        os.path.join(path_flair, "test.txt"), mode="w", encoding="utf-8"
    ) as flair_file:
        for token in flair_datasets["test"]:
            flair_file.write(token + "\n")


def get_data_for_train(spacy_docs):
    """_summary_

    Args:
        spacy_docs (_type_): _description_
    """
    flair_datasets = create_flair_datasets(spacy_docs)
    save_flair_datasets(flair_datasets, params["output_path"])


def get_doc_for_inference(doc):
    """_summary_

    Args:
        doc (_type_): _description_

    Returns:
        _type_: _description_
    """
    document = []
    for sent in doc.sents:
        document += [token.text.replace("\n", "") for token in sent] + [""]

    cleaned_doc = []
    for i in range(0, len(document) - 1):
        if not document[i] == document[i + 1] == "":
            cleaned_doc.append(document[i])
            if (not document[i] in [".", "?", "!", '"']) and document[i + 1] == "":
                cleaned_doc.append(".")
    cleaned_doc.append(document[-1])
    print(get_doc_for_inference)
    return cleaned_doc


def get_data_for_inference(spacy_docs):
    """_summary_

    Args:
        spacy_docs (_type_): _description_
    """
    flair_datasets = [
        (get_doc_for_inference(doc), filename) for (doc, filename) in spacy_docs["test"]
    ]
    if not os.path.exists(os.path.join(params["output_path"], "test")):
        os.makedirs(os.path.join(params["output_path"], "inference"))
    for (doc, filename) in flair_datasets:
        pickle.dump(
            doc,
            open(
                os.path.join(
                    params["output_path"], "inference", filename[:-5] + ".pickle"
                ),
                "wb",
            ),
        )


def main():
    """Preprocess DWIE texts to train a FLAIR NER model."""
    spacy_docs = get_spacy_docs(params["input_path"])
    if params["train"]:
        get_data_for_train(spacy_docs)
    else:
        print("inference")
        get_data_for_inference(spacy_docs)


if __name__ == "__main__":
    logging.basicConfig(
        filename="data_preprocessing.log", encoding="utf-8", level=logging.DEBUG
    )
    parser = argparse.ArgumentParser(
        description="Arguments for the FLAIR Preprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input_path",
        help="Path to the dwie content directory",
        default=PATH_DWIE_DATA,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Path to save flair preprocessed files",
        default=PATH_DWIE_NER_FLAIR,
    )

    args = parser.parse_args()
    params = vars(args)
    main()
