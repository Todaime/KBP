"""Process the DWIE dataset to build files for the FLAIR NER model training.
"""

import logging
import argparse
import os
import json
import pickle
import spacy

from tqdm import tqdm


from data import DWIE_NER_TYPES, PATH_DWIE_DATA, PATH_DWIE_NER_FLAIR, UNWANTED_ENT_TYPES

SENTENCE_NER_END_CHAR = [". O", '" O', "? O", "! O"]


def get_concepts_types(concepts: list) -> dict:
    """Return the types of each concept.

    Args:
        concepts (list): entities in the doc

    Returns:
        dict: types for each concepts
    """
    concepts_types = {}
    for concept in concepts:
        selected_tag = None
        if concept["tags"] is not None:
            for tag in concept["tags"]:
                tag_type, value = tag.split("::")
                if tag_type == "type":
                    if value in UNWANTED_ENT_TYPES:
                        selected_tag = None
                        break
                    if value in DWIE_NER_TYPES and (
                        selected_tag is None
                        or DWIE_NER_TYPES[value] > DWIE_NER_TYPES[selected_tag]
                    ):
                        selected_tag = value

        if selected_tag is not None:
            concepts_types[concept["concept"]] = selected_tag
    return concepts_types


def is_in_other_span(spans: tuple, i: int) -> bool:
    """Check if a span is included in another.

    Args:
        spans (tuple): begin, end char position and type
        i (int): index of the span to check

    Returns:
        bool: wheter the span is included in another
    """
    (begin, end, _) = spans[i]
    length = end - begin
    for j, (s_begin, s_end, _) in enumerate(spans):
        s_length = s_end - s_begin
        if i != j and length < s_length and s_begin <= begin <= end <= s_end:
            return True
    return False


def get_spans(doc, mentions, concept_types):
    """Filter the spans.

    Args:
        doc (Spacy.doc): a Spacy document
        mentions (list): annotated mentions
        type_of_concepts (dict): concepts with their type

    Returns:
        list: filtered spans
    """
    spans = []

    for mention in mentions:
        if mention["concept"] in concept_types:
            m_type = concept_types[mention["concept"]]
            m_begin, m_end = mention["begin"], mention["end"]
            char_span = doc.char_span(m_begin, m_end, m_type)
            if char_span is not None:
                spans.append((m_begin, m_end, m_type))

    return [
        doc.char_span(begin, end, tag)
        for i, (begin, end, tag) in enumerate(spans)
        if not is_in_other_span(spans, i)
    ]


def get_spacy_docs(path_dwie: str) -> dict:
    """Transform raw texts to annotated sentences.

    Args:
        path_dwie (str): path to the DWIE annotated data

    Returns:
        docs in the train set, docs in the test set
    """
    nlp = spacy.load("en_core_web_sm")
    spacy_train = []
    spacy_test = []
    for filename in tqdm(os.listdir(path_dwie)):
        with open(os.path.join(path_dwie, filename), encoding="utf-8") as dwie_file:
            data = json.load(dwie_file)
        doc = nlp(data["content"])
        print(doc.sents)
        type_of_concepts = get_concepts_types(data["concepts"])
        spans = get_spans(doc, data["mentions"], type_of_concepts)
        if len(spans) > 0:
            doc.set_ents(spans)

        if data["tags"][1] == "train":
            spacy_train.append(doc)
        else:
            spacy_test.append(doc)
    return spacy_train, spacy_test


def spacy_to_flair_token(token) -> str:
    """Format spacy tokens to flair.

    Args:
        token ): a spacy token

    Returns:
        str: formated flair token
    """
    line = token.text
    if line != "\n":
        line += " " + token.ent_iob_
        if token.ent_iob_ != "O":
            line += "-" + token.ent_type_
        return line.replace("\n", "")
    return ""


def spacy_to_flair_sent(sent) -> list:
    """Translate a spacy sentence to a flair one.
    Args:
        sent (_type_): Spacy sentence.

    Returns:
        list[str]:list of flair token
    """
    return [spacy_to_flair_token(token) for token in sent] + [""]


def spacy_to_flair_doc(doc) -> list:
    """Transform spacy doc to flair format

    Args:
        doc: a spacy document

    Returns:
        list: list of flair token
    """
    sents = []
    for sent in doc.sents:
        sents += spacy_to_flair_sent(sent)
    return sents


def spacy_to_flair(docs: list) -> list[str]:
    """Transform spacy documents to comply with flair format.

    Args:
        docs (list): list fo spacy documents

    Returns:
        list[str]: flair dataset
    """
    dataset = []
    for doc in tqdm(docs):
        dataset += spacy_to_flair_doc(doc)
    return dataset


def clean_spaces(data: list) -> list:
    """Remove unnecessary tokens

    Args:
        data (list[str]): Flair dataset

    Returns:
        list[str]: Cleaned Flair dataset
    """
    cleaned_data = []
    for i in range(0, len(data) - 1):
        if not data[i] == data[i + 1] == "":
            cleaned_data.append(data[i])
            if (not data[i] in SENTENCE_NER_END_CHAR) and data[i + 1] == "":
                cleaned_data.append(". O")
    cleaned_data.append(data[-1])
    return cleaned_data


def get_data(spacy_docs, dataset):
    """Process training data.

    Args:
        spacy_docs (list): list of spacy document
        dataset (str): type of dataset
    """

    flair_datasets = clean_spaces(spacy_to_flair(spacy_docs))
    with open(
        os.path.join(params["output_path"], dataset + ".txt"),
        mode="w",
        encoding="utf-8",
    ) as flair_file:
        for token in flair_datasets:
            flair_file.write(token + "\n")


def main():
    """Preprocess DWIE texts to train a FLAIR NER model."""
    if not os.path.exists(params["output_path"]):
        os.makedirs(params["output_path"])
    if os.path.exists(params["output_path"] + "spacy_docs.pickle"):
        with open(params["output_path"] + "spacy_docs.pickle", "rb") as spacy_files:
            train_docs, test_docs = pickle.load(spacy_files)

    else:
        train_docs, test_docs = get_spacy_docs(params["input_path"])
        with open(params["output_path"] + "spacy_docs.pickle", "wb") as spacy_files:
            pickle.dump((train_docs, test_docs), spacy_files)
    get_data(train_docs, "train")
    get_data(train_docs, "test")


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
