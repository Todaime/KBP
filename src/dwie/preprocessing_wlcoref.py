"""Script to preprocess DWIE data for the model word-level coreference resoluton."""

import argparse
import logging
import os
import json
import jsonlines
import spacy

from tqdm import tqdm

from data import PATH_DWIE_DATA, PATH_DWIE_COREF_WL, SENTENCE_END_CHAR


def get_spacy_docs(path_dwie: str) -> list:
    """Get tokenized text with the help of spicy tokenizer.

    Args:
        path_dwie (str): path to the annotated data

    Returns:
        list: list of spacy document
    """
    nlp = spacy.load("en_core_web_sm")
    spacy_docs = []
    for filename in tqdm(os.listdir(path_dwie)):
        with open(os.path.join(path_dwie, filename), encoding="utf-8") as dwie_file:
            data = json.load(dwie_file)["content"]
        spacy_docs.append((nlp(data), filename))
    return spacy_docs


def get_wlcoref_sents(doc) -> list[str]:
    """_summary_

    Args:
        doc (_type_): _description_

    Returns:
        list[str]: _description_
    """
    sents = []
    for sent in doc.sents:
        sents += [token.text.replace("\n", "") for token in sent] + [""]
    return clean_spaces(sents)


def get_wlcoref_docs(docs: list) -> list:
    """Convert spacy doc to a list of token.

    Args:
        docs (list): list of spacy docs.

    Returns:
        list: list of documents.
    """
    return [(get_wlcoref_sents(doc), doc_name) for (doc, doc_name) in docs]


def clean_spaces(doc: list) -> list:
    """Delete useless token.

    Args:
        doc (list): document to clean

    Returns:
        list: cleaned doc
    """
    cleaned_doc = []
    for i in range(0, len(doc) - 1):
        if not doc[i] == doc[i + 1] == "":
            cleaned_doc.append(doc[i])
            if (not doc[i] in SENTENCE_END_CHAR) and doc[i + 1] == "":
                cleaned_doc.append(".")
    cleaned_doc.append(doc[-1])
    return cleaned_doc


def get_wlcoref_data(spacy_docs: list):
    """Format spacy doc to wl-coref input format.

    Args:
        spacy_docs (list): list of spacy documents

    Returns:
        list: list of formated dwie texts.
    """

    wlcoref_docs = get_wlcoref_docs(spacy_docs)
    docs = []
    for (wlcoref_doc, doc_name) in wlcoref_docs:
        sent_counter = 0
        sent_id = []
        cased_words = []
        for token in wlcoref_doc:
            if token != "":
                sent_id.append(sent_counter)
                cased_words.append(token)
            else:
                sent_counter += 1
        docs.append(
            {
                "document_id": "nw" + doc_name,
                "cased_words": cased_words,
                "sent_id": sent_id,
            }
        )
    return docs


def main():
    """Preprocess DWIE data for coreference resolution."""
    spacy_docs = get_spacy_docs(params["input_path"])
    wlcoref_data = get_wlcoref_data(spacy_docs)
    with jsonlines.open(params["output_path"], "w") as writer:
        writer.write_all(wlcoref_data)


if __name__ == "__main__":
    logging.basicConfig(
        filename="data_preprocessing.log", encoding="utf-8", level=logging.DEBUG
    )
    parser = argparse.ArgumentParser(
        description="Arguments to pre-process DWIE to wl-coref input format.",
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
        help="Path to the wlcoref directory",
        default=PATH_DWIE_COREF_WL,
    )

    args = parser.parse_args()
    params = vars(args)
    main()
