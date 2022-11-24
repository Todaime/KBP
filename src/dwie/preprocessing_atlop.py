"""This script process the DWIE dataset to build sets for ATLOP model training."""


import logging
import argparse
import os
import json
import pickle
import spacy
from tqdm import tqdm

from preprocessing_flair import get_concepts_types, is_in_other_span
from data import PATH_DWIE_DATA, PATH_DWIE_RE_ATLOP, SENTENCE_END_CHAR

PATH_SPACY_DOCS = os.path.join(PATH_DWIE_RE_ATLOP, "spacy_docs.pickle")


def get_ents_from_mentions(doc, mentions, concept_types):
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
            m_begin, m_end, m_c = mention["begin"], mention["end"], mention["concept"]
            char_span = doc.char_span(m_begin, m_end, m_c)
            if char_span is not None:
                spans.append((m_begin, m_end, m_c))

    return [
        doc.char_span(begin, end, str(conc))
        for i, (begin, end, conc) in enumerate(spans)
        if not is_in_other_span(spans, i)
    ]


def get_spacy_docs(path_dwie: str) -> dict[str, list]:
    """Load dwie annotated documents.
    Args:
        path_dwie (str): path to the annotated documents
    Returns:
        dict[str, list]: _description_
    """
    if os.path.exists(PATH_SPACY_DOCS):
        return pickle.load(open(PATH_SPACY_DOCS, "rb"))
    nlp = spacy.load("en_core_web_sm")
    spacy_docs = {"train": [], "test": []}
    for filename in tqdm(os.listdir(path_dwie)):
        with open(os.path.join(path_dwie, filename), encoding="utf-8") as dwie_file:
            data = json.load(dwie_file)
            doc = nlp(data["content"])
            concepts_type = get_concepts_types(data["concepts"])
            ents = get_ents_from_mentions(doc, data["mentions"], concepts_type)
            if len(ents) > 0:
                doc.set_ents(ents)
                spacy_docs[data["tags"][1]].append(
                    (filename, doc, data["concepts"], data["relations"], concepts_type)
                )
    with open(PATH_SPACY_DOCS, "wb") as file_spacy:
        pickle.dump(spacy_docs, file_spacy)
    return spacy_docs


def get_mentions_indexes(tagged_mentions: list) -> list:
    """Return the offsets of each mentions.

    Args:
        tagged_mentions (list): _description_

    Returns:
        list: mentions spans
    """
    mentions_indexes = []
    index = 0
    while index < (len(tagged_mentions)):
        start_position = index
        val = tagged_mentions[index]

        while index < len(tagged_mentions) and tagged_mentions[index] == val:
            index += 1
        end_position = index - 1

        if val != "":
            mentions_indexes.append((val, start_position, end_position))
    return mentions_indexes


def update_vertex_set(
    vertex_set,
    concept_to_idx,
    sent,
    tagged_sent,
    sent_id,
    entities_type,
    concept_types,
):
    mentions_indexes = get_mentions_indexes(tagged_sent)
    for (c, start, end) in mentions_indexes:
        concept = int(c)
        if concept in concept_types:
            if concept not in concept_to_idx:
                concept_to_idx[concept] = len(concept_to_idx)
                entities_type[concept_to_idx[concept]] = concept_types[concept]
                vertex_set.append([])

            vertex_set[concept_to_idx[concept]].append(
                {
                    "name": "_".join(sent[start : end + 1])
                    if end != start
                    else sent[start],
                    "sent_id": sent_id,
                    "pos": [start, end],
                    "type": concept_types[concept],
                }
            )


def format_docs_for_dwie(docs) -> list:
    """Format documents for ATLOP processing

    Args:
        docs (list): spacy documents, concepts, relations and concepts types

    Returns:
        list: formated documents.
    """
    formated_data = []
    for (filename, doc, concepts, relations, c_types) in tqdm(docs):
        sents = []
        concept_to_idx = {}
        entities_type = {}
        vertex_set = []
        for sent in doc.sents:
            cur_sent = []
            cur_tagged_sent = []
            for token in sent:
                if token.text != "\n":
                    cur_sent.append(token.text)
                    cur_tagged_sent.append(token.ent_type_)
                else:
                    if len(cur_sent) > 0:
                        sents.append(
                            cur_sent
                            if cur_sent[-1] in SENTENCE_END_CHAR
                            else cur_sent + ["."]
                        )
                    update_vertex_set(
                        vertex_set,
                        concept_to_idx,
                        cur_sent,
                        cur_tagged_sent,
                        len(sents) - 1,
                        entities_type,
                        c_types,
                    )
                    cur_sent = []
                    cur_tagged_sent = []
            sents.append(cur_sent)
            update_vertex_set(
                vertex_set,
                concept_to_idx,
                cur_sent,
                cur_tagged_sent,
                len(sents) - 1,
                entities_type,
                c_types,
            )
        labels = [
            {
                "h": concept_to_idx[relation["s"]],
                "t": concept_to_idx[relation["o"]],
                "r": relation["p"],
                "evidence": "",
            }
            for relation in relations
            if relation["s"] in concept_to_idx
            and relation["o"] in concept_to_idx
            and concepts[relation["o"]]["count"] > 0
            and concepts[relation["s"]]["count"] > 0
        ]

        formated_data.append(
            {
                "title": filename,
                "vertexSet": vertex_set,
                "sents": sents,
                "labels": labels,
                "entity_type": entities_type,
            }
        )
    return formated_data


def main():
    """Format DWIE documents for ATLOP training."""
    docs = get_spacy_docs(params["input_path"])

    os.makedirs(params["output_path"], exist_ok=True)

    formated_test = format_docs_for_dwie(docs["test"])
    formated_train = format_docs_for_dwie(
        docs["train"][: int(len(docs["train"]) * 0.85)]
    )
    formated_val = format_docs_for_dwie(docs["train"][int(len(docs["train"]) * 0.85) :])

    with open(
        os.path.join(params["output_path"], "test.json"), "w", encoding="UTF-8"
    ) as test_file:
        json.dump(formated_test, test_file)

    with open(
        os.path.join(params["output_path"], "train.json"),
        "w",
        encoding="UTF-8",
    ) as train_file:
        json.dump(formated_train, train_file)

    with open(
        os.path.join(params["output_path"], "dev.json"), "w", encoding="UTF-8"
    ) as val_file:
        json.dump(formated_val, val_file)


if __name__ == "__main__":
    logging.basicConfig(
        filename="data_preprocessing.log", encoding="utf-8", level=logging.DEBUG
    )
    parser = argparse.ArgumentParser(
        description="Arguments to generate the tests sequences.",
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
        help="Path to the RE corpus",
        default=PATH_DWIE_RE_ATLOP,
    )
    args = parser.parse_args()
    params = vars(args)
    main()
