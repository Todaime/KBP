"""Preprocess outputs from previous modules to predict relations with ATOP.
"""

from collections import defaultdict
import os
import json
import pickle
from tqdm import tqdm

PATH_DWIE_RE_ATLOP = "data/DWIE/RE/ATLOP"
PATH_DWIE_DATA = "data/DWIE/annotated_texts"
PATH_DWIE_NER_FLAIR_TEST = "data/DWIE/NER/Flair/predictions"
PATH_DWIE_COREF_WL_PREDICTIONS = "data/DWIE/COREF/WL"
SENTENCE_END_CHAR = [".", '"', "?", "!"]

PRONOUNS = [
    "they",
    "he",
    "her",
    "his",
    "him",
    "their",
    "them",
    "we",
    "it",
    "i",
    "our",
    "its",
    "this",
    "you",
    "the",
]

DWIE_NER_ONTOLOGY = {
    "entity": ["entity"],
    "language": ["language", "entity"],
    "location": ["location", "entity"],
    "loc": ["loc", "location", "entity"],
    "waterbody": ["waterbody", "location", "entity"],
    "facility": ["facility", "location", "entity"],
    "gpe": ["gpe", "location", "entity"],
    "gpe0": ["gpe0", "gpe", "location", "entity"],
    "gpe1": ["gpe1", "gpe", "location", "entity"],
    "gpe2": ["gpe2", "gpe", "location", "entity"],
    "regio": ["regio", "location", "entity"],
    "organization": ["organization", "entity"],
    "ngo": ["ngo", "organization", "entity"],
    "education_org": ["education_org", "organization", "entity"],
    "media": ["media", "organization", "entity"],
    "sport_team": ["sport_team", "organization", "entity"],
    "armed_mov": ["armed_movement", "organization", "entity"],
    "governmental_organisation": [
        "governmental_organisation",
        "organization",
        "entity",
    ],
    "agency": [
        "agency",
        "governmental_organisation",
        "organization",
        "entity",
    ],
    "armed_movement": ["armed_movement", "organization", "entity"],
    "company": ["company", "organization", "entity"],
    "igo": ["igo", "organization", "entity"],
    "so": ["so", "organization", "entity", "igo"],
    "party": ["party", "organization", "entity"],
    "person": ["person", "entity"],
    "deity": ["deity", "person", "entity"],
    "sport_player": ["sport_player", "person", "entity"],
    "artist": ["artist", "person", "entity"],
    "politics_per": ["politics_per", "person", "entity"],
    "manager": ["manager", "person", "entity"],
    "offender": ["offender", "person", "entity"],
    "employee": ["employee", "person", "entity"],
    "gov_per": ["gov_per", "person", "entity"],
    "journalist": ["journalist", "person", "entity"],
    "activist": ["activist", "person", "entity"],
    "politician": ["politician", "person", "entity"],
    "head_of_state": ["head_of_state", "person", "entity", "politician"],
    "head_of_gov": ["head_of_gov", "person", "entity", "politician"],
    "minister": ["minister", "person", "entity", "politician"],
    "ethnicity": ["ethnicity", "entity"],
    "event": ["event", "entity"],
    "competition": ["competition", "event", "entity"],
    "sport_competition": ["sport_competition", "competition", "event", "entity"],
    "misc": ["misc", "entity"],
    "work_of_art": ["work_of_art", "misc", "entity"],
    "object": ["object", "misc", "entity"],
    "treaty": ["treaty", "misc", "entity"],
    "other": ["other"],
    "gpe0-x": ["gpe0-x", "other"],
    "gpe1-x": ["gpe1-x", "other"],
    "loc-x": ["loc-x", "other"],
}


def construct_sents(text) -> list:
    """Split the text into sentences.

    Args:
        text : Document input

    Returns:
        list: list of document sentences
    """
    sents = []
    cur_sent = []
    for i, word in enumerate(text):
        cur_sent.append(word.text)
        if (
            word.text in SENTENCE_END_CHAR
            and (i + 1 == len(text) or text[i + 1].text != '"')
        ) or (word.text == '"' and text[i - 1].text in SENTENCE_END_CHAR):
            sents.append(cur_sent)
            cur_sent = []
    if len(cur_sent) > 0:
        sents.append(cur_sent)
    return sents


def map_ent(
    sents: list,
    ent: dict,
) -> dict:
    """Map an entity to sentences and indexes.

    Args:
        sents (list): list of tokenized sentences
        ent (dict): ner information

    Returns:
        dict: mapped ner prediction
    """

    tok_counter = 1
    for i, sent in enumerate(sents):
        for j in range(len(sent)):
            if ent["idx"][0] == tok_counter:
                return {
                    "name": ent["text"],
                    "type": ent["label"],
                    "sent_id": i,
                    "pos": [j, j + ent["idx"][1]],
                    "span": ent["span"],
                }
            tok_counter += 1


def map_doc_to_sent_indexes(ents: list, sents: list) -> list:
    """Map the entities to their position in the sentences.

    Args:
        ents list: the ner predictions
        sents list: the document

    Returns:
        list: list of mapped ent to the sents indexes.
    """
    return [map_ent(sents, ent) for ent in ents]


def rebuild_mention(tokens: list) -> str:
    """Transform the tokenized mention into the original one.

    Args:
        tokens (list): splited mention

    Returns:
        str: original mention
    """
    if tokens[0].lower() == "the" and len(tokens) > 1:
        tokens = tokens[1:]

    rebuilt = tokens[0]
    space_after = True
    for token in tokens[1:]:
        if token not in ["'s", "-", ",", "’s"]:
            if not space_after:
                space_after = True
            else:
                rebuilt += " "
        if token == "-":
            space_after = False
        rebuilt += token

    return rebuilt if rebuilt.lower() not in PRONOUNS else None


def get_corefs(filename: str) -> list:
    """Load the predicted coreferences.

    Args:
        filename (str): test file path

    Returns:
        list: predicted coreferences.
    """
    with open(
        os.path.join(PATH_DWIE_COREF_WL_PREDICTIONS, filename), "rb"
    ) as prediction_file:
        predicted_corefs = pickle.load(prediction_file)

    corefs = []
    for cluster in predicted_corefs["span_clusters"]:
        coref = []
        for (begin, end) in cluster:
            rebuilt_mention = rebuild_mention(
                predicted_corefs["cased_words"][begin:end]
            )
            if rebuilt_mention:
                coref.append(rebuilt_mention)
        corefs.append(set(coref))
    return corefs


def get_possible_clusters(name, corefs, ner_mentions, mentions_not_in_clusters):
    """Extend clusters.

    Args:
        name (_type_): _description_
        corefs (_type_): _description_
        ner_mentions (_type_): _description_
        mentions_not_in_clusters (_type_): _description_

    Returns:
        _type_: _description_
    """
    clusters = [tuple(cluster) for cluster in corefs if name in cluster]

    if " " in name:
        split = name.split(" ")
        acronym = "".join(i[0].upper() for i in split if len(i) > 3)
        clusters += [
            tuple(cluster)
            for cluster in corefs
            for s in split + [acronym]
            if s in cluster
        ]
        if acronym in mentions_not_in_clusters:
            clusters += [(acronym, name)]
            corefs += [{acronym, name}]
    else:
        for cluster in corefs:
            for cluster_mention in cluster:
                if (
                    " " in cluster_mention
                    and name in cluster_mention.split(" ")
                    and cluster_mention in ner_mentions
                ):
                    clusters += [tuple(cluster)]
    if len(clusters) == 0:
        for solo_mention in mentions_not_in_clusters:
            if (
                (" " in name and len([s for s in split if s == solo_mention]) > 0)
                or (
                    " " in solo_mention
                    and len([s for s in solo_mention.split(" ") if s == name]) > 0
                )
            ) and name != solo_mention:
                clusters += [(solo_mention, name)]
                corefs += [{solo_mention, name}]
    return clusters, corefs


def build_vertex_set(mentions, corefs):
    """Merge coreferences and predicted mentions.

    Args:
        mentions (list): predicted mentions.
        corefs (list): predicted corerefences.

    Returns:
        list: _description_
        dict:
    """
    cluster_to_text_ents = defaultdict(list)
    cluster_to_types = defaultdict(list)
    cluster_to_type = {}
    mentions_to_cluster = {}
    ner_mentions = [mention["name"] for mention in mentions]
    mentions_not_in_cluster = [
        mention["name"]
        for mention in mentions
        if len([cluster for cluster in corefs if mention["name"] in cluster]) == 0
    ]
    for i, mention in enumerate(mentions):
        if mention["name"].lower() in mentions_to_cluster:
            cluster_to_text_ents[mentions_to_cluster[mention["name"].lower()]].append(i)
        else:
            clusters, corefs = get_possible_clusters(
                mention["name"], corefs, ner_mentions, mentions_not_in_cluster
            )
            used_cluster = False
            for cluster in clusters:
                if (
                    len(cluster_to_types[cluster]) == 0
                    or mention["type"] in cluster_to_types[cluster]
                    or cluster_to_type[cluster] in DWIE_NER_ONTOLOGY[mention["type"]]
                ):
                    cluster_to_text_ents[cluster].append(i)
                    used_cluster = True
                    mentions_to_cluster[mention["name"].lower()] = cluster
                    if (
                        len(cluster_to_types[cluster]) == 0
                        or cluster_to_type[cluster]
                        in DWIE_NER_ONTOLOGY[mention["type"]]
                    ):
                        cluster_to_types[cluster] = DWIE_NER_ONTOLOGY[mention["type"]]
                        cluster_to_type[cluster] = mention["type"]
                    break
            if not used_cluster:
                cluster_to_text_ents[mention["name"].lower()].append(i)
                cluster_to_type[mention["name"].lower()] = mention["type"]
                mentions_to_cluster[mention["name"].lower()] = mention["name"].lower()

    vertex_set = [[mentions[i] for i in it] for k, it in cluster_to_text_ents.items()]
    entity_type = {
        i: cluster_to_type[k][1]
        for i, (k, it) in enumerate(cluster_to_text_ents.items())
    }
    return vertex_set, entity_type


def pre_process_file(filename):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(
        os.path.join(PATH_DWIE_NER_FLAIR_TEST, filename.split("json")[0] + "pickle"),
        "rb",
    ) as ner_f:
        ner_ents, text = pickle.load(ner_f)
    sents = construct_sents(text)
    mapped_doc = map_doc_to_sent_indexes(ner_ents, sents)
    corefs = get_corefs(filename)
    vertex_set, entity_type = build_vertex_set(mapped_doc, corefs)
    return [
        {
            "title": filename,
            "sents": sents,
            "vertexSet": vertex_set,
            "entity_type": entity_type,
        }
    ]


def main():
    """Process output from the previus modules to prepare for RE."""
    os.makedirs(PATH_DWIE_RE_ATLOP, exist_ok=True)
    for filename in tqdm(os.listdir(PATH_DWIE_DATA)):
        with open(os.path.join(PATH_DWIE_DATA, filename), encoding="utf-8") as file:
            json_file = json.load(file)
        if json_file["tags"][1] == "test":
            processed_file = pre_process_file(filename)
            with open(
                os.path.join(PATH_DWIE_RE_ATLOP, filename), "w", encoding="UTF-8"
            ) as file:
                json.dump(processed_file, file)


if __name__ == "__main__":
    main()
