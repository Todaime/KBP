"""Builder including DWIE components."""

import json
from collections import defaultdict
from dwie.data import DWIE_UNLINKABLE_TYPES, UNWANTED_ENT_TYPES
from builders.baseline import BaselineBuilder

PATH_DWIE_TEST_OUTPUT = "data/models/DWIE/test.json"


def filter_types(tags):
    parsed_types = []
    for tag in tags:
        tag_type, value = tag.split("::")
        if tag_type == "type":
            if value in UNWANTED_ENT_TYPES:
                return []
            parsed_types.append(value)
    return parsed_types


class DWIEBuilder(BaselineBuilder):
    def link_ent(self, vertex, built_kb, ent_types):
        pred = self.link_by_mentions(vertex, built_kb, ent_types)
        return (False, None) if pred is None else (True, pred)

    def build_kb(self, built_kb, filename):
        """_summary_
        Args:
            built_kb (_type_): _description_
            filename (_type_): _description_
        Returns:
            _type_: _description_
        """
        with open(PATH_DWIE_TEST_OUTPUT, "r", encoding="utf-8") as file:
            for line in file.readlines():
                text_infos = json.loads(line)
                if text_infos["id"] == filename[:-5]:
                    break

        ents_in_text = []
        concept_to_mentions = defaultdict(set)
        concept_to_types = defaultdict(list)
        concept_to_link = {}

        for concept in text_infos["concepts"]:
            concept_types = filter_types(concept["tags"])
            if len(concept_types) > 0:
                concept_to_types[concept["concept"]] = concept_types

        for mention in text_infos["mentions"]:
            if (
                mention["concept"] in concept_to_types
                and mention["text"] not in concept_to_mentions[mention["concept"]]
            ):
                concept_to_mentions[mention["concept"]].add(mention["text"])

        for concept, mentions in concept_to_mentions.items():
            linked = False
            linkable = (
                len(
                    [
                        1
                        for ent_type in concept_to_types[concept]
                        if ent_type in DWIE_UNLINKABLE_TYPES
                    ]
                )
                == 0
            )
            if linkable:
                linked, name = self.link_ent(
                    mentions, concept_to_types[concept], built_kb
                )
            if not linked:
                name = str(concept) + "_" + filename  # Id for the cluster
            built_kb["entities"][name]["attributes"].update(
                set(("mention", mention, filename) for mention in mentions)
            )
            if linkable:
                for mention in mentions:
                    lw_m = mention.lower()
                    if lw_m not in built_kb["mentions"]:
                        built_kb["mentions"][lw_m] = [name]
                    else:
                        built_kb["mentions"][lw_m].append(name)
            # add the infered types as attributes
            built_kb["entities"][name]["attributes"].update(
                set(
                    ("type", entity_type, filename)
                    for entity_type in concept_to_types[concept]
                )
            )
            ents_in_text.append(name)
            concept_to_link[concept] = name
        for rel in text_infos["relations"]:
            if rel["s"] in concept_to_mentions and rel["o"] in concept_to_mentions:
                built_kb["entities"][concept_to_link[rel["s"]]]["relations"].add(
                    (rel["p"], tuple(set(concept_to_mentions[rel["o"]])), filename)
                )
        built_kb["texts"][filename] = ents_in_text

        return built_kb
