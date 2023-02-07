"""KB builder using ELROND workflow."""

from collections import Counter
from builders.ner_coref_re_builder import NerCorefREBuilder
from dwie.data import DWIE_NER_ONTOLOGY, DWIE_UNLINKABLE_TYPES

FILTERED_ELEMENTS = []


class BaselineBuilder(NerCorefREBuilder):
    """Builder end-to-end."""

    def has_same_type(self, types_a: list, ent_b: dict) -> bool:
        """Check if both entities have matching types

        Args:
            types_a (list): types of first entity
            ent_b (dict): second entity

        Returns:
            bool: Wheter the types match or not.
        """
        types_b = [attr[1] for attr in ent_b["attributes"] if attr[0] == "type"]
        t_a = set(types_a)
        t_b = set(types_b)
        if t_a.issubset(t_b) or t_b.issubset(t_a):
            return True
        return False

    def link_by_mentions(self, mentions, ent_types, built_kb):
        """Link the entity to an existing one based on mention priors.

        Args:
            mentions (list): the entity mentions
            ent_types (list): the entity types
            built_kb (dict): the kb

        Returns:
            str: name of the matching ent if any
        """
        existing_ents = []
        for mention in mentions:
            cleaned_name = mention.replace(" - ", "-").replace(" 's ", "'s ")
            if len(built_kb["mentions"][cleaned_name.lower()]) > 0:
                existing_ents += built_kb["mentions"][cleaned_name.lower()]
            elif (
                cleaned_name[-1] == "s"
                and len(built_kb["mentions"][cleaned_name[:-1].lower()]) > 0
            ):
                existing_ents += built_kb["mentions"][cleaned_name[:-1].lower()]
            elif " " in cleaned_name and (
                len(
                    built_kb["mentions"][
                        "".join(
                            i[0].upper() for i in cleaned_name.split(" ") if len(i) > 3
                        ).lower()
                    ]
                )
                > 0
            ):
                existing_ents += built_kb["mentions"][
                    "".join(
                        i[0].upper() for i in cleaned_name.split(" ") if len(i) > 3
                    ).lower()
                ]
        if len(existing_ents) > 0:
            cand_ents = list(Counter(existing_ents).keys())
            filtered_cands = [
                ent
                for ent in cand_ents
                if self.has_same_type(ent_types, built_kb["entities"][ent])
            ]
            if len(filtered_cands) > 0:
                return filtered_cands[0]
        return None

    def link_ent(self, mentions, ent_types, built_kb):
        return pred self.link_by_mentions(mentions, ent_types, built_kb)

    def build_kb(self, built_kb, filename):
        """Populate the KB with information extracted from the text.
        Args:
            built_kb : KB to populate
            filename : file of the text to integrate
        """
        predictions = self.get_re_predictions(filename)
        ind_to_vertex = {}
        ind_to_mentions = {}
        built_kb["texts"][filename] = []
        for i, vertex in enumerate(predictions["vertexSet"]):

            v_mentions = list(set(mention["name"] for mention in vertex))
            v_type = predictions["entity_type"][str(i)]

            name = None
            if v_type not in DWIE_UNLINKABLE_TYPES:
                name = self.link_ent(v_mentions, DWIE_NER_ONTOLOGY[v_type], built_kb)

            if name is None:
                name = str(i) + "_" + filename
            built_kb["entities"][name]["attributes"].update(
                set(
                    ("type", ent_type, filename)
                    for ent_type in DWIE_NER_ONTOLOGY[v_type]
                )
            )
            built_kb["entities"][name]["attributes"].update(
                set(("mention", men, filename) for men in v_mentions)
            )
            built_kb["texts"][filename].append(name)

            for mention in v_mentions:
                built_kb["mentions"][
                    mention.replace(" - ", "-").replace(" 's ", "'s ").lower()
                ].append(name)

            ind_to_vertex[i] = name
            ind_to_mentions[i] = set(v_mentions)

        if "relations" in predictions:
            for rel in predictions["relations"]:
                built_kb["entities"][ind_to_vertex[rel["h"]]]["relations"].add(
                    (rel["r"], tuple(set(ind_to_mentions[rel["t"]])), filename)
                )
