"""KB builder using only a NER and a COREF module."""

import os
import pickle

from collections import defaultdict
from dwie.data import PATH_DWIE_COREF_WL_PREDICTIONS, PRONOUNS, DWIE_NER_ONTOLOGY

from . import ner_builder


class NerCorefBuilder(ner_builder.NerBuilder):
    """KB builder with a NER and a Coref model."""

    def rebuild_mention(self, tokens: list) -> str:
        """Rebuild a mention from list of tokens.

        Args:
            tokens (list): token

        Returns:
            str: mention
        """
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

    def get_coref_predictions(self, filename: str) -> list:
        """load the coreference predicted for the file.
        Args:
            filename (str): file used for IE.
        Returns:
            list: predicted coreference.
        """
        with open(
            os.path.join(PATH_DWIE_COREF_WL_PREDICTIONS, filename), "rb"
        ) as coref_file:
            predicted_corefs = pickle.load(coref_file)

        corefs = []
        for cluster in predicted_corefs["span_clusters"]:
            coref = []
            for (begin, end) in cluster:
                rebuilt_mention = self.rebuild_mention(
                    predicted_corefs["cased_words"][begin:end]
                )
                if rebuilt_mention:
                    coref.append(rebuilt_mention)
            corefs.append(set(coref))
        corefs.sort(key=len, reverse=True)
        # tester un regroupement des clusters avec mention similaire?
        return corefs

    def get_possible_clusters(
        self, name, corefs, ner_mentions, mentions_not_in_clusters
    ):
        """finds the clusters to which the entity could correspond.

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
        return (clusters, corefs)

    def build_kb(self, built_kb, filename):
        """Populate the KB with information extracted from the text.
        Args:
            built_kb : KB to populate
            filename : file of the text to integrate
        """
        ents_in_text = set()

        ner_entities = self.get_ner_predictions(filename)
        corefs = self.get_coref_predictions(filename)

        ents = {}
        cluster_to_text_ents = defaultdict(list)
        cluster_to_types = defaultdict(list)
        cluster_to_fine_grained = {}
        mentions_to_cluster = {}
        ner_mentions = [ner["text"] for ner in ner_entities]
        mentions_not_in_cluster = [
            mention
            for mention in ner_mentions
            if len([cluster for cluster in corefs if mention in cluster]) == 0
        ]
        for ner_ent in ner_entities:
            if ner_ent["text"] not in ents:
                ents[ner_ent["text"]] = ner_ent
                clusters, corefs = self.get_possible_clusters(
                    ner_ent["text"], corefs, ner_mentions, mentions_not_in_cluster
                )
                used_cluster = False
                for cluster in clusters:
                    if (
                        len(cluster_to_types[cluster]) == 0
                        or ner_ent["label"] in cluster_to_types[cluster]
                        or cluster[0] in DWIE_NER_ONTOLOGY[ner_ent["label"]]
                    ):
                        name = str(cluster) + "_" + filename
                        cluster_to_text_ents[cluster].append(ner_ent["text"])
                        used_cluster = True
                        mentions_to_cluster[ner_ent["text"].lower()] = cluster
                        if (
                            len(cluster_to_types[cluster]) == 0
                            or cluster_to_fine_grained[cluster]
                            in DWIE_NER_ONTOLOGY[ner_ent["label"]]
                        ):
                            cluster_to_types[cluster] = DWIE_NER_ONTOLOGY[
                                ner_ent["label"]
                            ]
                            cluster_to_fine_grained[cluster] = ner_ent["label"]
                        break
                if not used_cluster:
                    cluster_to_text_ents[ner_ent["text"].lower()].append(
                        ner_ent["text"]
                    )
                    cluster_to_fine_grained[ner_ent["text"].lower()] = ner_ent["label"]

                    mentions_to_cluster[ner_ent["text"].lower()] = ner_ent[
                        "text"
                    ].lower()
                    name = ner_ent["text"] + "_" + filename
                ents_in_text.add(name)

                built_kb["entities"][name]["attributes"].add(
                    (
                        "mention",
                        ner_ent["text"].replace(" - ", "-").replace(" 's ", "'s "),
                        filename,
                    )
                )

                built_kb["entities"][name]["attributes"].update(
                    [
                        ("type", entity_type, filename)
                        for entity_type in DWIE_NER_ONTOLOGY[ner_ent["label"]]
                    ]
                )
        built_kb["texts"][filename] = set(ents_in_text)
