"""KB builder using a NER, a COREF and a RE module."""

import os
import json
from builders import builder
from dwie.data import PATH_DWIE_RE_ATLOP_PREDS, DWIE_NER_ONTOLOGY

FILTERED_ELEMENTS = []


class NerCorefREBuilder(builder.Builder):
    """Builder with only a NER, a Coref and a RE module"""

    def get_re_predictions(self, filename: str):
        """Load RE predictions
        Args:
            filename (str): name of the file to loaod
        Returns:
            list: RE predictions
        """
        with open(
            os.path.join(PATH_DWIE_RE_ATLOP_PREDS, filename),
            "rb",
        ) as pred_file:
            re_predictions = json.load(pred_file)

        return re_predictions

    def build_kb(self, built_kb, filename):
        """Populate the KB with information extracted from the text.
        Args:
            built_kb : KB to populate
            filename : file of the text to integrate
        """

        predictions = self.get_re_predictions(filename)
        mentions = {}
        for i, vertex in enumerate(predictions["vertexSet"]):
            name = str(i) + "_" + filename
            v_type = predictions["entity_type"][str(i)]
            built_kb["entities"][name]["attributes"].update(
                set(
                    ("type", ent_type, filename)
                    for ent_type in DWIE_NER_ONTOLOGY[v_type]
                )
            )
            mentions[name] = []
            for mention in vertex:
                built_kb["entities"][name]["attributes"].add(
                    ("mention", mention["name"], filename)
                )
                mentions[name].append(mention["name"])
        built_kb["texts"][filename] = [
            str(i) + "_" + filename for i in range(len(predictions["vertexSet"]))
        ]
        if "relations" in predictions:
            for rel in predictions["relations"]:
                name_subj = str(rel["h"]) + "_" + filename
                name_obj = str(rel["t"]) + "_" + filename
                built_kb["entities"][name_subj]["relations"].add(
                    (rel["r"], tuple(set(mentions[name_obj])), filename)
                )
