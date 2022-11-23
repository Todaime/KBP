"""KB builder using only a NER module."""

import os
import pickle

from builders import builder
from dwie.data import PATH_DWIE_NER_FLAIR_TEST, DWIE_NER_ONTOLOGY

FILTERED_ELEMENTS = []


class NerBuilder(builder.Builder):
    """Builder with only a NER module"""

    def is_filtered(self, ent):
        """Filter some errors."""
        return (
            len([True for elem_to_filter in FILTERED_ELEMENTS if elem_to_filter in ent])
            > 0
        )

    def get_ner_predictions(self, filename: str):
        """_summary_
        Args:
            filename (_type_): _description_
        Returns:
            _type_: _description_
        """
        with open(
            os.path.join(
                PATH_DWIE_NER_FLAIR_TEST,
                filename.split("json")[0] + "pickle",
            ),
            "rb",
        ) as ner_prediction_file:
            ner_predictions, _ = pickle.load(ner_prediction_file)
        return ner_predictions

    def build_kb(self, built_kb, filename):
        """Populate the KB from informations in the text.
        Args:
            built_kb : KB to populate
            filename : file of the text to integrate
        """
        ner_entities = self.get_ner_predictions(filename)
        ents_in_text = []
        ents = []
        for ner_entity in ner_entities:
            attrs = set()
            if (
                ner_entity["label"] is not None
                and ner_entity["text"] not in ents
                and not self.is_filtered(ner_entity["text"])
            ):

                for entity_type in DWIE_NER_ONTOLOGY[ner_entity["label"]]:
                    attrs.add(("type", entity_type, filename))
                ents.append(ner_entity["text"])
                name = ner_entity["text"] + "_" + filename
                ents_in_text.append(name)
                built_kb["entities"][name]["attributes"].update(set(attrs))
                built_kb["entities"][name]["attributes"].add(
                    (
                        "mention",
                        ner_entity["text"].replace(" - ", "-").replace(" 's ", "'s "),
                        filename,
                    )
                )

        built_kb["texts"][filename] = ents_in_text
