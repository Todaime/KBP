"""Script to predict Mentions using a trained NER Model and a Gazetteer.
"""

import argparse
import logging
import os
import pickle
import json
from flair.data import Sentence
from flair.models import SequenceTagger

PATH_DWIE_DATA = "data/DWIE/annotated_texts"
PATH_DWIE_TEST_FILES = "data/DWIE/test_files.pickle"
PATH_DWIE_NER_FLAIR_TEST = "data/DWIE/NER/Flair/predictions"
PATH_DWIE_NER_FLAIR_MODEL = (
    "data/models/NER/Flair/taggers/sota-ner-flair/final-model.pt"
)


def predict(model_path, output_path):
    """Predict NER tags on the DWIE test set.
    Args:
        model_path (str): path of the trained model
        output_path (str): path of the output
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = SequenceTagger.load(model_path)

    with open(PATH_DWIE_TEST_FILES, "rb") as list_file:
        test_files = pickle.load(list_file)

    for filename in test_files:

        with open(
            os.path.join(PATH_DWIE_DATA, filename), encoding="UTF-8"
        ) as dwie_file:
            data = json.load(dwie_file)["content"]

        sents = Sentence(data)
        model.predict(sents)
        content = [
            {
                "text": data[ent.start_position : ent.end_position],
                "span": (ent.start_position, ent.end_position),
                "label": ent.get_label("ner").value,
                "idx": (ent[0].idx, len(ent)),
            }
            for ent in sents.get_spans("ner")
        ]
        with open(
            os.path.join(output_path, filename[:-4] + "pickle"),
            "wb",
        ) as res_file:
            pickle.dump(
                (content, sents),
                res_file,
            )


def main():
    """_summary_

    Args:
        params (_type_): _description_
    """
    predict(params["model_path"], params["output_path"])


if __name__ == "__main__":
    logging.basicConfig(
        filename="flair_prediction.log", encoding="utf-8", level=logging.DEBUG
    )
    parser = argparse.ArgumentParser(
        description="Arguments to generate Flair predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_path",
        help="Path to the flair test data",
        default=PATH_DWIE_TEST_FILES,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Path to the flair output directory",
        default=PATH_DWIE_NER_FLAIR_TEST,
    )

    parser.add_argument(
        "-m",
        "--model_path",
        help="Path to the flair model trained on DWIE",
        default=PATH_DWIE_NER_FLAIR_MODEL,
    )

    args = parser.parse_args()
    params = vars(args)
    main()
