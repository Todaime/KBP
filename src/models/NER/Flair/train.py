"""Script to train a flair NER Model."""

import argparse
import logging
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from dwie.data import (
    PATH_FLAIR_CHECKPOINT,
    PATH_FLAIR_BASE_MODEL,
    PATH_DWIE_NER_FLAIR,
)


def load_corpus(input_path: str):
    """Load the DWIE formated dataset.

    Args:
        input_path (str): path to the formated dataset

    Returns:
       flair.datasets.ColumnCorpus : the Flair corpus
    """
    columns = {0: "text", 1: "ner"}  # define columns

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(
        input_path,
        columns,
        train_file="train.txt",
        test_file="test.txt",
    )
    return corpus


def train(corpus, resume):
    """Train a NER model

    Args:
        corpus: Flair corpus
    """
    # Make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type="ner")

    # Initialize fine-tuneable transformer embeddings WITH document context
    embeddings = TransformerWordEmbeddings(
        model="roberta-base", layers="-1", subtoken_pooling="first_last", fine_tune=True
    )

    # Initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type="ner",
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False,
    )

    # Initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # Run fine-tuning
    if resume:
        trained_model = SequenceTagger.load(PATH_FLAIR_CHECKPOINT)
        trainer.resume(trained_model)
    else:
        trainer.fine_tune(
            PATH_FLAIR_BASE_MODEL,
            learning_rate=1e-5,
            max_epochs=30,
            checkpoint=True,
            mini_batch_size=8,
        )


def main():
    """Train a FLAIR Ner model."""
    corpus = load_corpus(params["input_path"])
    train(corpus, params["resume"])


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
        help="Path to the flair formated DWIE datasets",
        default=PATH_DWIE_NER_FLAIR,
    )
    parser.add_argument("-r", "--resume", action="store_true", default=False)
    args = parser.parse_args()
    params = vars(args)
    main()
