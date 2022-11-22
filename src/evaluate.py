"""
Script to evaluate a KBP system.
"""

import os
import logging
import argparse
import csv
import matplotlib.pyplot as plt
import numpy
from benchmark.benchmarker import Benchmarker


from dwie.data import PATH_DWIE_GROUNDTRUTH_KBS, PATH_DWIE_BENCHMARK, PATH_DWIE_INIT_KB

logging.basicConfig(
    filename="benchmark.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Benchmarker")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

DEFAULT_NB_SEQUENCE = 10
DEFAULT_TEST_SIZE = 100
DEFAULT_STEP = 2
PATH_BENCHMARK = PATH_DWIE_BENCHMARK
PATH_INIT_KB = "data/DWIE/groundtruth"
PATH_INPUT = PATH_BENCHMARK + "/input"


def new_entity():
    return {"attributes": [], "relations": []}


def draw_figure(
    f1_micro: list,
    f1_macro: list,
    std_micro: list,
    std_macro: list,
):
    """Plot and save the results.

    Args:
        f1_micro (list): Avg Micro F1 at each step
        f1_macro (list): Avg Macro F1 at each step
        std_micro (list): Std Micro F1 at each step
        std_macro (list): Std Macro F1 at each step
        params (dict): Evaluation parameters
    """

    fig, axis = plt.subplots(2, 1, constrained_layout=True)
    x_coords = list(range(0, params["test_size"], params["steps"]))
    axis[0].plot(x_coords, f1_micro)
    axis[0].fill_between(
        x_coords, f1_micro - std_micro, f1_micro + std_micro, alpha=0.3
    )
    axis[1].plot(x_coords, f1_macro)
    axis[1].fill_between(
        x_coords, f1_macro - std_macro, f1_macro + std_macro, alpha=0.3
    )
    axis[0].set_xlabel("Nb texts")
    axis[1].set_xlabel("Nb texts")
    axis[1].set_title("F1-macro")
    axis[0].set_title("F1-micro")
    fig.savefig(
        os.path.join(
            params["path_output"],
            params["mode"],
            params["run_name"] + ".png",
        )
    )


def get_final_scores(scores: list[list[float]]):
    """Compute mean and std of the scores.

    Args:
        scores (list[list[float]]): Evaluation scores at each step for each sequence
    """
    mean_macro = [sum(score_seqs) / len(score_seqs) for score_seqs in zip(*scores[1])]
    mean_micro = [sum(score_seqs) / len(score_seqs) for score_seqs in zip(*scores[0])]
    std_macro = numpy.std(scores[1], axis=0)
    std_micro = numpy.std(scores[0], axis=0)
    return mean_macro, mean_micro, std_macro, std_micro


def save_scores(scores: list[list[float]], results: dict):
    """Save evaluation score.

    Args:
        params (dict): Evaluation parameters
        scores (list[list[float]]): Evaluation scores at each step for each sequence
        results (dict): Results of the last sequence for error analysis
    """
    os.makedirs(os.path.join(params["path_output"], params["mode"]), exist_ok=True)

    mean_macro, mean_micro, std_macro, std_micro = get_final_scores(scores)
    draw_figure(mean_micro, mean_macro, std_micro, std_macro)

    path_results = os.path.join(params["path_output"], params["mode"])
    with open(
        os.path.join(path_results, "metrics.csv"),
        "a",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerows([[params["run_name"]], mean_micro, mean_macro, " "])

    with open(
        os.path.join(
            path_results,
            params["run_name"] + "_" + "error_pairs.csv",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(results["error_pairs"])
    with open(
        os.path.join(
            path_results,
            params["run_name"] + "_" + "good_pairs.csv",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(results["good_pairs"])

    with open(
        os.path.join(
            path_results,
            params["run_name"] + "_" + "fp.csv",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(results["fp"])
    with open(
        os.path.join(
            path_results,
            params["run_name"] + "_" + "fn.csv",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(results["fn"])


def main():
    """Benchmark a KBP system."""
    benchmarker = Benchmarker(
        params, PATH_DWIE_INIT_KB if params["mode"] == "warm_start" else None
    )
    scores, results = benchmarker.run_benchmark()
    save_scores(scores, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for the end-to-end evaluation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--nb_shuffle",
        help="number of generated sequences to be evaluated",
        default=DEFAULT_NB_SEQUENCE,
        type=int,
    )

    parser.add_argument(
        "-t",
        "--test_size",
        default=DEFAULT_TEST_SIZE,
        type=int,
        help="Number of texts in the test set",
    )
    parser.add_argument(
        "-i",
        "--path_input",
        default=PATH_INPUT,
        help="Path where the created databases are stored",
    )
    parser.add_argument(
        "-g",
        "--path_groundtruth",
        default=PATH_DWIE_GROUNDTRUTH_KBS,
        help="Path where the reference databases are stored",
    )
    parser.add_argument(
        "-o",
        "--path_output",
        default=PATH_DWIE_BENCHMARK,
        help="Path where system score are written",
    )
    parser.add_argument(
        "-s",
        "--steps",
        default=DEFAULT_STEP,
        help="Interval of textes for the construction of the base",
    )

    parser.add_argument(
        "-r",
        "--run_name",
        help="name of the run",
    )
    parser.add_argument(
        "--mode",
        default="cold_start",
        help="Mode of evaluation",
        choices=["cold_start", "warm_start"],
    )
    args = parser.parse_args()
    params = vars(args)
    main()
