"""
Benchmarker for the KBP system evaluation.
"""

import os
from collections import defaultdict
import pickle
import logging
from typing import DefaultDict

import numpy as np

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from benchmark.utils import load_kbs, get_macro_f1, get_entity_pair_f1

Entity = dict[str, list[tuple[str, str, str]]]

logger = logging.getLogger("Benchmarker")


class Benchmarker:
    """Benchmarker for KBP systems"""

    def __init__(self, params, path_init_kb=None):
        self.nb_shuffle = params["nb_shuffle"]
        self.path_groundtruth = params["path_groundtruth"]
        self.path_input = os.path.join(
            params["path_input"], params["mode"], params["run_name"]
        )
        self.steps = params["steps"]
        self.test_size = params["test_size"]
        self.init_kb = None
        if path_init_kb is not None:
            with open(path_init_kb, "rb") as init_file:
                self.init_kb = pickle.load(init_file)

    def get_micro_f1(
        self,
        pairs: list,
        ref_entities: dict,
        built_entities: dict,
        unused_refs: dict,
        unused_builts: dict,
    ) -> float:
        """Computes the micro-f1 score.

        Args:
            pairs (list): The matched pair with their similarity metrics
            ref_entities (list): Entities from the groundtruth base
            built_entities (list): Entities from the built base
            unused_refs (list): Groundtruth entities without match
            unused_builts (list): Built entities without match

        Returns:
            float: The micro F1-score
        """
        tt_true_positive = sum(metrics["TP"] for (_, _, metrics) in pairs)
        init_elems = (
            0
            if self.init_kb is None
            else (
                sum(
                    len(self.init_kb[init_ent]["attributes"])
                    + len(self.init_kb[init_ent]["relations"])
                    for init_ent in self.init_kb.keys()
                )
            )
        )
        nb_to_predict = (
            sum(
                len(ref_entities[ref_ent]["relations"])
                + len(ref_entities[ref_ent]["attributes"])
                for ref_ent in ref_entities
            )
            - init_elems
        )
        nb_predicted = (
            sum(
                len(built_entities[built_ent]["relations"])
                + len(built_entities[built_ent]["attributes"])
                for built_ent in built_entities
            )
            - init_elems
        )

        micro_recall = 0 if nb_to_predict == 0 else tt_true_positive / nb_to_predict
        micro_precision = 0 if nb_predicted == 0 else tt_true_positive / nb_predicted

        if nb_to_predict == nb_predicted == 0:
            return 1
        if (micro_precision + micro_recall) == 0:
            return 0
        return 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    def associate_entities(
        self,
        similarity_scores: dict,
        ents_ref: dict,
        ents_built: dict,
    ) -> tuple:
        """Align entities to produce an optimal F1-score.

        Args:
            similarity_scores (dict): pair of matched entities by similarity score
            ents_ref (dict): entities from the groundtruth KB
            ents_built (dict): entities from the built KB

        Returns:
            list: matched entities,
            list: unmatched entitties from the groundtruth KB
            list: unmatched entities from the built KB
        """
        unused_ref = list(ents_ref.keys())
        unused_built = list(ents_built.keys())
        matched_ents = []

        if not self.init_kb is None:
            for pairs in similarity_scores.values():
                for (ent_ref, ent_built, pair_metrics) in pairs:
                    if ent_ref == ent_built and ent_ref in self.init_kb:
                        unused_built.remove(ent_built)
                        unused_ref.remove(ent_ref)
                        # We only pairs with additional informations.
                        if not (
                            len(ents_ref[ent_ref]["attributes"])
                            == len(ents_built[ent_built]["attributes"])
                            == len(self.init_kb[ent_ref]["attributes"])
                            and len(ents_ref[ent_ref]["relations"])
                            == len(ents_built[ent_built]["relations"])
                            == len(self.init_kb[ent_ref]["relations"])
                        ):
                            matched_ents.append((ent_ref, ent_built, pair_metrics))

        used_ref_matching, used_built_matching = [], []
        dict_of_metrics = DefaultDict(dict)
        score_matrix = np.zeros((len(unused_ref), len(unused_built)))
        for score, pairs in similarity_scores.items():
            for (ent_ref, ent_built, pair_metrics) in pairs:
                if ent_ref in unused_ref and ent_built in unused_built:
                    score_matrix[
                        unused_ref.index(ent_ref), unused_built.index(ent_built)
                    ] = score
                    dict_of_metrics[ent_ref][ent_built] = pair_metrics
        sc_1 = linear_sum_assignment(score_matrix, maximize=True)
        for i, j in zip(sc_1[0], sc_1[1]):
            if score_matrix[i, j] > 0:
                matched_ents.append(
                    (
                        unused_ref[i],
                        unused_built[j],
                        dict_of_metrics[unused_ref[i]][unused_built[j]],
                    )
                )
                used_ref_matching.append(unused_ref[i])
                used_built_matching.append(unused_built[j])
        return (
            matched_ents,
            [ent for ent in unused_ref if ent not in used_ref_matching],
            [ent for ent in unused_built if ent not in used_built_matching],
        )

    def get_matching_scores(self, ref_entities: dict, built_entities: dict) -> dict:
        """Compute similarity score between all possible entities pair.

        Args:
            ref_entities (dict): Entities in the reference base.
            built_entities (dict): Entities in the built base.

        Returns:
            dict: Matching pairs referenced by their score.
        """
        similarity_scores = defaultdict(list)
        for ref, ref_info in ref_entities.items():
            if not self.init_kb is None and ref in self.init_kb:
                pair_metrics = get_entity_pair_f1(
                    ref_info,
                    built_entities["entities"][ref],
                    len(self.init_kb[ref]["relations"])
                    + len(self.init_kb[ref]["attributes"]),
                )
                similarity_scores[pair_metrics["F1"]].append((ref, ref, pair_metrics))
            else:
                candidates = {
                    ent
                    for text_ref in ref_info["texts"]
                    for ent in built_entities["texts"][text_ref]
                }
                for candidate in candidates:
                    if self.init_kb is None or candidate not in self.init_kb:
                        pair_metrics = get_entity_pair_f1(
                            ref_info, built_entities["entities"][candidate]
                        )
                        if pair_metrics["F1"] > 0:
                            similarity_scores[pair_metrics["F1"]].append(
                                (ref, candidate, pair_metrics)
                            )
        return similarity_scores

    def perform_matching(
        self,
        ref_kb: dict,
        built_kb: dict,
    ) -> dict:
        """Align entity between the groundtruth and the built bases.

        Args:
            ref_kb (dict): The reference knowledge base.
            built_kb (dict): The built knowledge base.

        Returns:
            dict: Matching pairs referenced by their score.
        """
        similarity_scores = self.get_matching_scores(ref_kb, built_kb)
        return self.associate_entities(similarity_scores, ref_kb, built_kb["entities"])

    def get_logs_for_pairs(self, pairs: list):
        """Create results for error analyis on matched entities.

        Args:
            pairs (list): matched entities

        Returns:
            list: information for pairs with errors
            list: pairs without error
        """
        error_rows = []
        good_rows = []
        for pair in pairs:
            if len(pair[2]["fn_elems"]) + len(pair[2]["fp_elems"]) > 0:
                error_rows.append([f"{pair[0]} | {pair[1]}"])
                if len(pair[2]["fp_elems"]) > 0:
                    error_rows.append(["Error :"] + pair[2]["fp_elems"])
                if len(pair[2]["fn_elems"]) > 0:
                    error_rows.append(["Missing :"] + pair[2]["fn_elems"])
                error_rows.append(["Good :"] + pair[2]["tp_elems"])
                error_rows.append(["----"])
            else:
                good_rows.append([f"{pair[0]} | {pair[1]}"])
                good_rows.append(["Good :"] + pair[2]["tp_elems"])
                good_rows.append(["----"])
        return error_rows, good_rows

    def get_logs_for_ents(self, ents: list, ents_info: dict) -> list:
        """Format entities information for error analysis.

        Args:
            ents (list):  Name of entities
            ents_info (dict): Information of the entities

        Returns:
            list: Formated entites information
        """
        rows = []
        for ent in ents:
            rows.append([ent])
            rows.append(["attrs :"] + list(ents_info[ent]["attributes"]))
            rows.append(["rel :"] + list(ents_info[ent]["relations"]))
            rows.append([" "])
        return rows

    def measure_distance(self, path_ref: str, path_build: str) -> dict:
        """Measure the distance between two KB.

        Args:
            path_ref (str): path to the reference knowledge base
            path_build (str): path to the built knowledge base

        Returns:
            dict: metrics and their value
        """

        ref_kb, built_kb = load_kbs(path_ref, path_build)
        for e in built_kb["entities"]:
            if len(list(built_kb["entities"][e]["attributes"])) == 0:
                print(e, built_kb["entities"][e])
        pairs, false_negative, false_positive = self.perform_matching(ref_kb, built_kb)

        error_pairs_logs, good_pairs_logs = self.get_logs_for_pairs(pairs)
        fn_logs = self.get_logs_for_ents(false_negative, ref_kb)
        fp_logs = self.get_logs_for_ents(false_positive, built_kb["entities"])

        return {
            "F1_macro": get_macro_f1(pairs, false_negative, false_positive),
            "F1_micro": self.get_micro_f1(
                pairs, ref_kb, built_kb["entities"], false_negative, false_positive
            ),
            "error_pairs": error_pairs_logs,
            "good_pairs": good_pairs_logs,
            "fn": fn_logs,
            "fp": fp_logs,
        }

    def run_benchmark(self):
        """Measure the performances of the end-to-end system.

        Returns:
            list: F1 scores for each sequence
            dict: Output of the last KB state for error analysis
        """
        scores = [[], []]
        for i in tqdm(range(self.nb_shuffle)):
            scores[0].append([])
            scores[1].append([])
            for step in tqdm(range(0, 100, self.steps)):
                path_ref = (
                    os.path.join(
                        self.path_groundtruth,
                        "cold_start" if self.init_kb is None else "warm_start",
                        str(i),
                        str(step),
                    )
                    + ".pickle"
                )
                path_built = (
                    os.path.join(self.path_input, str(i), str(step)) + ".pickle"
                )
                if os.path.exists(path_ref) and os.path.exists(path_built):
                    result = self.measure_distance(path_ref, path_built)
                    logger.info(
                        "Sequence %s/%s, step %s/%s : %s",
                        i,
                        self.nb_shuffle,
                        step,
                        self.test_size,
                        result,
                    )
                    print(result["F1_micro"], result["F1_macro"])
                    scores[0][i].append(result["F1_micro"])
                    scores[1][i].append(result["F1_macro"])
                else:
                    logger.info("Stop due to missing knowledge base.")
                    break
        return scores, result
