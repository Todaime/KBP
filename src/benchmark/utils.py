"""Functions for the benchmarker."""

import copy
import pickle


def load_kbs(ref_path: str, built_path: str) -> tuple[dict, dict]:
    """Load the model and built kb.

    Args:
        ref_path (str): path to the model kb
        built_path (str): path to the built kb

    Returns:
        tuple[dict, dict]: loaded KBs
    """
    with open(ref_path, "rb") as ref_file:
        ref_kb = pickle.load(ref_file)

    with open(built_path, "rb") as built_file:
        built_kb = pickle.load(built_file)
    return ref_kb, built_kb


def get_macro_f1(pairs: list, false_negative: list, false_positive: list) -> int:
    """Compute the macro F1 score.

    Args:
        pairs (list): Selected pairs
        false_negative (list): entities of the groundtruth kb without match
        false_positive (list): entities of the built kb without match

    Returns:
        int: Macro F1 score.
    """
    tt_ents = len(pairs) + len(false_negative) + len(false_positive)
    return (
        1 if tt_ents == 0 else sum(metrics["F1"] for (_, _, metrics) in pairs) / tt_ents
    )


def get_matching_rels(rels_a: list, rels_b: list):
    """Check relations in common.

    Args:
        rels_a (list): relations from first entity
        rels_b (list): relations from second entity

    Returns:
        relations in common, unmatched relations of a, unmatched relations of b
    """
    common_rels = []
    unmatched_b = []
    unmatched_a = copy.deepcopy(rels_a)
    for rel_b in rels_b:
        exists = False
        rel_b_pred = rel_b[0]
        rel_b_obj = rel_b[1]
        rel_b_file = rel_b[2]
        for rel_a in unmatched_a:
            if (
                rel_b_pred == rel_a[0]
                and rel_b_file == rel_a[2]
                and all(mention in rel_a[1] for mention in rel_b_obj)
            ):
                exists = True
                unmatched_a.remove(rel_a)
                common_rels.append(rel_b)
                break
        if not exists:
            unmatched_b.append(rel_b)

    return common_rels, list(unmatched_a), unmatched_b


def get_matching_attrs(attrs_a: list, attrs_b: list):
    """Check attributes in common.

    Args:
        attrs_a (list): attributes of the first entity
        attrs_b (list): attributes of the second entity

    Returns:
        attributes in common, unmatched attributes of a, unmatched attributes of b
    """
    tp_attr = []
    fp_attr = []
    attrs_a_copy = copy.deepcopy(attrs_a)
    for attr_b in attrs_b:
        if attr_b in attrs_a_copy:
            tp_attr.append(attr_b)
            attrs_a_copy.remove(attr_b)
        else:
            fp_attr.append(tuple(attr_b))
    return tp_attr, list(attrs_a_copy), fp_attr


def get_entity_pair_f1(ent_a: dict, ent_b: dict, nb_elems_init=0) -> dict[str, float]:
    """Compute the metrics at the entity level.

    Args:
        entity_a (Entity): Entity from the reference base
        entity_b (Entity): Entity built with the evaluated system
        nb_elems_init (int): Number of elements in the init KB

    Returns:
        dict[str, float]: F1 score, True Positive, Precision and Recall
    """
    tp_rel, fn_rel, fp_rel = get_matching_rels(
        set(ent_a["relations"]), set(ent_b["relations"])
    )
    tp_attr, fn_attr, fp_attr = get_matching_attrs(
        set(ent_a["attributes"]), set(ent_b["attributes"])
    )

    tp_elems = tp_attr + tp_rel
    fn_elems = fn_attr + fn_rel
    fp_elems = fp_attr + fp_rel
    true_positive = max(0, len(tp_elems) - nb_elems_init)
    recall = (
        0
        if len(ent_a["relations"]) + len(ent_a["attributes"]) - nb_elems_init == 0
        else true_positive
        / (len(ent_a["relations"]) + len(ent_a["attributes"]) - nb_elems_init)
    )
    precision = (
        0
        if len(ent_b["relations"]) + len(ent_b["attributes"]) - nb_elems_init == 0
        else true_positive
        / (len(ent_b["relations"]) + len(ent_b["attributes"]) - nb_elems_init)
    )

    return {
        "F1": (
            (2 * recall * precision / (recall + precision))
            if precision + recall > 0
            else 0
        ),
        "TP": true_positive,
        "P": precision,
        "R": recall,
        "tp_elems": tp_elems,
        "fn_elems": fn_elems,
        "fp_elems": fp_elems,
    }
