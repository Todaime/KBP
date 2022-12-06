"""Custom loaders for classifier models"""

from collections import defaultdict

import pickle
import os
import random
from sqlalchemy import all_
import torch

from torch.utils.data import DataLoader, Dataset

import numpy as np
from annoy import AnnoyIndex
from utils import sample_range_excluding


# for embedding entities during inference
class EntitySet(Dataset):
    def __init__(self, entities, max_len, tokenizer):
        self.entities = entities
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):

        entity = self.entities[list(self.entities.keys())[index]]
        entity_ids = (
            entity["text"]
            + [self.tokenizer.pad_token_id] * (self.max_len - len(entity["text"]))
        )[: self.max_len]
        entity_masks = (
            [1] * len(entity["text"]) + [0] * (self.max_len - len(entity["text"]))
        )[: self.max_len]
        entity_token_ids = torch.tensor(entity_ids).long()
        entity_masks = torch.tensor(entity_masks).long()
        return entity_token_ids, entity_masks


# For embedding all the mentions during inference
class MentionSet(Dataset):
    def __init__(self, mentions, max_len, tokenizer):
        self.mentions = mentions
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        mention = self.mentions[index]

        mention_ids = (
            mention["text"]
            + [self.tokenizer.pad_token_id] * (self.max_len - len(mention["text"]))
        )[: self.max_len]
        mention_masks = (
            [1] * len(mention["text"]) + [0] * (self.max_len - len(mention["text"]))
        )[: self.max_len]
        mention_ids = torch.tensor(mention_ids).long()
        mention_mask = torch.tensor(mention_masks).long()

        return (mention_ids, mention_mask)


def get_labels(samples, all_entity_map, ent_to_indexes):
    # get labels for samples
    labels = []
    for sample in samples:
        sample_labels = [
            all_entity_map[index] for index in ent_to_indexes[sample["ent"]]
        ]
        labels.append(sample_labels)
    return labels


def get_group_indices(samples):
    # get list of group indices for passages come from the same document
    doc_ids = np.unique([s["doc_id"] for s in samples])
    group_indices = {k: [] for k in doc_ids}
    for i, sample in enumerate(samples):
        doc_id = sample["doc_id"]
        group_indices[doc_id].append(i)
    return list(group_indices.values())


def get_entity_map(entities):
    #  get all entity map: map from entity title to index
    entity_map = {}
    for i, ent in enumerate(entities):
        entity_map[ent] = i
    assert len(entity_map) == len(entities)
    return entity_map


class RetrievalSet(Dataset):
    def __init__(
        self,
        mentions,
        entities,
        labels,
        max_len,
        tokenizer,
        candidates,
        num_cands,
        rands_ratio,
        type_loss,
        num_pos,
        num_rands,
        num_hards,
    ):
        self.num_pos = num_pos
        self.num_hards = num_hards
        self.num_rands = num_rands
        self.num_negs = num_rands + num_hards
        self.mentions = mentions
        self.candidates = candidates
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.labels = labels
        self.num_cands = num_cands
        self.rands_ratio = rands_ratio
        self.all_entity_token_ids = np.array(
            [
                (
                    value["text"]
                    + [self.tokenizer.pad_token_id]
                    * (self.max_len - len(value["text"]))
                )[: self.max_len]
                for _, value in entities.items()
            ]
        )
        self.all_entity_masks = np.array(
            [
                ([1] * len(value["text"]) + [0] * (self.max_len - len(value["text"])))[
                    : self.max_len
                ]
                for _, value in entities.items()
            ]
        )
        self.entities = entities
        self.type_loss = type_loss

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        """
        :param index: The index of mention
        :return: mention_token_ids,mention_masks,entity_token_ids,entity_masks : 1 X L
                entity_hard_token_ids, entity_hard_masks: k X L  (k<=10)
        """
        # process mention
        mention = self.mentions[index]
        mention_ids = mention["text"] + [self.tokenizer.pad_token_id] * (
            self.max_len - len(mention["text"])
        )
        mention_masks = [1] * len(mention["text"]) + [0] * (
            self.max_len - len(mention["text"])
        )
        mention_token_ids = torch.tensor(mention_ids[: self.max_len]).long()
        mention_masks = torch.tensor(mention_masks[: self.max_len]).long()
        # process entity
        cand_ids = random.sample(self.labels[index], 1)

        # non-hard and non-label for random negatives
        rand_cands = sample_range_excluding(
            len(self.entities),
            self.num_rands,
            set(self.labels[index]).union(set(self.candidates[index])),
        )
        cand_ids += rand_cands
        # process hard negatives
        if self.candidates is not None:
            # hard negatives
            hard_negs = random.sample(
                list(set(self.candidates[index]) - set(self.labels[index])),
                self.num_hards,
            )
            cand_ids += hard_negs
        passage_labels = torch.tensor([1] * self.num_pos + [0] * self.num_negs).long()
        candidate_token_ids = self.all_entity_token_ids[cand_ids].tolist()
        candidate_masks = self.all_entity_masks[cand_ids].tolist()
        assert passage_labels.size(0) == self.num_cands
        candidate_token_ids = torch.tensor(candidate_token_ids).long()
        assert candidate_token_ids.size(0) == self.num_cands
        candidate_masks = torch.tensor(candidate_masks).long()
        return (
            mention_token_ids,
            mention_masks,
            candidate_token_ids,
            candidate_masks,
            passage_labels,
        )


def load_data(data_dir):
    """
    :param data_dir
    :return: mentions, entities,doc
    """
    print("begin loading data")

    def load_train_mentions(max_kb_mentions=5):
        mentions = []
        used_for_ent_kb = defaultdict(list)
        entities = defaultdict(list)
        ent_to_indexes = defaultdict(list)
        used_as_mention = defaultdict(list)
        with open(os.path.join(data_dir, "train_ents_to_texts.pickle"), "rb") as file:
            train_ent_mentions = pickle.load(file)

        for ent, tokenized_files_parts in train_ent_mentions.items():
            tfp = tokenized_files_parts[0]
            if len(tokenized_files_parts) == 1:
                entities[tfp["index"]] = tfp
                ent_to_indexes[ent].append(tfp["index"])
                used_for_ent_kb[ent].append(tfp["filename"])
            else:
                nb_exemples = 0
                unused_files = list(
                    set(
                        k["filename"]
                        for k in tokenized_files_parts
                        if k["filename"] != tfp["filename"]
                    )
                )
                for other_tfp in tokenized_files_parts[1:]:
                    if len(unused_files) >= (max_kb_mentions - nb_exemples):
                        if (
                            other_tfp["filename"] in unused_files
                            and other_tfp["index"] not in entities
                        ):
                            unused_files.remove(other_tfp["filename"])
                            entities[other_tfp["index"]] = other_tfp
                            ent_to_indexes[ent].append(other_tfp["index"])
                            used_for_ent_kb[ent].append(other_tfp["filename"])
                            nb_exemples += 1
                    elif (
                        tfp["filename"] != other_tfp["filename"]
                        and other_tfp["index"] not in entities
                    ):
                        entities[other_tfp["index"]] = other_tfp
                        ent_to_indexes[ent].append(other_tfp["index"])
                        used_for_ent_kb[ent].append(other_tfp["filename"])
                        nb_exemples += 1
                    if nb_exemples >= max_kb_mentions:
                        break
                if nb_exemples > 0:
                    mentions.append(tfp)
                    used_as_mention[ent].append(tfp["filename"])
                else:
                    for tfp in tokenized_files_parts[:max_kb_mentions]:
                        entities[tfp["index"]] = tfp
                        ent_to_indexes[ent].append(tfp["index"])
        return mentions, entities, ent_to_indexes

    def load_val_mentions(entities, ent_to_indexes, max_exemples=2):
        mentions = []
        max_exemples = 2
        with open(os.path.join(data_dir, "test_ents_to_texts.pickle"), "rb") as file:
            ent_to_mentions = pickle.load(file)
        for ent, tokenized_files_parts in ent_to_mentions.items():
            nb_exemples = 0
            base = (
                [entities[index] for index in ent_to_indexes[ent]]
                if ent in ent_to_indexes
                else []
            )
            nb_files = len(set(k["filename"] for k in tokenized_files_parts))
            used_files = []
            for tfp in tokenized_files_parts:
                if len(base) > 0:
                    if nb_files >= max_exemples:
                        if tfp["filename"] not in used_files:
                            mentions.append(tfp)
                            nb_exemples += 1
                    else:
                        mentions.append(tfp)
                        nb_exemples += 1
                elif nb_files > 1:
                    if len(used_files) == 0:
                        entities[tfp["index"]] = tfp
                        ent_to_indexes[ent].append(tfp["index"])
                        used_files.append(tfp["filename"])
                    elif tfp["filename"] not in used_files:
                        mentions.append(tfp)
                        used_files.append(tfp["filename"])
                        nb_exemples += 1
                if nb_exemples >= max_exemples:
                    break

        return mentions, entities, ent_to_indexes

    samples_train, entities, ent_to_indexes = load_train_mentions()
    samples_val, entities, ent_to_indexes = load_val_mentions(entities, ent_to_indexes)
    return samples_train, samples_val, entities, ent_to_indexes


def get_embeddings(loader, model, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks = batch
            kwargs = {"mention_token_ids": input_ids, "mention_masks": input_masks}
            embed = model(**kwargs)[0].detach()
            embeddings.append(embed.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    model.train()
    return embeddings


def get_hard_negative(mention_embeddings, all_entity_embeds, k, max_num_postives):
    index = AnnoyIndex(all_entity_embeds.shape[1], "angular")
    for i in range(all_entity_embeds.shape[0]):
        index.add_item(i, all_entity_embeds[i, :])
    index.build(10)
    scores = []
    hard_indices = []
    for m in mention_embeddings:

        res = index.get_nns_by_vector(m, n=k + max_num_postives, include_distances=True)
        scores.append([2 - r for r in res[1]])
        hard_indices.append(res[0])

    del mention_embeddings
    del index
    return hard_indices, scores


def make_single_loader(data_set, bsz, shuffle):
    loader = DataLoader(data_set, bsz, shuffle=shuffle)
    return loader


def get_loader_from_candidates(
    samples,
    entities,
    labels,
    max_len,
    tokenizer,
    candidates,
    num_cands,
    rands_ratio,
    type_loss,
    shuffle,
    bsz,
    num_pos,
    num_rands,
    num_hards,
):
    data_set = RetrievalSet(
        samples,
        entities,
        labels,
        max_len,
        tokenizer,
        candidates,
        num_cands,
        rands_ratio,
        type_loss,
        num_pos,
        num_rands,
        num_hards,
    )
    loader = make_single_loader(data_set, bsz, shuffle)
    return loader


def get_loaders(
    samples_train,
    samples_val,
    entities,
    max_len,
    tokenizer,
    mention_bsz,
    entity_bsz,
):
    #  get all mention and entity dataloaders
    train_men_loader = None
    if samples_train is not None:
        train_mention_set = MentionSet(samples_train, max_len, tokenizer)
        train_men_loader = make_single_loader(train_mention_set, mention_bsz, False)
    val_mention_set = MentionSet(samples_val, max_len, tokenizer)
    entity_set = EntitySet(entities, max_len, tokenizer)
    entity_loader = make_single_loader(entity_set, entity_bsz, False)
    val_men_loader = make_single_loader(val_mention_set, mention_bsz, False)

    return train_men_loader, val_men_loader, entity_loader
