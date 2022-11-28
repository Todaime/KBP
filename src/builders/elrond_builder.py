"""Builder to populate a KB with the informations extracted from a text.
"""


import os
from collections import Counter, defaultdict
import pickle
import random
import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizerFast
import numpy as np
from annoy import AnnoyIndex
from . import baseline

from dwie.data import (
    PATH_DWIE_LINK_MERIT_PREDS,
    PATH_DWIE_LINK_WARM_CONTENT,
    DWIE_UNLINKABLE_TYPES,
    PATH_DWIE_LINK_MERIT_CLASSIFIER,
    DWIE_NER_ONTOLOGY,
)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tokenizer():
    tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")
    special_tokens_dict = {"additional_special_tokens": ["[Ent]", "[/Ent]", "[Inter]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


class ElrondBuilder(baseline.BaselineBuilder):
    def __init__(self, mode):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.tokenizer = load_tokenizer()
        self.all_embeds = []
        self.annoy = None
        self.map_ent = []
        if mode == "warm_start":
            with open(PATH_DWIE_LINK_WARM_CONTENT, "rb") as warm_file:
                self.all_embeds, self.map_ent = pickle.load(warm_file)
                self.annoy = self.build_annoy_structure()

        set_seeds(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlbertForSequenceClassification.from_pretrained(
            "albert-base-v2", num_labels=2
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        cpt = (
            torch.load(PATH_DWIE_LINK_MERIT_CLASSIFIER)
            if self.device.type == "cuda"
            else torch.load(
                PATH_DWIE_LINK_MERIT_CLASSIFIER,
                map_location=torch.device("cpu"),
            )
        )
        self.model.load_state_dict(cpt["sd"])
        self.model.to(self.device)
        self.model.eval()
        self.filter = filter
        self.counter_link = 0
        self.counter = 0

    def build_annoy_structure(self):
        all_embs = np.array(self.all_embeds)
        index = AnnoyIndex(all_embs.shape[1], "angular")
        for i in range(all_embs.shape[0]):
            index.add_item(i, all_embs[i, :])
        index.build(10)
        return index

    def retrieve_similar_passages(self, embedded_samples, types, kb):
        ind = -1
        cands = []
        counter_ent = defaultdict(int)
        for i, emb in enumerate(embedded_samples):
            res = self.annoy.get_nns_by_vector(emb, n=100, include_distances=True)
            cands_for_passage = []
            for k, r in enumerate(res[0]):
                if self.has_same_type(types, kb["entities"][self.map_ent[r]["name"]]):
                    cands_for_passage.append(
                        (2 - res[1][k], r, i, self.map_ent[r]["name"])
                    )
                if len(cands_for_passage) > 8:
                    break
            cands += cands_for_passage
        cands = sorted(cands, key=lambda x: x[0], reverse=True)
        filtered_cands = []
        for _, ind, p_ind, cand_name in cands:
            if counter_ent[cand_name] <= 4:
                filtered_cands.append((ind, p_ind))
                if len(filtered_cands) == 1:
                    return filtered_cands
                counter_ent[cand_name] += 1
        return filtered_cands

    def classify(self, t_a, similar_id):
        t_b = self.map_ent[similar_id]["text_ids"]
        texts_ids = (
            t_a[:-1] + [self.tokenizer.convert_tokens_to_ids("[Inter]")] + t_b[1:]
        )
        assert len(texts_ids) <= 512
        ids = (
            torch.from_numpy(
                np.asarray(
                    (
                        texts_ids
                        + [self.tokenizer.pad_token_id] * (512 - len(texts_ids))
                    )[:512]
                ).astype("long")
            )
            .unsqueeze(0)
            .to(self.device)
        )
        mask = (
            torch.from_numpy(
                np.asarray(
                    ([1] * len(texts_ids) + [0] * (512 - len(texts_ids)))[:512]
                ).astype("long")
            )
            .unsqueeze(0)
            .to(self.device)
        )

        pred = np.argmax(
            self.model(input_ids=ids, attention_mask=mask)[0].cpu().detach().numpy()
        )
        return pred == 1

    def link_by_model(self, tokenized_texts, embeddings, types, kb):
        if self.annoy is not None:
            preds = []
            similar_cands = self.retrieve_similar_passages(embeddings, types, kb)
            for (ind, passage_ind) in similar_cands:
                if self.map_ent[ind]["name"] not in preds and self.classify(
                    tokenized_texts[passage_ind], ind
                ):
                    preds.append(self.map_ent[ind]["name"])
            if len(preds) > 0:
                return True, preds[0]
        return False, None

    def link_by_mentions(self, vertex, kb, types):
        cand_ents = []
        seen_names = []
        for mention in vertex:
            if mention["name"] not in seen_names:
                seen_names.append(mention["name"])
                cleaned_name = (
                    mention["name"].replace(" - ", "-").replace(" 's ", "'s ")
                )
                if len(kb["mentions"][cleaned_name.lower()]) > 0:
                    cand_ents += kb["mentions"][cleaned_name.lower()]
                elif (
                    cleaned_name[-1] == "s"
                    and len(kb["mentions"][cleaned_name[:-1].lower()]) > 0
                ):
                    cand_ents += kb["mentions"][cleaned_name[:-1].lower()]
                elif " " in cleaned_name and (
                    len(
                        kb["mentions"][
                            "".join(
                                i[0].upper()
                                for i in cleaned_name.split(" ")
                                if len(i) > 3
                            ).lower()
                        ]
                    )
                    > 0
                ):
                    cand_ents += kb["mentions"][
                        "".join(
                            i[0].upper() for i in cleaned_name.split(" ") if len(i) > 3
                        ).lower()
                    ]
        if len(cand_ents) > 0:
            cand_ents = list(Counter(cand_ents).keys())
        filtered_cands = (
            None
            if cand_ents is None
            else [
                ent
                for ent in cand_ents
                if self.has_same_type(types, kb["entities"][ent])
            ]
        )
        if filtered_cands is None or len(filtered_cands) == 0:
            return (False, None)
        return (True, filtered_cands[0])

    def link_ent(self, vertex, tok_texts, embs, types, kb):
        if len(self.all_embeds) > 0:
            linked, name = self.link_by_model(tok_texts, embs, types, kb)
            if not linked:
                linked, name = self.link_by_mentions(vertex, kb, types)
            if linked:
                return linked, name
        return False, None

    def build_kb(self, built_kb, filename):
        """_summary_
        Args:
            built_kb (_type_): _description_
            filename (_type_): _description_
        Returns:
            _type_: _description_
        """
        with open(
            os.path.join(PATH_DWIE_LINK_MERIT_PREDS, filename),
            "rb",
        ) as file_preds:
            text_infos = pickle.load(file_preds)

        ents_in_text = []
        vertex_to_name = {}
        vertex_to_mentions = defaultdict(list)

        for i, vertex in enumerate(text_infos["vertexSet"]):
            ent_type = vertex[0]["type"]
            ent_types = DWIE_NER_ONTOLOGY[ent_type]
            linked = False
            linkable = ent_type not in DWIE_UNLINKABLE_TYPES
            if linkable:
                linked, name = self.link_ent(
                    vertex,
                    text_infos["tokenized"][i],
                    text_infos["encoded"][i],
                    ent_types,
                    built_kb,
                )
            if not linked:
                name = str(i) + "_" + filename  # Id for the cluster
            vertex_to_name[i] = name
            mentions = set(
                (
                    "mention",
                    mention["name"].replace(" - ", "-").replace(" 's ", "'s "),
                    filename,
                )
                for mention in vertex
            )
            built_kb["entities"][name]["attributes"].update(mentions)

            # add the infered types as attributes
            built_kb["entities"][name]["attributes"].update(
                set(("type", entity_type, filename) for entity_type in ent_types)
            )
            ents_in_text.append(name)
            vertex_to_mentions[i] += [ment for _, ment, _, in mentions]
            if linkable:
                for emb, tok_text in zip(
                    text_infos["encoded"][i], text_infos["tokenized"][i]
                ):
                    self.all_embeds.append(emb)
                    self.map_ent.append(
                        {
                            "name": name,
                            "text_ids": tok_text,
                            "filename": filename,
                        }
                    )
                for _, mention, _ in mentions:
                    lw_mention = mention.lower()
                    if lw_mention not in built_kb["mentions"]:
                        built_kb["mentions"][lw_mention] = [name]
                    else:
                        built_kb["mentions"][lw_mention].append(name)

        if "relations" in text_infos:
            for rel in text_infos["relations"]:
                if rel["h"] in vertex_to_name and rel["t"] in vertex_to_name:
                    built_kb["entities"][vertex_to_name[rel["h"]]]["relations"].add(
                        (rel["r"], tuple(set(vertex_to_mentions[rel["t"]])), filename)
                    )

        if len(self.all_embeds) > 0:
            self.annoy = self.build_annoy_structure()

        built_kb["texts"][filename] = ents_in_text
        return built_kb
