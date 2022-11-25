from tqdm import tqdm
import ujson as json

PATH_DWIE_REL_2_ID = "data/DWIE/RE/ATLOP/rel2id.json"
PATH_DWIE_TYPE_2_ID = "data/DWIE/RE/ATLOP/type2id.json"

dwie_rel2id = json.load(open(PATH_DWIE_REL_2_ID, "r"))
dwie_type2id = json.load(open(PATH_DWIE_TYPE_2_ID, "r"))


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i : i + n]) == n
        res += [l[i : i + n]]
    return res


def read_dwie(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample["vertexSet"]
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append(
                    (
                        sent_id,
                        pos[0],
                    )
                )
                entity_end.append(
                    (
                        sent_id,
                        pos[1],
                    )
                )
        for i_s, sent in enumerate(sample["sents"]):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)
        train_triple = {}
        if "labels" in sample:
            for label in sample["labels"]:
                if label["r"] in dwie_rel2id and label["h"] != label["t"]:
                    evidence = label["evidence"]

                    r = int(dwie_rel2id[label["r"]])
                    if (label["h"], label["t"]) not in train_triple:
                        train_triple[(label["h"], label["t"])] = [
                            {"relation": r, "evidence": evidence}
                        ]
                    else:
                        train_triple[(label["h"], label["t"])].append(
                            {"relation": r, "evidence": evidence}
                        )
        entity_pos = []
        entity_type = []
        for i, ent in enumerate(entities):
            entity_pos.append([])
            for ment in ent:
                start = sent_map[ment["sent_id"]][ment["pos"][0]]
                end = sent_map[ment["sent_id"]][ment["pos"][1]]
                entity_pos[-1].append(
                    (
                        start,
                        end,
                    )
                )
            if "entity_type" in sample and sample["entity_type"]:
                entity_type.append(sample["entity_type"][str(i)])

        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(dwie_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(dwie_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1
        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[: max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        feature = {
            "input_ids": input_ids,
            "entity_pos": entity_pos,
            "labels": relations,
            "hts": hts,
            "title": sample["title"],
            "entity_type": entity_type,
        }
        if len(hts) > 0:
            features.append(feature)
    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features
