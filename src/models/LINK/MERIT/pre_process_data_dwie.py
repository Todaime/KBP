import argparse
from collections import defaultdict
import json
import pickle
import os

from tqdm import tqdm

from utils import load_tokenizer


UNWANTED_ENT_TYPES = [
    "footer",
    "none",
    "skip",
    "time",
    "money",
    "value",
    "role",
    "religion",
    "religion-x",
]

UNWANTED_TAG_TYPES = [
    "topic",
    "iptc",
    "gender",
    "sport",
    "slot",
    "meta",
    "sector",
    "sport_event",
    "policy",
]
DWIE_UNLINKABLE_TYPES = ["gpe1-x", "gpe0-x", "loc-x"]
PATH_DWIE_DATA = "data/DWIE/annotated_texts"
PATH_MERIT_INPUT = "data/models/LINK/MERIT/input/dwie_"
DWIE_PATH = "data/datasets/DWIE/annos_with_content"
MODEL_NAME = "Albert"
INPUT_LENGHT = 256


def tokenize_text(filename, ent_name, tokenizer, elems, text, max_lenght, counter):

    input_length = max_lenght - 2  # -2 For beginning and end token
    tokenized_text = tokenizer.tokenize(text)
    all_elems = tokenizer(text, return_offsets_mapping=True)
    span_token_offsets = []
    i = 1
    j = 0
    tokenizer_offset = 0 if MODEL_NAME == "Albert" else 1
    while j < len(elems["spans"]) and i < len(tokenized_text):
        if all_elems["offset_mapping"][i][0] + tokenizer_offset == elems["spans"][j][0]:
            start_offset = i - 1
            while all_elems["offset_mapping"][i][0] < elems["spans"][j][1] and i < len(
                tokenized_text
            ):
                i += 1
            span_token_offsets.append((start_offset, i - 1))
            j += 1
        else:
            i += 1
    # +1 because of [BOS]
    text_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    if len(text_ids) + 2 * len(elems["spans"]) < input_length:
        text_ids = tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + tokenized_text + ["[SEP]"]
        )
        spans = [
            (s[0] + 1 + 2 * i, s[1] + 2 + 2 * i)
            for i, s in enumerate(span_token_offsets)
        ]
        for i, (s_span, e_span) in enumerate(spans):
            text_ids.insert(s_span, tokenizer.convert_tokens_to_ids("[Ent]"))
            text_ids.insert(e_span, tokenizer.convert_tokens_to_ids("[/Ent]"))
            assert "[/Ent]" == tokenizer.decode(text_ids[e_span])
            assert "[Ent]" == tokenizer.decode(text_ids[s_span])

        return (
            [
                {
                    "ent": ent_name,
                    "filename": filename,
                    "text": text_ids,
                    "spans": spans,
                    "offset": 0,
                    "index": counter,
                    "types": elems["types"],
                    "relations": elems["relations"],
                }
            ],
            counter + 1,
        )

    else:
        data = []
        start_spans = [span[0] for span in span_token_offsets]
        start = 0
        while start <= len(text_ids):
            instance_tokens = ["[CLS]"]
            instance_spans = []
            instance_spans_index = []
            i = 0
            while len(instance_tokens) < input_length - 1 and start + i < len(
                tokenized_text
            ):
                if start + i in start_spans:
                    span = span_token_offsets[start_spans.index(start + i)]
                    instance_spans_index.append(start_spans.index(start + i))
                    length_span = span[1] - span[0]
                    if len(instance_tokens) + 2 + length_span < input_length:
                        instance_spans.append(
                            (
                                len(instance_tokens),
                                len(instance_tokens) + length_span + 1,
                            )
                        )
                        instance_tokens += (
                            ["[Ent]"] + tokenized_text[span[0] : span[1]] + ["[/Ent]"]
                        )
                        i += length_span
                    else:
                        break
                else:
                    instance_tokens.append(tokenized_text[start + i])
                    i += 1
            instance_ids = tokenizer.convert_tokens_to_ids(instance_tokens + ["[SEP]"])
            if len(instance_spans) > 0:
                for (span_s, span_e) in instance_spans:
                    assert "[/Ent]" == tokenizer.decode(instance_ids[span_e])
                    assert "[Ent]" == tokenizer.decode(instance_ids[span_s])

                data.append(
                    {
                        "filename": filename,
                        "text": instance_ids,
                        "spans": instance_spans,
                        "offset": start,
                        "index": counter,
                        "ent": ent_name,
                        "types": elems["types"],
                        "relations": elems["relations"],
                    }
                )
                counter += 1
            if start + i == len(tokenized_text):
                break
            start += i // 2
    return (data, counter) if len(data) > 0 else (None, counter)


def extract_entities_info(content, filename):
    entities_info = defaultdict(dict)
    id_to_used_ent = {}
    for concept in content["concepts"]:
        if concept["tags"] is not None:
            concept_id = concept["concept"]
            if "link" in concept and concept["link"] is not None:
                ent_name = concept["link"]
            else:
                ent_name = str(concept_id) + "_" + filename
            spans = [
                (
                    mention["begin"],
                    mention["end"],
                    mention["text"],
                )
                for mention in content["mentions"]
                if mention["concept"] == concept_id
            ]

            parsed_tags = []
            for tag in concept["tags"]:
                tag_type, value = tag.split("::")
                if tag_type == "type":
                    if value in UNWANTED_ENT_TYPES:
                        parsed_tags = []
                        break
                    parsed_tags.append(value)

        if len(spans) > 0 and len(parsed_tags) > 0:
            id_to_used_ent[concept_id] = ent_name
            entities_info[ent_name]["relations"] = []
            entities_info[ent_name]["id"] = concept_id
            entities_info[ent_name]["spans"] = spans
            entities_info[ent_name]["types"] = parsed_tags

    for relation in content["relations"]:
        subj = relation["s"]
        pred = relation["p"]

        obj = relation["o"]
        if subj in id_to_used_ent and obj in id_to_used_ent:
            entities_info[id_to_used_ent[subj]]["relations"].append(
                (pred, id_to_used_ent[obj])
            )
            entities_info[id_to_used_ent[obj]]["relations"].append(
                (pred, id_to_used_ent[subj])
            )
    return entities_info


def process_dwie_dataset(args):
    tokenizer = load_tokenizer(args.model_name)
    test_ents_to_texts = defaultdict(list)
    train_ents_to_texts = defaultdict(list)
    val_ents_to_texts = defaultdict(list)
    texts_to_ents = defaultdict(list)
    mentions_to_ents = defaultdict(list)
    mentions_list = []
    ent_to_type = {}
    counter_train = 0
    counter_test = 0
    counter = 0
    nb_val_files = 0
    linking_score = 0
    for filename in tqdm(os.listdir(args.input_path)):
        with open(
            os.path.join(args.input_path, filename), "r", encoding="utf-8"
        ) as file:
            text_content = json.load(file)
            if text_content["tags"][1] == "test":
                entities_info, counter_test = extract_entities_info(
                    text_content, counter_test, False
                )
            else:
                if nb_val_files < 100:
                    nb_val_files += 1
                entities_info, counter_train = extract_entities_info(
                    text_content,
                    counter_train,
                    True,
                )
            for (ent_name, ent_elems) in entities_info.items():
                for _, _, span_text in ent_elems["spans"]:
                    mentions_list.append(span_text)
                    if ent_name not in mentions_to_ents[span_text]:
                        mentions_to_ents[span_text].append(ent_name)
                tokenized_text, counter = tokenize_text(
                    filename,
                    ent_name,
                    tokenizer,
                    ent_elems,
                    text_content["content"].replace("\xa0", " "),
                    args.input_length,
                    counter,
                )
                ent_to_type[ent_name] = ent_elems["types"]
                if tokenized_text is not None:
                    if text_content["tags"][1] == "test":
                        test_ents_to_texts[ent_name] = (
                            test_ents_to_texts[ent_name] + tokenized_text
                        )

                    elif nb_val_files <= 100:
                        val_ents_to_texts[ent_name] = (
                            val_ents_to_texts[ent_name] + tokenized_text
                        )
                    else:
                        train_ents_to_texts[ent_name] = (
                            train_ents_to_texts[ent_name] + tokenized_text
                        )

                    texts_to_ents[filename].append(ent_name)

    for mention in mentions_list:
        linking_score += 1 / len(mentions_to_ents[mention])
    print("Ambiguity cost :", linking_score / len(mentions_list))
    return (
        test_ents_to_texts,
        val_ents_to_texts,
        train_ents_to_texts,
        texts_to_ents,
        ent_to_type,
    )


def main():
    """Pre process DWIE data to train MERIT."""
    (
        ents_to_texts_train,
        ents_to_texts_val,
        ents_to_texts_test,
        texts_to_ents,
        ent_to_type,
    ) = process_dwie_dataset(args)

    os.makedirs(args.output_path + args.model_name, exist_ok=True)

    with open(
        os.path.join(args.output_path + args.model_name, "ents_to_texts_test.pickle"),
        "wb",
    ) as file_test:
        pickle.dump(ents_to_texts_test, file_test)

    with open(
        os.path.join(args.output_path + args.model_name, "ents_to_texts_val.pickle"),
        "wb",
    ) as file_val:
        pickle.dump(ents_to_texts_val, file_val)

    with open(
        os.path.join(args.output_path + args.model_name, "ents_to_texts_train.pickle"),
        "wb",
    ) as file_train:
        pickle.dump(ents_to_texts_train, file_train)

    with open(
        os.path.join(args.output_path + args.model_name, "texts_to_ents.pickle"), "wb"
    ) as file_tte:
        pickle.dump(texts_to_ents, file_tte)

    with open(
        os.path.join(args.output_path + args.model_name, "ent_to_type.pickle"), "wb"
    ) as file_type:
        pickle.dump(ent_to_type, file_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_length", type=int, default=INPUT_LENGHT, help="the length of inputs"
    )
    parser.add_argument(
        "--stride", type=int, default=INPUT_LENGHT // 2, help="the length of inputs"
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help="Name of the model encoder",
    )
    parser.add_argument(
        "--input_path",
        default=PATH_DWIE_DATA,
        help="Path to the dwie dataset",
    )
    parser.add_argument(
        "--output_path",
        default=PATH_MERIT_INPUT,
        help="Path to save the required input for MERIT.",
    )
    args = parser.parse_args()

    main()
