import argparse
from collections import defaultdict
import pickle
import os
import unidecode
from tqdm import tqdm

from utils import load_tokenizer

PATH_INPUT_MERIT = "data/models/LINK/MERIT/input/aida_"
AIDA_PATH = "data/datasets/AIDA/data.tsv"
MODEL_NAME = "Albert"
INPUT_LENGHT = 256


def get_doc_id(line):
    doc_set = "train"
    if "testa" in line:
        doc_set = "validation"
    elif "testb" in line:
        doc_set = "test"
    rest = line.replace("-DOCSTART- (", "")
    return rest[: rest.index(" ")].replace("testa", "").replace("testb", ""), doc_set


def compute_length(text_list, word_length):
    length = 0
    for token in text_list[:-word_length]:
        if token != " ":
            length += len(token)
    return length


def format_aida(aida_path):
    aida_data = []
    quote_before = False
    # whiteSpaceInFront = True
    space_behind = True
    doc = {"doc_id": "", "text": "", "doc_set": "", "entities": {}}
    with open(aida_path, "r", encoding="UTF-8") as file:
        raw_aida = file.readlines()
    for aida_line in raw_aida:
        if len(aida_line) > 0:
            data = aida_line.split("\t")
            if "DOCSTART" in data[0]:
                if len(doc["text"]) > 0:
                    aida_data.append(doc)
                doc_id, doc_set = get_doc_id(data[0])
                doc = {"doc_id": doc_id, "doc_set": doc_set, "text": "", "entities": {}}
                quote_before = False
            else:
                if data[0] != "\n":
                    char = data[0].replace("\n", " ").strip()
                    # char = data[0].strip()
                    # if we should insert a white space
                    space_front = space_behind
                    space_behind = True
                    if len(doc["text"]) > 0 and len(char) >= 1:
                        if len(char) == 1:
                            if char in ["?", "!", ",", ".", ")", "]", "}"]:
                                space_front = False
                            elif char == '"':
                                if quote_before:
                                    space_front = False
                                if not quote_before:
                                    space_behind = False
                                quote_before = not quote_before
                            elif char in ["(", "[", "{"]:
                                space_behind = False
                        else:
                            if not (char[0].isalpha() or char[0].isdigit()):
                                space_front = False
                            else:
                                space_front = True
                        if space_front:
                            doc["text"] += " "
                    doc["text"] += char

                    if len(data) > 1:
                        if "B" == data[1] and data[3] != "--NME--":
                            current_text_length = compute_length(
                                doc["text"], len(data[0])
                            )
                            span = (
                                data[2],
                                (
                                    current_text_length,
                                    len(data[2].replace(" ", "")),
                                ),
                            )
                            if data[3] not in doc["entities"]:
                                doc["entities"][data[3]] = [span]
                            else:
                                doc["entities"][data[3]].append(span)
    if len(doc["text"]) > 0:
        aida_data.append(doc)
    print(len(aida_data))
    return aida_data


def char2token(text, index):
    char2token_list = []
    for i, tok in enumerate(text):
        char2token_list += [i] * len(tok.replace("▁", ""))
    return char2token_list[index]


def clean_text(text):
    return unidecode.unidecode(text.lower()).replace("i'", "i<unk>")


def tokenize_ent_text(
    filename, ent_name, tokenizer, init_spans, text, max_lenght, counter
):
    input_length = max_lenght - 2  # -2 For beginning and end token
    tokenized_text = tokenizer.tokenize(text)
    i = 1
    spans = [
        (
            char2token(tokenized_text, span[1][0]),
            char2token(tokenized_text, span[1][0] + span[1][1] - 1) + 1,
        )
        for span in init_spans
    ]
    # +1 because of [BOS]
    text_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    if len(text_ids) + 2 * len(spans) < input_length:
        text_ids = tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + tokenized_text + ["[SEP]"]
        )
        spans = [(s[0] + 1 + 2 * i, s[1] + 2 + 2 * i) for i, s in enumerate(spans)]
        for i, (s_span, e_span) in enumerate(spans):
            text_ids.insert(s_span, tokenizer.convert_tokens_to_ids("[Ent]"))
            text_ids.insert(e_span, tokenizer.convert_tokens_to_ids("[/Ent]"))
            assert "[/Ent]" == tokenizer.decode(text_ids[e_span])
            assert "[Ent]" == tokenizer.decode(text_ids[s_span])
            assert tokenizer.decode(text_ids[s_span + 1 : e_span]) == clean_text(
                init_spans[i][0]
            ), "{:s}, {:s}".format(
                tokenizer.decode(text_ids[s_span + 1 : e_span]),
                clean_text(init_spans[i][0]),
            )
        if "<unk>" in tokenizer.decode(text_ids):
            print(tokenizer.decode(text_ids))
        return (
            [
                {
                    "ent": ent_name,
                    "filename": filename,
                    "text": text_ids,
                    "spans": spans,
                    "offset": 0,
                    "index": counter,
                }
            ],
            counter + 1,
        )

    else:
        data = []
        start_spans = [span[0] for span in spans]
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
                    span = spans[start_spans.index(start + i)]
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
                for k, (span_s, span_e) in enumerate(instance_spans):
                    assert "[/Ent]" == tokenizer.decode(instance_ids[span_e])
                    assert "[Ent]" == tokenizer.decode(instance_ids[span_s])
                    assert tokenizer.decode(
                        instance_ids[span_s + 1 : span_e]
                    ) == clean_text(
                        init_spans[instance_spans_index[k]][0]
                    ), "{:s}, {:s}".format(
                        tokenizer.decode(instance_ids[s_span + 1 : e_span]),
                        clean_text(init_spans[instance_spans_index[k]][0]),
                    )
                data.append(
                    {
                        "filename": filename,
                        "text": instance_ids,
                        "spans": instance_spans,
                        "offset": start,
                        "index": counter,
                        "ent": ent_name,
                    }
                )
                if "<unk>" in tokenizer.decode(instance_ids):
                    print(tokenizer.decode(instance_ids))
                counter += 1
            if start + i == len(tokenized_text):
                break
            start += i // 2
    return (data, counter) if len(data) > 0 else (None, counter)


def process_aida_dataset(args):
    tokenizer = load_tokenizer(args.model_name)
    test_ents_to_texts = defaultdict(list)
    train_ents_to_texts = defaultdict(list)
    validation_ents_to_texts = defaultdict(list)
    texts_to_ents = defaultdict(list)
    counter = 0
    mentions_to_ents = defaultdict(list)
    mentions_list = []
    linking_score = 0
    formated_texts = format_aida(args.input_path)
    for text_data in tqdm(formated_texts):
        for ent_name, ent_spans in text_data["entities"].items():
            if "NME" not in ent_name:
                for span_text, _ in ent_spans:
                    mentions_list.append(span_text)
                    if ent_name not in mentions_to_ents[span_text]:
                        mentions_to_ents[span_text].append(ent_name)
                tokenized_text, counter = tokenize_ent_text(
                    text_data["doc_id"],
                    ent_name,
                    tokenizer,
                    ent_spans,
                    text_data["text"],
                    args.input_length,
                    counter,
                )
                if tokenized_text is not None:
                    if text_data["doc_set"] == "validation":
                        validation_ents_to_texts[ent_name] = (
                            validation_ents_to_texts[ent_name] + tokenized_text
                        )

                    elif text_data["doc_set"] == "test":
                        test_ents_to_texts[ent_name] = (
                            test_ents_to_texts[ent_name] + tokenized_text
                        )
                    else:
                        train_ents_to_texts[ent_name] = (
                            train_ents_to_texts[ent_name] + tokenized_text
                        )

                    texts_to_ents[text_data["doc_id"]].append(ent_name)
    for mention in mentions_list:
        linking_score += 1 / len(mentions_to_ents[mention])
    print("Ambiguity cost :", linking_score / len(mentions_list))
    print("Ents in train set : ", len(train_ents_to_texts))
    print("Ents in validation set : ", len(validation_ents_to_texts))
    print("Ents in test set : ", len(test_ents_to_texts))
    return (
        validation_ents_to_texts,
        test_ents_to_texts,
        train_ents_to_texts,
        texts_to_ents,
    )


def main():
    (
        val_ents_to_texts,
        test_ents_to_texts,
        train_ents_to_texts,
        texts_to_ents,
    ) = process_aida_dataset(args)
    os.makedirs(args.output_path + args.model_name, exist_ok=True)
    with open(
        os.path.join(
            args.output_path + args.model_name, "validation_ents_to_texts.pickle"
        ),
        "wb",
    ) as ett_file:
        pickle.dump(val_ents_to_texts, ett_file)
    with open(
        os.path.join(args.output_path + args.model_name, "test_ents_to_texts.pickle"),
        "wb",
    ) as ett_file:
        pickle.dump(test_ents_to_texts, ett_file)
    with open(
        os.path.join(args.output_path + args.model_name, "train_ents_to_texts.pickle"),
        "wb",
    ) as ett_file:
        pickle.dump(train_ents_to_texts, ett_file)
    with open(
        os.path.join(args.output_path + args.model_name, "texts_to_ents.pickle"), "wb"
    ) as tte_file:
        pickle.dump(texts_to_ents, tte_file)


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
        default=AIDA_PATH,
        help="Path to the aida dataset",
    )
    parser.add_argument(
        "--output_path",
        default=PATH_INPUT_MERIT,
        help="Path to save the required input for MERIT like model",
    )
    args = parser.parse_args()

    main()
