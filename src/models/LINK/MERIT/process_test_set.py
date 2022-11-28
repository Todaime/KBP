"""Process test set by encoding text samples with train model."""
import os
import argparse
import json
import pickle
from tqdm import tqdm
import torch
from utils import load_model, load_tokenizer


INPUT_LEN = 256
MODEL_NAME = "Albert"

PATH_DWIE_DATA = "data/DWIE/annotated_texts"
PATH_DWIE_LINK_MERIT_PREDS = "data/DWIE/LINK/MERIT/predictions"
PATH_DWIE_RE_ATLOP_PREDS = "data/DWIE/RE/ATLOP/predictions"
PATH_RETRIEVER_CONFIG = "data/models/LINK/MERIT/config_retriever.json"

TOKENIZER_NAME = "Albert"


def get_spans(mentions: list, new_line_index=int) -> list:
    """Get all mentions offsets."""
    span = []
    for mention in mentions:
        if mention["span"][0] > new_line_index:
            span.append([mention["span"][0] - 2, mention["span"][1] - 2])
        else:
            span.append((mention["span"]))
    return span


def tokenize_text_for_vertex(tokenizer, init_spans, text):
    """_summary_

    Args:
        tokenizer (_type_): _description_
        spans (_type_): _description_
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    input_length = INPUT_LEN - 2  # -2 For beginning and end token
    tokenized_text = tokenizer.tokenize(text)
    all_elems = tokenizer(text, return_offsets_mapping=True)

    span_token_offsets = []
    i = 1
    j = 0
    tokenizer_offset = 0 if MODEL_NAME == "Albert" else 1
    while j < len(init_spans) and i < len(tokenized_text):
        if all_elems["offset_mapping"][i][0] + tokenizer_offset == init_spans[j][0]:
            start_offset = i - 1
            while all_elems["offset_mapping"][i][0] < init_spans[j][1] and i < len(
                tokenized_text
            ):
                i += 1
            span_token_offsets.append((start_offset, i - 1))
            j += 1
        else:
            i += 1
    # +1 because of [BOS]
    text_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    if len(text_ids) + 2 * len(init_spans) < input_length:
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

        return [{"text": text_ids, "spans": spans}]
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

            data.append({"text": instance_ids, "spans": instance_spans})
        if start + i == len(tokenized_text):
            break
        start += i // 2
    return data


def get_embedding(text_ids, tokenizer, model, device):
    t_ids = (text_ids + [tokenizer.pad_token_id] * (config["max_len"] - len(text_ids)))[
        : config["max_len"]
    ]

    t_mask = ([1] * len(text_ids) + [0] * (config["max_len"] - len(text_ids)))[
        : config["max_len"]
    ]
    kwargs = {
        "mention_token_ids": torch.tensor(t_ids).long().unsqueeze(0).to(device),
        "mention_masks": torch.tensor(t_mask).long().unsqueeze(0).to(device),
    }
    return model(**kwargs)[0].detach().cpu().numpy().reshape(-1)


def main():
    """Encode extracted entities."""
    if not os.path.exists(PATH_DWIE_LINK_MERIT_PREDS):
        os.makedirs(PATH_DWIE_LINK_MERIT_PREDS)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Sets torch.cuda behavior
    tokenizer = load_tokenizer(TOKENIZER_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, tokenizer, config["init"], device)
    cpt = (
        torch.load(config["model_path"])
        if device.type == "cuda"
        else torch.load(config["model_path"], map_location=torch.device("cpu"))
    )
    model.load_state_dict(cpt["sd"])
    model.to(device)
    model.eval()

    for filename in tqdm(os.listdir(PATH_DWIE_RE_ATLOP_PREDS)):
        with open(
            os.path.join(PATH_DWIE_RE_ATLOP_PREDS, filename), "r", encoding="utf-8"
        ) as file_re:
            data = json.load(file_re)
        with open(
            os.path.join(PATH_DWIE_DATA, filename), "r", encoding="utf-8"
        ) as file_data:
            text = json.load(file_data)["content"]

        new_line_offset = text.rfind("\n")
        data["tokenized"] = []
        data["encoded"] = []
        for v_set in data["vertexSet"]:
            spans = get_spans(v_set, new_line_offset)
            v_samples = tokenize_text_for_vertex(tokenizer, spans, text)
            data["tokenized"].append([sample["text"] for sample in v_samples])
            data["encoded"].append(
                [
                    get_embedding(sample["text"], tokenizer, model, device)
                    for sample in v_samples
                ]
            )

        with open(os.path.join(PATH_DWIE_LINK_MERIT_PREDS, filename), "wb") as out_file:
            pickle.dump(data, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default=PATH_DWIE_DATA,
        help="Path to the dwie dataset",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=PATH_RETRIEVER_CONFIG,
        help="Path to the config file for training",
    )
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    # Set environment variables before all else.
    main()
