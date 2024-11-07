# Break down long text reviews into chunks and create embeddings using CLIP
import re
import json

import clip
import torch
import numpy as np


def break_text(text, max_len=77):
    """Return a list of text chunks, each of length max_len or less, eagerly breaking at punctuation."""
    words = text.split()
    curr_chunk = ""
    chunks = []

    for word in words:
        if len(curr_chunk) + len(word) + 1 <= max_len:
            curr_chunk += word + " "

            if re.search(r"[.,;!?]$", word):
                chunks.append(curr_chunk.strip())
                curr_chunk = ""
        else:
            chunks.append(curr_chunk.strip())
            curr_chunk = word + " "

    if curr_chunk:
        chunks.append(curr_chunk.strip())

    results = [chunk for chunk in chunks if len(chunk) > 5 and len(chunk) < max_len]
    return results


if __name__ == "__main__":
    SPLIT = "train"
    PROMPT_TEMPLATE = "A photo of a restaurant, "

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    with open("filter_all_t.json", "r") as f:
        data = json.load(f)

    data = data[SPLIT]

    text_lookup = {}

    for item in data:
        curr_review_key = item["user_id"] + "_" + item["business_id"]
        text_lookup[curr_review_key] = item["review_text"]
        for entry in item["history_reviews"]:
            text_lookup[entry[0]] = entry[1]

    text_embeddings = {}
    prompt_template_len = len(PROMPT_TEMPLATE)

    count = 0
    for text_id, text in text_lookup.items():
        text_chunks = break_text(text, max_len=77 - prompt_template_len)

        text = clip.tokenize(text_chunks).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = text_features.cpu()

        text_embeddings[text_id] = text_features
        count += 1
        if count % 10000 == 0:
            print(count)

    np.savez_compressed(f"embeddings_text_{SPLIT}.npz", **text_embeddings)
