import json
import os
import requests

import numpy as np
import ray  # pip install ray

SPLIT = "train"  # "train", "val", "test"
LOCAL_DATA_PATH = os.path.join("data", SPLIT)


def collect_pic_ids(data, name: str = "test"):
    pic_ids = []
    for item in data:
        for pic_id in item["pics"]:
            pic_ids.append(pic_id)
    np.save(f"{name}_pic_ids.npy", pic_ids)
    return pic_ids


@ray.remote
def download_image(pic_id):
    url = f"https://lh5.googleusercontent.com/p/{pic_id}"
    if not os.path.exists(LOCAL_DATA_PATH):
        os.makedirs(LOCAL_DATA_PATH)
    pic_path = os.path.join(LOCAL_DATA_PATH, f"{pic_id}.jpg")

    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(pic_path, "wb") as f:
            f.write(r.content)
        return None
    except requests.RequestException as e:
        return f"Failed to download {pic_id}: {e}"


if __name__ == "__main__":
    with open("filter_all_t.json", "r") as f:
        data = json.load(f)

    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]

    # collect_pic_ids(train_data, "train")
    # collect_pic_ids(val_data, "val")
    # collect_pic_ids(test_data, "test")

    ray.init()
    pic_ids = np.load(f"{SPLIT}_pic_ids.npy", allow_pickle=True)
    futures = [download_image.remote(pic_id) for pic_id in pic_ids]

    retry = []
    for i, result in enumerate(ray.get(futures)):
        if i % 1000 == 0:
            print(f"Completed: {i} images")
        if result is not None:
            retry.append(pic_ids[i])

    print(
        f"Failed to download {len(retry)} images"
    )  # Upon inspection, these are because they are no longer available
