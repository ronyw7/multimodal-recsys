# Generate image embeddings with CLIP

import os
import torch
import clip
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, pic_ids, image_data_path, preprocess):
        self.pic_ids = pic_ids
        self.image_data_path = image_data_path
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pic_ids)

    def __getitem__(self, idx):
        pic_id = self.pic_ids[idx]
        pic_path = os.path.join(self.image_data_path, pic_id + ".jpg")
        image = Image.open(pic_path).convert("RGB")
        image_input = self.preprocess(image)
        return pic_id, image_input


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    SPLIT = "train"  # 'train', 'val', 'test'
    train_pic_ids_valid = np.load(f"{SPLIT}_pic_ids_valid.npy")

    dataset = ImageDataset(train_pic_ids_valid, f"data/{SPLIT}", preprocess)
    batch_size = 32
    num_workers = 8

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    pics_lookup = {}

    model.eval()
    with torch.no_grad():
        for batch_pic_ids, batch_images in dataloader:
            batch_images = batch_images.to(device, non_blocking=True)

            image_features = model.encode_image(batch_images)

            image_features = image_features.cpu()

            for pic_id, image_feature in zip(batch_pic_ids, image_features):
                pics_lookup[str(pic_id)] = image_feature

    np.savez_compressed(f"embeddings_pics_{SPLIT}.npz", **pics_lookup)
