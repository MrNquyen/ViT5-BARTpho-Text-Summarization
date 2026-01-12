import torch
import json
import os

from torch.utils.data import Dataset, DataLoader

from utils.utils import load_json, load_npy
from icecream import ic
from tqdm import tqdm

#----------DATASET----------
class ViInforgraphicDataset(Dataset):
    def __init__(self, dataset_config, split):
        super().__init__()

        #-- Load features directory
        imdb_path = dataset_config["imdb_files"][split]

        #-- Load data
        imdb = load_npy(imdb_path)
        self.data = []
        for item in tqdm(imdb, desc=f"Loading {split} split"):
            im_id = item["image_id"]
            #-- Load data
            self.data.append({
                "id": im_id,
                "im_path": item["image_path"],
                "caption": item["caption_str"],
                "ocr_description": item["ocr_description"].replace("\n", " "),
            })



    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    list_id = [item["id"] for item in batch]
    list_captions = [item["caption"] for item in batch]
    list_ocr_descriptions = [item["ocr_description"] for item in batch]
    
    return {
        "list_id": list_id,
        "list_captions": list_captions,
        "list_ocr_descriptions": list_ocr_descriptions,
    }


def get_loader(dataset_config, batch_size, split):
    if split not in ["train", "val", "test"]:
        raise ValueError(f"No split found for {split}")
    dataset = ViInforgraphicSummarizeDataset(dataset_config, split)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=split=="train",
        collate_fn=collate_fn,
        num_workers=4,  # Add multiple workers
        pin_memory=True
    )
    return dataloader
