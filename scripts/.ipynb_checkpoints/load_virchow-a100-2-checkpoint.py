import logging
logging.basicConfig(
    filename="load_virchow_a100_2.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger()


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import h5py
import os
import timm
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch.utils.data import DataLoader, Dataset


# In[3]:


model = timm.create_model(
    "hf-hub:paige-ai/Virchow2",
    pretrained=True,
    mlp_layer=SwiGLUPacked,
    act_layer=torch.nn.SiLU
)
model.eval().cuda()

config = resolve_data_config(model.pretrained_cfg, model=model)
transform = create_transform(**config)


# In[4]:


class TileFolderDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx]


# In[5]:


# ───────────── Embedding + Saving ─────────────
def extract_and_save(tile_folder, h5_output_path, batch_size=96):
    dataset = TileFolderDataset(tile_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    all_embeddings = []
    all_coords = []

    for batch_imgs, batch_paths in tqdm(dataloader, desc=os.path.basename(tile_folder)):
        batch_imgs = batch_imgs.cuda()
        with torch.no_grad():
            out = model(batch_imgs)  # shape: (B, 261, 1280)

        cls = out[:, 0]
        patch_tokens = out[:, 5:]  # skip register tokens
        mean = patch_tokens.mean(dim=1)
        embedding = torch.cat([cls, mean], dim=-1)  # (B, 2560)
        all_embeddings.append(embedding.cpu())

        # Extract x, y from filename (e.g., TCGA-XX_L1_1232_2048.png)
        for path in batch_paths:
            base = os.path.splitext(os.path.basename(path))[0]
            try:
                x, y = map(int, base.split("_")[-2:])
            except:
                x, y = 0, 0
            all_coords.append((x, y))

    all_embeddings = torch.cat(all_embeddings, dim=0)     # (N, 2560)
    all_coords = torch.tensor(all_coords)                 # (N, 2)

    with h5py.File(h5_output_path, "w") as f:
        f.create_dataset("features", data=all_embeddings.numpy())
        f.create_dataset("coords", data=all_coords.numpy())

    log.info(f"✅ Saved {all_embeddings.shape[0]} embeddings to {h5_output_path}")


# In[ ]:


tile_root_dir = "virchow_tiles_gpu1"               # root directory containing subfolders for each WSI
output_dir = "virchow_features"                   # where to save .h5 files
os.makedirs(output_dir, exist_ok=True)

for slide_folder in sorted(os.listdir(tile_root_dir)):
    slide_path = os.path.join(tile_root_dir, slide_folder)
    if not os.path.isdir(slide_path):
        continue  # skip files

    h5_output_path = os.path.join(output_dir, f"{slide_folder}.h5")

    if os.path.exists(h5_output_path):
        log.info(f"✅ Skipping {slide_folder}, already exists.")
        continue

    try:
        extract_and_save(slide_path, h5_output_path, batch_size=96)
    except Exception as e:
        log.info(f"❌ Failed to process {slide_folder}: {e}")

