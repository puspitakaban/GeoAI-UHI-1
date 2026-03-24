# src/dataset.py

import os
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
import rasterio


class STGLDataset(Dataset):
    def __init__(self, config_path, split="train"):
        """
        config_path : path to configs/config.yml
        split       : "train", "val", or "test"
        """

        # --- load config ---
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.split          = split
        self.image_size     = self.cfg["image_size"]       # 64
        self.data_root      = self.cfg["data_root"]        # data/raw/STGL/
        self.use_alphaearth = self.cfg.get("use_alphaearth", False)
        self.ae_dim         = self.cfg.get("alphaearth_dim", 64)

        # --- pick the right map file based on split ---
        split_map = {
            "train": self.cfg["train_file"],
            "val":   self.cfg["val_file"],
            "test":  self.cfg["test_file"],
        }
        filename = split_map[split]

        # --- build path to the RGB map ---
        map_path = os.path.join(
            self.data_root, "maps",
            self._get_sensor_folder(filename),
            filename
        )

        # --- load the full large RGB map ---
        print(f"Loading {split} map: {map_path}")
        self.map_image = Image.open(map_path).convert("RGB")
        w, h = self.map_image.size
        print(f"  Map size: {w}x{h} pixels")

        # --- slice RGB map into patches ---
        self.patches = self._extract_patches(self.map_image, self.image_size)
        self.patch_coords = self._extract_patch_coords(w, h, self.image_size)
        print(f"  → {len(self.patches)} patches of {self.image_size}x{self.image_size}")

        # --- load AlphaEarth embeddings if enabled ---
        self.ae_embeddings = None
        if self.use_alphaearth:
            self.ae_embeddings = self._load_alphaearth_embeddings(w, h)

        # --- image transform: PIL → tensor, normalize to [-1, 1] ---
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std= [0.5, 0.5, 0.5])
        ])

    # ----------------------------------------------------------------
    # AlphaEarth loading
    # ----------------------------------------------------------------

    def _load_alphaearth_embeddings(self, map_w, map_h):
        """
        Downloads AlphaEarth embeddings from Hugging Face (Major-TOM dataset)
        and resizes them to match our RGB map dimensions.

        Each pixel in the embedding = 64-dimensional vector.
        We resize so the embedding grid aligns 1:1 with our image patches.
        """
        ae_dir  = self.cfg.get("alphaearth_dir", "data/embeddings/alphaearth/")
        year    = self.cfg.get("alphaearth_year", 2022)
        os.makedirs(ae_dir, exist_ok=True)

        # --- try local cache first ---
        local_path = os.path.join(ae_dir, f"alphaearth_{year}.tif")

        if not os.path.exists(local_path):
            print(f"  Downloading AlphaEarth {year} embeddings from Hugging Face...")
            print(f"  (Major-TOM/Core-AlphaEarth-Embeddings — prototype subset)")
            try:
                # download metadata to find a relevant grid cell
                downloaded = hf_hub_download(
                    repo_id="Major-TOM/Core-AlphaEarth-Embeddings",
                    filename="metadata.parquet",
                    repo_type="dataset",
                    local_dir=ae_dir,
                )
                print(f"  Metadata downloaded: {downloaded}")

                # for now use the first available grid cell as a placeholder
                # in production you'd match by lat/lon of your STGL region
                import pandas as pd
                meta = pd.read_parquet(downloaded, columns=["grid_cell", "subdir"])
                first_cell = meta.iloc[0]
                subdir     = first_cell["subdir"]
                grid_cell  = first_cell["grid_cell"]

                tif_url = (
                    f"https://huggingface.co/datasets/Major-TOM/"
                    f"Core-AlphaEarth-Embeddings/resolve/main/"
                    f"{subdir}/{grid_cell}.tif"
                )
                # download the .tif embedding file
                import urllib.request
                urllib.request.urlretrieve(tif_url, local_path)
                print(f"  AlphaEarth embeddings saved to: {local_path}")

            except Exception as e:
                print(f"  WARNING: Could not download AlphaEarth embeddings: {e}")
                print(f"  Falling back to zero embeddings (model will still run)")
                # return zero tensor — safe fallback so training doesn't crash
                num_patches = len(self.patches)
                return torch.zeros(num_patches, self.ae_dim)

        # --- load the .tif file ---
        print(f"  Loading AlphaEarth embeddings from: {local_path}")
        with rasterio.open(local_path) as src:
            # shape: [64, H, W]  (64 bands = 64 embedding dimensions)
            ae_array = src.read().astype(np.float32)

        ae_bands, ae_h, ae_w = ae_array.shape
        print(f"  AlphaEarth grid: {ae_bands} dims x {ae_h}x{ae_w} pixels")

        # --- resize embedding grid to match our map size ---
        # we do this by computing one embedding per patch
        # by averaging the AE values under each patch's footprint
        patch_embeddings = self._pool_embeddings_to_patches(
            ae_array, map_w, map_h, self.image_size
        )
        print(f"  Pooled to {len(patch_embeddings)} patch embeddings")

        # normalize embeddings to [-1, 1] range
        patch_embeddings = self._normalize_embeddings(patch_embeddings)

        return patch_embeddings   # shape: [num_patches, 64]

    def _pool_embeddings_to_patches(self, ae_array, map_w, map_h, patch_size):
        """
        For each image patch, compute the average AlphaEarth embedding
        within that patch's footprint.

        ae_array : [64, ae_h, ae_w]  — full embedding grid
        returns  : [num_patches, 64] — one vector per patch
        """
        _, ae_h, ae_w = ae_array.shape
        patch_embeddings = []

        for top in range(0, map_h - patch_size + 1, patch_size):
            for left in range(0, map_w - patch_size + 1, patch_size):

                # map patch coordinates → AE grid coordinates
                ae_top   = int(top  / map_h * ae_h)
                ae_left  = int(left / map_w * ae_w)
                ae_bot   = int((top  + patch_size) / map_h * ae_h)
                ae_right = int((left + patch_size) / map_w * ae_w)

                # clamp to valid range
                ae_top   = max(0, min(ae_top,   ae_h - 1))
                ae_left  = max(0, min(ae_left,  ae_w - 1))
                ae_bot   = max(ae_top  + 1, min(ae_bot,   ae_h))
                ae_right = max(ae_left + 1, min(ae_right, ae_w))

                # average all AE pixels within this patch footprint → [64]
                region = ae_array[:, ae_top:ae_bot, ae_left:ae_right]
                vec    = region.mean(axis=(1, 2))
                patch_embeddings.append(vec)

        return torch.tensor(np.stack(patch_embeddings), dtype=torch.float32)

    def _normalize_embeddings(self, embeddings):
        """
        Normalize AlphaEarth embeddings to zero mean, unit variance.
        Important — raw AE values have arbitrary scale.
        """
        mean = embeddings.mean(dim=0, keepdim=True)
        std  = embeddings.std(dim=0, keepdim=True) + 1e-6  # avoid div by zero
        return (embeddings - mean) / std

    # ----------------------------------------------------------------
    # Patch extraction helpers
    # ----------------------------------------------------------------

    def _get_sensor_folder(self, filename):
        name = filename.lower()
        if name.startswith("bosonplus"):  return "Bosonplus"
        elif name.startswith("boson"):    return "Boson"
        elif name.startswith("dji"):      return "DJI"
        else:                             return "satellite"

    def _extract_patches(self, image, patch_size):
        w, h    = image.size
        patches = []
        for top in range(0, h - patch_size + 1, patch_size):
            for left in range(0, w - patch_size + 1, patch_size):
                patches.append(image.crop((left, top,
                                           left + patch_size,
                                           top  + patch_size)))
        return patches

    def _extract_patch_coords(self, map_w, map_h, patch_size):
        """
        Store the (left, top) pixel coordinate of every patch.
        Useful for matching patches back to their map location.
        """
        coords = []
        for top in range(0, map_h - patch_size + 1, patch_size):
            for left in range(0, map_w - patch_size + 1, patch_size):
                coords.append((left, top))
        return coords

    # ----------------------------------------------------------------
    # PyTorch Dataset interface
    # ----------------------------------------------------------------

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        """
        Returns one sample dictionary:
          rgb       : [3, 64, 64]  — input RGB patch tensor
          thermal   : [3, 64, 64]  — target (placeholder for now)
          alphaearth: [64]         — geospatial embedding for this patch
          coords    : (left, top)  — pixel location in the original map
        """
        rgb_patch  = self.patches[index]
        rgb_tensor = self.transform(rgb_patch)

        sample = {
            "rgb":     rgb_tensor,
            "thermal": rgb_tensor,         # placeholder — real pairing later
            "coords":  self.patch_coords[index],
        }

        # attach AlphaEarth embedding if available
        if self.ae_embeddings is not None:
            sample["alphaearth"] = self.ae_embeddings[index]  # [64]
        else:
            # zero vector if AE is disabled — model handles gracefully
            sample["alphaearth"] = torch.zeros(self.ae_dim)

        return sample
