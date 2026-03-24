# src/dataset_aa.py
# Ann Arbor drone dataset with official pairing + real weather metadata

import os
import json
import math
import yaml
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

Image.MAX_IMAGE_PIXELS = None


class AnnArborDataset(Dataset):
    def __init__(self, config_path, split="train"):

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.split       = split
        self.image_size  = self.cfg["image_size"]
        self.register    = self.cfg.get("register_images", True)
        self.use_ae      = self.cfg.get("use_alphaearth", False)
        self.use_weather = self.cfg.get("use_weather", False)
        self.ae_dim      = self.cfg.get("alphaearth_dim", 64)
        self.weather_dim = self.cfg.get("weather_dim", 9)

        train_rgb_dir     = self.cfg["train_rgb_dir"]
        train_thermal_dir = self.cfg["train_thermal_dir"]
        test_rgb_dir      = self.cfg["test_rgb_dir"]
        val_split         = self.cfg.get("val_split", 0.15)

        # --- load official pairing map ---
        split_path = self.cfg.get("train_test_split")
        with open(split_path) as f:
            self.split_map = json.load(f)
        print(f"Loaded pairing map: {len(self.split_map)} entries")

        # --- load weather metadata ---
        self.weather_meta = {}
        if self.use_weather:
            meta_path = self.cfg.get("weather_metadata")
            with open(meta_path) as f:
                self.weather_meta = json.load(f)
            print(f"Loaded weather metadata: {len(self.weather_meta)} entries")

        # --- build paired file list using split_map ---
        if split in ["train", "val"]:
            all_pairs = self._build_pairs_from_map(
                train_rgb_dir, train_thermal_dir
            )
            random.seed(self.cfg.get("seed", 42))
            random.shuffle(all_pairs)
            n_val = max(1, int(len(all_pairs) * val_split))

            if split == "train":
                self.pairs = all_pairs[n_val:]
            else:
                self.pairs = all_pairs[:n_val]

        elif split == "test":
            self.pairs = self._build_test_pairs(test_rgb_dir)

        print(f"Split: {split} → {len(self.pairs)} image pairs")

        # --- build patch index (lazy — no images loaded yet) ---
        print("Building patch index...")
        self.patch_index = []   # (pair_idx, left, top)

        for pair_idx, pair in enumerate(self.pairs):
            with Image.open(pair["rgb"]) as img:
                w, h = img.size
            patch_size = self.image_size
            for top in range(0, h - patch_size + 1, patch_size):
                for left in range(0, w - patch_size + 1, patch_size):
                    self.patch_index.append((pair_idx, left, top))

        print(f"Total patches: {len(self.patch_index)}")

        # --- transforms ---
        self.to_tensor    = transforms.ToTensor()
        self.normalize    = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1
        )

        # --- image cache ---
        self._cache     = {}
        self._cache_max = 4

    # ----------------------------------------------------------------
    # Build pairs using official split_map
    # ----------------------------------------------------------------

    def _build_pairs_from_map(self, rgb_dir, thermal_dir):
        """
        Uses train_test_split.json to build exact pairs.
        Format: {"0.JPG": ["thermal_orig.JPG", "rgb_orig.JPG"]}
        Files in your folders are already renamed to 0.JPG, 1.JPG...
        """
        pairs = []
        missing = 0

        for short_name, (thermal_orig, rgb_orig) in self.split_map.items():
            rgb_path     = os.path.join(rgb_dir,     short_name)
            thermal_path = os.path.join(thermal_dir, short_name)

            if os.path.exists(rgb_path) and os.path.exists(thermal_path):
                pairs.append({
                    "num":           short_name,
                    "rgb":           rgb_path,
                    "thermal":       thermal_path,
                    "thermal_orig":  thermal_orig,  # for weather lookup
                    "rgb_orig":      rgb_orig,
                })
            else:
                missing += 1

        print(f"  Pairs found : {len(pairs)}")
        if missing > 0:
            print(f"  Missing     : {missing} (files not on disk)")

        return pairs

    def _build_test_pairs(self, test_rgb_dir):
        """Test split — RGB only, lookup thermal_orig from split_map."""
        pairs = []
        for short_name, (thermal_orig, rgb_orig) in self.split_map.items():
            rgb_path = os.path.join(test_rgb_dir, short_name)
            if os.path.exists(rgb_path):
                pairs.append({
                    "num":          short_name,
                    "rgb":          rgb_path,
                    "thermal":      None,
                    "thermal_orig": thermal_orig,
                    "rgb_orig":     rgb_orig,
                })
        return pairs

    # ----------------------------------------------------------------
    # Weather vector from metadata
    # ----------------------------------------------------------------

    def _get_weather_vector(self, thermal_orig_name):
        """
        Builds a 9-dim normalized weather vector for one image.
        Uses the original DJI thermal filename as key.

        Features:
          0: temperature_2m        (°C, normalized -40 to 50)
          1: relative_humidity_2m  (%, normalized 0-100)
          2: total_cloud_cover     (%, normalized 0-100)
          3: wind_speed_10m        (km/h, normalized 0-60)
          4: wind_direction sin    (encodes direction as sin)
          5: wind_direction cos    (encodes direction as cos)
          6: direct_radiation      (W/m², normalized 0-1000)
          7: diffuse_radiation     (W/m², normalized 0-300)
          8: weather_code          (WMO code, normalized 0-99)
        """
        if not self.use_weather or thermal_orig_name not in self.weather_meta:
            return torch.zeros(self.weather_dim)

        m = self.weather_meta[thermal_orig_name]

        # encode wind direction as sin/cos
        # this avoids the 359°→0° discontinuity problem
        wind_dir_rad = math.radians(m.get("wind_direction_10m", 0))
        wind_sin     = math.sin(wind_dir_rad)
        wind_cos     = math.cos(wind_dir_rad)

        def norm(val, lo, hi):
            """Normalize value to [-1, 1] range."""
            return max(-1.0, min(1.0, 2 * (val - lo) / (hi - lo) - 1))

        vec = [
            norm(m.get("temperature_2m",        0),   -40,  50),
            norm(m.get("relative_humidity_2m",   0),     0, 100),
            norm(m.get("total_cloud_cover",      0),     0, 100),
            norm(m.get("wind_speed_10m",         0),     0,  60),
            wind_sin,                                          # already in [-1,1]
            wind_cos,                                          # already in [-1,1]
            norm(m.get("direct_radiation",       0),     0, 1000),
            norm(m.get("diffuse_radiation",      0),     0,  300),
            norm(m.get("weather_code",           0),     0,   99),
        ]

        return torch.tensor(vec, dtype=torch.float32)

    # ----------------------------------------------------------------
    # Lazy image loading
    # ----------------------------------------------------------------

    def _load_pair(self, pair_idx):
        if pair_idx in self._cache:
            return self._cache[pair_idx]

        pair    = self.pairs[pair_idx]
        rgb_img = Image.open(pair["rgb"]).convert("RGB")

        if pair.get("thermal"):
            thermal_img = Image.open(pair["thermal"]).convert("RGB")
            if self.register:
                thermal_img = self._register(rgb_img, thermal_img)
        else:
            thermal_img = Image.new("RGB", rgb_img.size, (0, 0, 0))

        if len(self._cache) >= self._cache_max:
            del self._cache[next(iter(self._cache))]

        self._cache[pair_idx] = (rgb_img, thermal_img)
        return rgb_img, thermal_img

    def _register(self, rgb_img, thermal_img):
        return thermal_img.resize(rgb_img.size, Image.LANCZOS)

    # ----------------------------------------------------------------
    # Augmentation
    # ----------------------------------------------------------------

    def _augment(self, rgb_patch, thermal_patch):
        if random.random() > 0.5:
            rgb_patch     = TF.hflip(rgb_patch)
            thermal_patch = TF.hflip(thermal_patch)
        if random.random() > 0.7:
            rgb_patch     = TF.vflip(rgb_patch)
            thermal_patch = TF.vflip(thermal_patch)
        if random.random() > 0.5:
            angle         = random.uniform(-15, 15)
            rgb_patch     = TF.rotate(rgb_patch, angle)
            thermal_patch = TF.rotate(thermal_patch, angle)
        if random.random() > 0.5:
            rgb_patch = self.color_jitter(rgb_patch)
        if random.random() > 0.7:
            arr       = np.array(rgb_patch, dtype=np.float32)
            noise     = np.random.normal(0, 6, arr.shape)
            arr       = np.clip(arr + noise, 0, 255).astype(np.uint8)
            rgb_patch = Image.fromarray(arr)
        return rgb_patch, thermal_patch

    # ----------------------------------------------------------------
    # PyTorch interface
    # ----------------------------------------------------------------

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, index):
        pair_idx, left, top = self.patch_index[index]
        pair = self.pairs[pair_idx]

        rgb_img, thermal_img = self._load_pair(pair_idx)

        box           = (left, top,
                         left + self.image_size,
                         top  + self.image_size)
        rgb_patch     = rgb_img.crop(box)
        thermal_patch = thermal_img.crop(box)

        if self.split == "train":
            rgb_patch, thermal_patch = self._augment(rgb_patch, thermal_patch)

        rgb_tensor     = self.normalize(self.to_tensor(rgb_patch))
        thermal_tensor = self.normalize(self.to_tensor(thermal_patch))

        # get real weather vector for this specific image
        weather_vec = self._get_weather_vector(pair["thermal_orig"])

        return {
            "rgb":        rgb_tensor,
            "thermal":    thermal_tensor,
            "alphaearth": torch.zeros(self.ae_dim),
            "weather":    weather_vec,             # real per-image weather!
            "lat":        pair.get("lat", 0.0),    # GPS lat (from metadata)
            "lng":        pair.get("lng", 0.0),    # GPS lng (from metadata)
        }


# ----------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("Testing AnnArborDataset with metadata files...")
    ds = AnnArborDataset("configs/config.yml", split="train")

    print(f"\nTrain patches: {len(ds)}")
    sample = ds[0]
    print(f"\nSample:")
    print(f"  rgb shape    : {sample['rgb'].shape}")
    print(f"  thermal shape: {sample['thermal'].shape}")
    print(f"  weather      : {sample['weather']}")
    print(f"  weather dim  : {sample['weather'].shape}")
    print("\nAll good!")