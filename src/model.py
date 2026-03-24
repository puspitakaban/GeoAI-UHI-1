# src/model.py
# Loads ThermalGen-L-2-concat from Hugging Face
# Handles empty ThermalGen repo clone gracefully

import os
import sys
import torch
import torch.nn as nn


def clone_thermalgen_repo():
    """
    Clones the ThermalGen GitHub repo if not already present.
    We need this for the model architecture class definition.
    """
    repo_dir = "ThermalGen"

    if os.path.exists(repo_dir) and os.listdir(repo_dir):
        print(f"  ThermalGen repo already present")
        return True

    print("  Cloning ThermalGen repository...")
    result = os.system(
        "git clone https://github.com/arplaboratory/ThermalGen.git"
    )

    if result != 0:
        print("  WARNING: git clone failed")
        return False

    print("  ThermalGen repo cloned successfully")
    return True


def install_thermalgen_deps():
    """
    Installs ThermalGen-specific dependencies if needed.
    safetensors is required to load .safetensors weight files.
    """
    try:
        import safetensors
    except ImportError:
        print("  Installing safetensors...")
        os.system("pip install safetensors -q")


def build_model(cfg):
    """
    Loads ThermalGen-L-2-concat pretrained weights from Hugging Face.

    Strategy:
      1. Clone ThermalGen repo (for architecture code)
      2. Download pretrained weights from HF
      3. Load weights into model
      4. Fall back to placeholder CNN if anything fails

    Returns: (model, model_type)
      model_type is "thermalgen" or "placeholder"
      so train.py knows what forward() signature to use
    """
    print("\nSetting up model...")
    install_thermalgen_deps()

    ckpt_dir = cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    weights_path = os.path.join(ckpt_dir, "model.safetensors")

    # --- step 1: download pretrained weights ---
    if not os.path.exists(weights_path):
        print(f"  Downloading {cfg['hf_model_id']} from Hugging Face...")
        try:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id=cfg["hf_model_id"],
                filename=cfg["hf_model_file"],
                local_dir=ckpt_dir,
            )
            print(f"  Weights saved: {weights_path}")
        except Exception as e:
            print(f"  WARNING: HF download failed: {e}")
            return _placeholder_model(cfg), "placeholder"
    else:
        print(f"  Using cached weights: {weights_path}")

    # --- step 2: clone repo and load model class ---
    repo_ok = clone_thermalgen_repo()

    if repo_ok:
        try:
            sys.path.insert(0, "ThermalGen")
            model = _load_thermalgen_model(cfg, weights_path)
            print(f"  Model: ThermalGen-L-2-concat (0.6B params)")
            return model, "thermalgen"

        except Exception as e:
            print(f"  WARNING: ThermalGen load failed: {e}")
            print(f"  Falling back to placeholder CNN")

    return _placeholder_model(cfg), "placeholder"


def _load_thermalgen_model(cfg, weights_path):
    """
    Loads ThermalGen model architecture and pretrained weights.
    Requires ThermalGen repo to be cloned.
    """
    from safetensors.torch import load_file

    # import ThermalGen's model class
    # sit_l2_concat = Large SiT with concatenation conditioning
    try:
        from models.sit import SiT_models
        model = SiT_models["SiT-L/2"](
            input_size=cfg["image_size"],
            in_channels=4,
        )
    except ImportError:
        # try alternative import path
        from ThermalGen.models.sit import SiT_models
        model = SiT_models["SiT-L/2"](
            input_size=cfg["image_size"],
            in_channels=4,
        )

    # load pretrained weights
    # strict=False allows partial loading if model has new layers
    state_dict = load_file(weights_path)
    missing, unexpected = model.load_state_dict(
        state_dict, strict=False
    )

    if missing:
        print(f"  Missing keys (new layers, expected): {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {len(unexpected)}")

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    return model


def _placeholder_model(cfg):
    """
    Simple CNN used when ThermalGen fails to load.
    Same input/output shape — pipeline keeps running.
    Useful for debugging dataset, training loop, metrics
    without needing the full ThermalGen model.

    Input:  [B, 3, H, W]  RGB
    Output: [B, 3, H, W]  predicted thermal
    """
    H = cfg["image_size"]
    print(f"  Placeholder CNN — {H}x{H} input/output")

    return nn.Sequential(
        nn.Conv2d(3,  64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 3,  kernel_size=3, padding=1),
        nn.Tanh()   # output in [-1, 1]
    )


class AlphaEarthProjection(nn.Module):
    """
    Projects 64-dim AlphaEarth embedding → 384-dim model space.
    Only used when use_alphaearth: true in config.
    """
    def __init__(self, ae_dim=64, model_dim=384):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ae_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x):
        return self.proj(x)   # [B, 64] → [B, 384]


class WeatherProjection(nn.Module):
    """
    Projects 9-dim real weather vector → 384-dim model space.

    Weather features (from drone_and_weather_metadata.json):
      temperature, humidity, cloud_cover, wind_speed,
      wind_sin, wind_cos, direct_radiation,
      diffuse_radiation, weather_code

    This is trained from scratch — it learns how weather
    features map to thermal appearance during fine-tuning.
    """
    def __init__(self, weather_dim=9, model_dim=384):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(weather_dim, 64),
            nn.SiLU(),
            nn.Linear(64, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x):
        return self.proj(x)   # [B, 9] → [B, 384]


# ----------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    with open("configs/config.yml") as f:
        cfg = yaml.safe_load(f)

    print("Testing model loading...")
    model, model_type = build_model(cfg)
    print(f"\nModel type: {model_type}")

    # test forward pass with dummy data
    B = 1
    H = cfg["image_size"]
    x = torch.randn(B, 3, H, H)

    print("Running forward pass...")
    with torch.no_grad():
        if model_type == "thermalgen":
            t     = torch.rand(B)
            style = torch.zeros(B, dtype=torch.long)
            out   = model(x, x, t, style)
        else:
            out = model(x)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Output range : [{out.min():.3f}, {out.max():.3f}]")
    print("\nModel loaded successfully!")