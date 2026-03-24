# src/train.py

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm                      # progress bar
from huggingface_hub import hf_hub_download

# import our own files
sys.path.insert(0, os.path.dirname(__file__))
from dataset import STGLDataset


# ============================================================
# 1. LOAD CONFIG
# ============================================================

def load_config(config_path="configs/config.yml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print("Config loaded:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    return cfg


# ============================================================
# 2. LOAD MODEL FROM HUGGING FACE
# ============================================================

def load_model(cfg):
    """
    Downloads ThermalGen-B/2 pretrained weights from Hugging Face
    and loads them into the model.

    We clone the ThermalGen GitHub repo to get the model class,
    then load the pretrained .safetensors weights on top.
    """
    print("\nSetting up ThermalGen model...")

    # clone ThermalGen repo if not already present
    # this gives us the model architecture code
    repo_dir = "ThermalGen"
    if not os.path.exists(repo_dir):
        print("Cloning ThermalGen repository...")
        os.system("git clone https://github.com/arplaboratory/ThermalGen.git")
    sys.path.insert(0, repo_dir)

    # download pretrained weights from Hugging Face
    ckpt_dir = cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    weights_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        print("Downloading pretrained weights from Hugging Face...")
        weights_path = hf_hub_download(
            repo_id=cfg["hf_model_id"],           # xjh19972/ThermalGen-B-2
            filename=cfg["hf_model_file"],         # model.safetensors
            local_dir=ckpt_dir,
        )
        print(f"Weights saved to: {weights_path}")

    # load ThermalGen's own model class from the cloned repo
    try:
        from ThermalGen.models.sit import SiT_models
        model_name = "SiT-B/2"                    # matches sit_b2 in config
        model = SiT_models[model_name](
            input_size=cfg["image_size"],
            in_channels=4,                         # 4 = 3 RGB + 1 latent channel
        )

        # load pretrained weights
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained weights loaded: {weights_path}")

    except Exception as e:
        print(f"WARNING: Could not load ThermalGen model class: {e}")
        print("Falling back to simple CNN placeholder for pipeline testing...")
        model = _placeholder_model(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    return model


def _placeholder_model(cfg):
    """
    A tiny CNN used only if ThermalGen fails to load.
    Keeps the pipeline runnable so you can debug other parts.
    Input:  [B, 3, 64, 64] RGB
    Output: [B, 3, 64, 64] predicted thermal
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 3, kernel_size=3, padding=1),
        nn.Tanh()                # output in [-1, 1] matching our normalization
    )


# ============================================================
# 3. ALPHAEARTH PROJECTION LAYER
# ============================================================

class AlphaEarthProjection(nn.Module):
    """
    Projects the 64-dim AlphaEarth embedding into the same
    space the model uses for conditioning.

    Think of it as a translator:
      AlphaEarth speaks 64-dim geospatial language
      ThermalGen speaks 384-dim model language
      This layer translates between them
    """
    def __init__(self, ae_dim=64, model_dim=384):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ae_dim, model_dim),
            nn.SiLU(),                      # smooth activation
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, ae_embed):
        # ae_embed: [B, 64] → output: [B, 384]
        return self.proj(ae_embed)


# ============================================================
# 4. LOSS FUNCTION
# ============================================================

def flow_matching_loss(predicted_velocity, target_velocity):
    """
    Flow matching loss — the core of how ThermalGen learns.

    Instead of comparing images directly, we compare the
    'direction of change' (velocity) the model predicts
    versus the actual direction from noisy → clean image.

    MSE loss on velocities is simple and effective here.
    """
    return nn.functional.mse_loss(predicted_velocity, target_velocity)


# ============================================================
# 5. ONE TRAINING STEP
# ============================================================

def train_step(model, ae_proj, batch, optimizer, cfg, device):
    """
    Everything that happens for one batch:
      1. move data to device (CPU in our case)
      2. add noise to thermal image (flow matching setup)
      3. forward pass through model
      4. compute loss
      5. backprop + update weights
    """
    rgb      = batch["rgb"].to(device)        # [B, 3, 64, 64]
    thermal  = batch["thermal"].to(device)    # [B, 3, 64, 64]  (target)
    ae_embed = batch["alphaearth"].to(device) # [B, 64]

    B = rgb.shape[0]

    # --- flow matching: sample random timestep t ∈ [0, 1] ---
    t = torch.rand(B, device=device)          # random noise level per sample

    # --- create noisy thermal: interpolate between noise and clean ---
    noise   = torch.randn_like(thermal)       # pure random noise
    z_t     = (1 - t.view(B,1,1,1)) * noise + t.view(B,1,1,1) * thermal

    # --- target velocity: direction from noise → clean thermal ---
    target_velocity = thermal - noise         # what the model should predict

    # --- project AlphaEarth embedding ---
    ae_context = ae_proj(ae_embed)            # [B, 64] → [B, 384]

    # --- forward pass ---
    # style_idx = 0 for Boson dataset (index from config)
    style_idx = torch.zeros(B, dtype=torch.long, device=device)

    try:
        # full ThermalGen forward pass
        pred_velocity = model(z_t, rgb, t, style_idx)
    except TypeError:
        # placeholder CNN fallback (doesn't use t or style_idx)
        pred_velocity = model(rgb)

    # --- compute loss ---
    loss = flow_matching_loss(pred_velocity, target_velocity)

    # --- backpropagation ---
    optimizer.zero_grad()    # clear gradients from last step
    loss.backward()          # compute new gradients

    # gradient clipping — prevents instability on CPU
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(ae_proj.parameters()),
        max_norm=cfg["grad_clip"]
    )

    optimizer.step()         # update weights

    return loss.item()       # return scalar loss value for logging


# ============================================================
# 6. MAIN TRAINING LOOP
# ============================================================

def train(config_path="configs/config.yml"):

    # --- setup ---
    cfg    = load_config(config_path)
    device = torch.device(cfg["device"])     # "cpu"
    torch.manual_seed(cfg["seed"])           # reproducibility

    print(f"\nDevice: {device}")
    print(f"Image size: {cfg['image_size']}x{cfg['image_size']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Epochs: {cfg['epochs']}\n")

    # --- dataset ---
    print("Loading datasets...")
    train_dataset = STGLDataset(config_path, split="train")
    val_dataset   = STGLDataset(config_path, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],      # 0 on Windows CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # --- model + AlphaEarth projection ---
    model   = load_model(cfg).to(device)
    ae_proj = AlphaEarthProjection(
        ae_dim=cfg["alphaearth_dim"],        # 64
        model_dim=384                         # ThermalGen-B hidden size
    ).to(device)

    # --- optimizer ---
    # combine model + ae_proj parameters so both get updated
    all_params = list(model.parameters()) + list(ae_proj.parameters())
    optimizer  = torch.optim.AdamW(
        all_params,
        lr=cfg["learning_rate"],             # 0.0001
        weight_decay=1e-4                    # mild regularization
    )

    # --- training loop ---
    print(f"\nStarting training for {cfg['epochs']} epochs...")
    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):

        # --- train ---
        model.train()
        ae_proj.train()
        train_losses = []

        for step, batch in enumerate(tqdm(train_loader,
                                          desc=f"Epoch {epoch}/{cfg['epochs']}")):
            loss = train_step(model, ae_proj, batch, optimizer, cfg, device)
            train_losses.append(loss)

            if step % cfg["log_every"] == 0:
                print(f"  step {step:4d} | loss: {loss:.4f}")

        avg_train_loss = sum(train_losses) / len(train_losses)

        # --- validate ---
        model.eval()
        ae_proj.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                rgb      = batch["rgb"].to(device)
                thermal  = batch["thermal"].to(device)
                ae_embed = batch["alphaearth"].to(device)
                B        = rgb.shape[0]
                t        = torch.rand(B, device=device)
                noise    = torch.randn_like(thermal)
                z_t      = (1 - t.view(B,1,1,1)) * noise + t.view(B,1,1,1) * thermal
                target_v = thermal - noise
                style_idx = torch.zeros(B, dtype=torch.long, device=device)

                try:
                    pred_v = model(z_t, rgb, t, style_idx)
                except TypeError:
                    pred_v = model(rgb)

                val_losses.append(flow_matching_loss(pred_v, target_v).item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"\nEpoch {epoch:3d} | "
              f"train loss: {avg_train_loss:.4f} | "
              f"val loss: {avg_val_loss:.4f}")

        # --- save checkpoint every N epochs ---
        if epoch % cfg["save_every"] == 0:
            ckpt_path = os.path.join(
                cfg["checkpoint_dir"],
                f"epoch_{epoch:04d}.pt"
            )
            torch.save({
                "epoch":          epoch,
                "model":          model.state_dict(),
                "ae_proj":        ae_proj.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "train_loss":     avg_train_loss,
                "val_loss":       avg_val_loss,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        # --- save best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(cfg["checkpoint_dir"], "best_model.pt")
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "ae_proj":    ae_proj.state_dict(),
                "val_loss":   best_val_loss,
            }, best_path)
            print(f"New best model saved (val loss: {best_val_loss:.4f})")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(cfg['checkpoint_dir'], 'best_model.pt')}")


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train("configs/config.yml")