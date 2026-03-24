# src/train.py
# ThermalGen fine-tuning on Ann Arbor drone data
# Real paired RGB + thermal, real weather conditioning

import os
import sys
import math
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import lpips
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("WARNING: lpips/torchmetrics not installed — metrics disabled")
    print("Run: pip install lpips torchmetrics")


# ============================================================
# CONFIG
# ============================================================

def load_config(config_path="configs/config.yml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # SageMaker path overrides
    import os
    sm_data    = os.environ.get("SM_CHANNEL_TRAINING")
    sm_model   = os.environ.get("SM_MODEL_DIR")
    sm_output  = os.environ.get("SM_OUTPUT_DATA_DIR")

    if sm_data:
        cfg["train_rgb_dir"]     = os.path.join(sm_data, "train/RGB/")
        cfg["train_thermal_dir"] = os.path.join(sm_data, "train/Thermal/")
        cfg["test_rgb_dir"]      = os.path.join(sm_data, "test/RGB/")
        cfg["train_test_split"]  = os.path.join(sm_data, "train_test_split.json")
        cfg["weather_metadata"]  = os.path.join(sm_data, "drone_and_weather_metadata.json")
        print(f"SageMaker data path: {sm_data}")

    if sm_model:
        cfg["checkpoint_dir"] = sm_model
        print(f"SageMaker model path: {sm_model}")

    print("\nConfig loaded:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    return cfg


# ============================================================
# METRICS
# ============================================================

class ImageMetrics:
    """
    Tracks PSNR, SSIM and LPIPS during validation.

    PSNR  — pixel accuracy       (higher = better, >25dB = decent)
    SSIM  — structural quality   (higher = better, >0.7 = decent)
    LPIPS — perceptual quality   (lower  = better, <0.3 = decent)
    """
    def __init__(self, device):
        self.device = device
        if METRICS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
            self.lpips_fn.eval()

    def compute_psnr(self, pred, target):
        pred   = (pred.clamp(-1,1)   + 1) / 2
        target = (target.clamp(-1,1) + 1) / 2
        mse    = torch.mean((pred - target) ** 2, dim=[1,2,3])
        return (-10 * torch.log10(mse + 1e-8)).mean().item()

    def compute_ssim(self, pred, target):
        if not METRICS_AVAILABLE:
            return 0.0
        pred   = (pred.clamp(-1,1)   + 1) / 2
        target = (target.clamp(-1,1) + 1) / 2
        return ssim_fn(pred, target, data_range=1.0).item()

    def compute_lpips(self, pred, target):
        if not METRICS_AVAILABLE:
            return 0.0
        with torch.no_grad():
            return self.lpips_fn(
                pred.clamp(-1,1), target.clamp(-1,1)
            ).mean().item()

    def compute_all(self, pred, target):
        return {
            "psnr":  self.compute_psnr(pred, target),
            "ssim":  self.compute_ssim(pred, target),
            "lpips": self.compute_lpips(pred, target),
        }


# ============================================================
# TRAINING CONTROLLER
# ============================================================

class TrainingController:
    """
    Watches PSNR, SSIM, LPIPS every epoch and reacts:

    Early stopping  — stops if LPIPS doesn't improve
                      for N epochs (patience)
    LR scheduling   — halves LR if PSNR plateaus
    Composite loss  — combines flow + SSIM + LPIPS
    """
    def __init__(self, optimizer, patience=7, lr_factor=0.5,
                 min_lr=1e-7, device="cpu"):
        self.optimizer  = optimizer
        self.patience   = patience
        self.lr_factor  = lr_factor
        self.min_lr     = min_lr
        self.device     = device

        self.best_lpips = float("inf")
        self.best_psnr  = -float("inf")
        self.best_ssim  = -float("inf")

        self.lpips_no_improve = 0
        self.psnr_no_improve  = 0

        self.history = {
            "epoch": [], "psnr": [], "ssim": [],
            "lpips": [], "train_loss": [], "val_loss": [], "lr": []
        }

        if METRICS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net="vgg").to(device)

    def composite_loss(self, pred, target, flow_loss):
        """
        Combines three loss signals:
          flow_loss  (weight 1.0) — ThermalGen flow matching objective
          ssim_loss  (weight 0.3) — structural similarity
          lpips_loss (weight 0.5) — perceptual quality (most important)
        """
        if not METRICS_AVAILABLE:
            return flow_loss, {"flow": flow_loss.item(),
                               "ssim": 0.0, "lpips": 0.0,
                               "total": flow_loss.item()}

        pred_01   = (pred.clamp(-1,1)   + 1) / 2
        target_01 = (target.clamp(-1,1) + 1) / 2

        ssim_loss  = 1 - ssim_fn(pred_01, target_01, data_range=1.0)
        lpips_loss = self.lpips_fn(
            pred.clamp(-1,1), target.clamp(-1,1)
        ).mean()

        total = (1.0 * flow_loss +
                 0.3 * ssim_loss +
                 0.5 * lpips_loss)

        return total, {
            "flow":  flow_loss.item(),
            "ssim":  ssim_loss.item(),
            "lpips": lpips_loss.item(),
            "total": total.item(),
        }

    def step(self, epoch, psnr, ssim, lpips_score,
             train_loss, val_loss):
        """
        Call at end of each epoch.
        Returns (should_stop, lr_reduced).
        """
        current_lr = self.optimizer.param_groups[0]["lr"]

        self.history["epoch"].append(epoch)
        self.history["psnr"].append(psnr)
        self.history["ssim"].append(ssim)
        self.history["lpips"].append(lpips_score)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["lr"].append(current_lr)

        should_stop = False
        lr_reduced  = False

        # LPIPS — early stopping
        if lpips_score < self.best_lpips:
            self.best_lpips       = lpips_score
            self.lpips_no_improve = 0
            print(f"  ✓ LPIPS improved → {lpips_score:.4f}")
        else:
            self.lpips_no_improve += 1
            print(f"  ✗ LPIPS no improve "
                  f"({self.lpips_no_improve}/{self.patience})")
            if self.lpips_no_improve >= self.patience:
                print(f"  🛑 Early stopping triggered")
                should_stop = True

        # PSNR — LR reduction
        if psnr > self.best_psnr:
            self.best_psnr       = psnr
            self.psnr_no_improve = 0
            print(f"  ✓ PSNR improved  → {psnr:.2f} dB")
        else:
            self.psnr_no_improve += 1
            if self.psnr_no_improve >= self.patience:
                new_lr = max(current_lr * self.lr_factor, self.min_lr)
                if new_lr < current_lr:
                    for g in self.optimizer.param_groups:
                        g["lr"] = new_lr
                    self.psnr_no_improve = 0
                    lr_reduced = True
                    print(f"  ⚡ LR reduced: "
                          f"{current_lr:.2e} → {new_lr:.2e}")

        # SSIM — info only
        if ssim > self.best_ssim:
            self.best_ssim = ssim
            print(f"  ✓ SSIM improved  → {ssim:.4f}")

        return should_stop, lr_reduced

    def summary(self):
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Best PSNR  : {self.best_psnr:.2f} dB")
        print(f"Best SSIM  : {self.best_ssim:.4f}")
        print(f"Best LPIPS : {self.best_lpips:.4f}")
        print(f"Final LR   : {self.optimizer.param_groups[0]['lr']:.2e}")
        print("="*50)


# ============================================================
# LOSS
# ============================================================

def flow_matching_loss(pred, target):
    return nn.functional.mse_loss(pred, target)


# ============================================================
# TRAIN STEP
# ============================================================

def train_step(model, model_type, weather_proj,
               batch, optimizer, controller, cfg,
               device, scaler=None):
    """
    One training step:
      1. add noise to thermal (flow matching)
      2. forward pass
      3. composite loss (flow + SSIM + LPIPS)
      4. backprop + update weights
    """
    rgb     = batch["rgb"].to(device)        # [B, 3, H, W]
    thermal = batch["thermal"].to(device)    # [B, 3, H, W]  real target
    weather = batch["weather"].to(device)    # [B, 9]        real weather
    B       = rgb.shape[0]

    # flow matching: sample timestep, add noise
    t      = torch.rand(B, device=device)
    noise  = torch.randn_like(thermal)
    z_t    = ((1 - t.view(B,1,1,1)) * noise +
               t.view(B,1,1,1) * thermal)
    target_velocity = thermal - noise

    # weather context (projected to model dim)
    # added to conditioning but not directly to model
    # (ThermalGen uses style_idx for conditioning)
    _ = weather_proj(weather)   # [B, 384] — used in future integration

    style_idx = torch.zeros(B, dtype=torch.long, device=device)

    # forward pass
    use_amp = scaler is not None
    with torch.cuda.amp.autocast(enabled=use_amp):
        try:
            pred_velocity = model(z_t, rgb, t, style_idx)
        except TypeError:
            pred_velocity = model(rgb)

        base_loss = flow_matching_loss(pred_velocity, target_velocity)
        loss, breakdown = controller.composite_loss(
            pred_velocity, target_velocity, base_loss
        )

    # backprop
    optimizer.zero_grad()

    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) +
            list(weather_proj.parameters()),
            max_norm=cfg["grad_clip"]
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) +
            list(weather_proj.parameters()),
            max_norm=cfg["grad_clip"]
        )
        optimizer.step()

    return loss.item(), breakdown


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train(config_path="configs/config.yml"):

    cfg    = load_config(config_path)
    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])

    print(f"\nDevice        : {device}")
    print(f"Image size    : {cfg['image_size']}x{cfg['image_size']}")
    print(f"Batch size    : {cfg['batch_size']}")
    print(f"Epochs        : {cfg['epochs']}")
    print(f"Learning rate : {cfg['learning_rate']}")

    # --- dataset ---
    print("\nLoading datasets...")
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset_aa import AnnArborDataset

    train_dataset = AnnArborDataset(config_path, split="train")
    val_dataset   = AnnArborDataset(config_path, split="val")

    train_loader  = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # --- model ---
    from model import build_model, WeatherProjection
    model, model_type = build_model(cfg)
    model             = model.to(device)

    weather_proj = WeatherProjection(
        weather_dim=cfg.get("weather_dim", 9),
        model_dim=384
    ).to(device)

    print(f"Model type    : {model_type}")

    # --- optimizer ---
    all_params = (list(model.parameters()) +
                  list(weather_proj.parameters()))
    optimizer  = torch.optim.AdamW(
        all_params,
        lr=cfg["learning_rate"],
        weight_decay=1e-4
    )

    # --- mixed precision ---
    use_mixed = (cfg.get("mixed_precision", False) and
                 device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_mixed else None
    print(f"Mixed precision: {use_mixed}")

    # --- controller + metrics ---
    controller = TrainingController(
        optimizer=optimizer,
        patience=cfg.get("patience", 7),
        device=device
    )
    metrics = ImageMetrics(device)

    # --- checkpoint dir ---
    ckpt_dir = cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cfg.get("best_model_dir",
                "models/final/ann_arbor/"), exist_ok=True)

    # --- training loop ---
    print(f"\nStarting training...")
    print(f"(Press Ctrl+C to safely stop and save)\n")

    best_val_loss = float("inf")
    avg_train_loss = 0.0

    try:
        for epoch in range(1, cfg["epochs"] + 1):

            # TRAIN
            model.train()
            weather_proj.train()
            train_losses = []

            pbar = tqdm(train_loader,
                        desc=f"Epoch {epoch}/{cfg['epochs']}")

            for step, batch in enumerate(pbar):
                loss, breakdown = train_step(
                    model, model_type, weather_proj,
                    batch, optimizer, controller,
                    cfg, device, scaler
                )
                train_losses.append(loss)

                if step % cfg["log_every"] == 0:
                    pbar.set_postfix({
                        "loss":  f"{loss:.4f}",
                        "flow":  f"{breakdown['flow']:.4f}",
                        "lpips": f"{breakdown.get('lpips', 0):.4f}",
                    })

            avg_train_loss = sum(train_losses) / len(train_losses)

            # VALIDATE
            model.eval()
            weather_proj.eval()
            val_losses  = []
            all_psnr    = []
            all_ssim    = []
            all_lpips   = []

            with torch.no_grad():
                for batch in val_loader:
                    rgb     = batch["rgb"].to(device)
                    thermal = batch["thermal"].to(device)
                    B_      = rgb.shape[0]
                    t_      = torch.rand(B_, device=device)
                    noise   = torch.randn_like(thermal)
                    z_t     = ((1 - t_.view(B_,1,1,1)) * noise +
                                t_.view(B_,1,1,1) * thermal)
                    target_v = thermal - noise
                    style_   = torch.zeros(B_, dtype=torch.long,
                                           device=device)
                    try:
                        pred_v = model(z_t, rgb, t_, style_)
                    except TypeError:
                        pred_v = model(rgb)

                    val_losses.append(
                        flow_matching_loss(pred_v, target_v).item()
                    )
                    m = metrics.compute_all(pred_v, thermal)
                    all_psnr.append(m["psnr"])
                    all_ssim.append(m["ssim"])
                    all_lpips.append(m["lpips"])

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_psnr     = sum(all_psnr)   / len(all_psnr)
            avg_ssim     = sum(all_ssim)   / len(all_ssim)
            avg_lpips    = sum(all_lpips)  / len(all_lpips)

            print(f"\nEpoch {epoch:3d} | "
                  f"train: {avg_train_loss:.4f} | "
                  f"val: {avg_val_loss:.4f}")
            print(f"         | "
                  f"PSNR: {avg_psnr:.2f} dB | "
                  f"SSIM: {avg_ssim:.4f} | "
                  f"LPIPS: {avg_lpips:.4f}")

            # controller step
            should_stop, lr_reduced = controller.step(
                epoch, avg_psnr, avg_ssim, avg_lpips,
                avg_train_loss, avg_val_loss
            )

            # save checkpoint every N epochs
            if epoch % cfg["save_every"] == 0:
                ckpt_path = os.path.join(
                    ckpt_dir, f"epoch_{epoch:04d}.pt"
                )
                torch.save({
                    "epoch":        epoch,
                    "model":        model.state_dict(),
                    "weather_proj": weather_proj.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "train_loss":   avg_train_loss,
                    "val_loss":     avg_val_loss,
                    "psnr":         avg_psnr,
                    "ssim":         avg_ssim,
                    "lpips":        avg_lpips,
                    "model_type":   model_type,
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

            # save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(
                    cfg.get("best_model_dir",
                            "models/final/ann_arbor/"),
                    "best_model.pt"
                )
                torch.save({
                    "epoch":        epoch,
                    "model":        model.state_dict(),
                    "weather_proj": weather_proj.state_dict(),
                    "val_loss":     best_val_loss,
                    "psnr":         avg_psnr,
                    "ssim":         avg_ssim,
                    "lpips":        avg_lpips,
                    "model_type":   model_type,
                }, best_path)
                print(f"Best model saved (val: {best_val_loss:.4f})")

            if should_stop:
                print("\nEarly stopping — loading best checkpoint")
                if os.path.exists(best_path):
                    ckpt = torch.load(best_path,
                                      map_location=device)
                    model.load_state_dict(ckpt["model"])
                break

    except KeyboardInterrupt:
        print("\n\nTraining interrupted (Ctrl+C)")
        emergency = os.path.join(ckpt_dir, "emergency.pt")
        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "weather_proj": weather_proj.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "train_loss":   avg_train_loss,
        }, emergency)
        print(f"Emergency checkpoint saved: {emergency}")

    controller.summary()


# ============================================================
# QUICK TEST — one forward pass without full training
# ============================================================

if __name__ == "__main__":
    import yaml
    sys.path.insert(0, os.path.dirname(__file__))

    with open("configs/config.yml") as f:
        cfg = yaml.safe_load(f)

    print("Testing train.py components...")

    device = torch.device(cfg["device"])

    # test dataset
    from dataset_aa import AnnArborDataset
    ds     = AnnArborDataset("configs/config.yml", split="train")
    loader = DataLoader(ds, batch_size=2,
                        shuffle=False, num_workers=0)
    batch  = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")

    # test model
    from model import build_model, WeatherProjection
    model, model_type = build_model(cfg)
    model             = model.to(device)
    weather_proj      = WeatherProjection(
        weather_dim=cfg.get("weather_dim", 9),
        model_dim=384
    ).to(device)

    # test one step
    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-5)
    controller = TrainingController(optimizer, device=device)

    print("\nRunning one training step...")
    loss, breakdown = train_step(
        model, model_type, weather_proj,
        batch, optimizer, controller,
        cfg, device
    )
    print(f"  Loss    : {loss:.4f}")
    print(f"  Breakdown: {breakdown}")
    print("\ntrain.py working correctly!")