# main.py
# Entry point for ThermalGen Ann Arbor
# Usage:
#   python main.py --mode train
#   python main.py --mode eval
#   python main.py --mode generate --input photo.jpg

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="ThermalGen — Ann Arbor RGB to Thermal"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "generate"],
        help="train | eval | generate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yml",
        help="path to config file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="(generate mode) path to input RGB image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generated_thermal.png",
        help="(generate mode) where to save output"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="(eval/generate) path to checkpoint .pt file"
    )
    return parser.parse_args()


# ============================================================
# TRAIN
# ============================================================

def run_train(args):
    print("=" * 55)
    print("MODE: Training on Ann Arbor drone data")
    print(f"Config     : {args.config}")
    print("=" * 55)

    sys.path.insert(0, "src")
    from train import train
    train(args.config)


# ============================================================
# EVAL
# ============================================================

def run_eval(args):
    print("=" * 55)
    print("MODE: Evaluation")
    print(f"Config     : {args.config}")
    print(f"Checkpoint : {args.checkpoint}")
    print("=" * 55)

    if args.checkpoint is None:
        # look for best model automatically
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        best_path = os.path.join(
            cfg.get("best_model_dir", "models/final/ann_arbor/"),
            "best_model.pt"
        )
        if os.path.exists(best_path):
            args.checkpoint = best_path
            print(f"Auto-found best model: {best_path}")
        else:
            print("No checkpoint found. Run training first:")
            print("  python main.py --mode train")
            return

    import yaml
    import torch
    from torch.utils.data import DataLoader

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])
    sys.path.insert(0, "src")

    from dataset_aa import AnnArborDataset
    from model import build_model, WeatherProjection

    # load model
    model, model_type = build_model(cfg)
    model             = model.to(device)
    weather_proj      = WeatherProjection(
        weather_dim=cfg.get("weather_dim", 9),
        model_dim=384
    ).to(device)

    # load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    weather_proj.load_state_dict(ckpt["weather_proj"])
    print(f"Restored from epoch {ckpt['epoch']}")
    print(f"  Val loss : {ckpt.get('val_loss', 'N/A'):.4f}")
    print(f"  PSNR     : {ckpt.get('psnr',     0):.2f} dB")
    print(f"  SSIM     : {ckpt.get('ssim',     0):.4f}")
    print(f"  LPIPS    : {ckpt.get('lpips',    0):.4f}")

    # run on val set
    val_dataset = AnnArborDataset(args.config, split="val")
    val_loader  = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    try:
        import lpips
        from torchmetrics.functional import \
            structural_similarity_index_measure as ssim_fn
        from train import ImageMetrics
        metrics    = ImageMetrics(device)
        all_psnr   = []
        all_ssim   = []
        all_lpips  = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                rgb     = batch["rgb"].to(device)
                thermal = batch["thermal"].to(device)
                B_      = rgb.shape[0]
                t_      = torch.rand(B_, device=device)
                noise   = torch.randn_like(thermal)
                z_t     = ((1 - t_.view(B_,1,1,1)) * noise +
                            t_.view(B_,1,1,1) * thermal)
                style_  = torch.zeros(B_, dtype=torch.long,
                                      device=device)
                try:
                    pred = model(z_t, rgb, t_, style_)
                except TypeError:
                    pred = model(rgb)

                m = metrics.compute_all(pred, thermal)
                all_psnr.append(m["psnr"])
                all_ssim.append(m["ssim"])
                all_lpips.append(m["lpips"])

        print(f"\nEval results on {len(val_dataset)} val patches:")
        print(f"  PSNR  : {sum(all_psnr)/len(all_psnr):.2f} dB")
        print(f"  SSIM  : {sum(all_ssim)/len(all_ssim):.4f}")
        print(f"  LPIPS : {sum(all_lpips)/len(all_lpips):.4f}")

    except ImportError:
        print("Install metrics: pip install lpips torchmetrics")


# ============================================================
# GENERATE
# ============================================================

def run_generate(args):
    print("=" * 55)
    print("MODE: Generate thermal from RGB image")
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print("=" * 55)

    if args.input is None:
        print("ERROR: provide --input path/to/image.jpg")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: file not found: {args.input}")
        sys.exit(1)

    import yaml
    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])
    sys.path.insert(0, "src")

    from model import build_model, WeatherProjection
    model, model_type = build_model(cfg)
    model             = model.to(device)
    weather_proj      = WeatherProjection(
        weather_dim=cfg.get("weather_dim", 9),
        model_dim=384
    ).to(device)

    # load checkpoint
    if args.checkpoint:
        print(f"Loading: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        weather_proj.load_state_dict(ckpt["weather_proj"])
        print(f"Epoch {ckpt['epoch']} | "
              f"PSNR {ckpt.get('psnr',0):.2f} dB | "
              f"LPIPS {ckpt.get('lpips',0):.4f}")
    else:
        print("No checkpoint — using pretrained HF weights")

    # preprocess input
    H = cfg["image_size"]
    transform = transforms.Compose([
        transforms.Resize((H, H)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                             std= [0.5,0.5,0.5])
    ])
    rgb_tensor = transform(
        Image.open(args.input).convert("RGB")
    ).unsqueeze(0).to(device)

    # zero weather (no metadata for arbitrary input)
    weather = torch.zeros(
        1, cfg.get("weather_dim", 9)
    ).to(device)

    # generate — 50 denoising steps
    model.eval()
    print("Generating thermal image (50 steps)...")

    with torch.no_grad():
        z     = torch.randn_like(rgb_tensor)
        steps = 50
        for i in range(steps):
            t_val = 1.0 - (i / steps)
            t_    = torch.tensor([t_val]).to(device)
            style = torch.zeros(1, dtype=torch.long).to(device)
            try:
                velocity = model(z, rgb_tensor, t_, style)
            except TypeError:
                velocity = model(rgb_tensor)
            z = z + velocity * (1.0 / steps)

    # save output
    output = (z.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    os.makedirs(os.path.dirname(args.output)
                if os.path.dirname(args.output) else ".",
                exist_ok=True)
    transforms.ToPILImage()(output).save(args.output)
    print(f"Saved: {args.output}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: config not found: {args.config}")
        print("Expected: configs/config.yml")
        sys.exit(1)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "generate":
        run_generate(args)