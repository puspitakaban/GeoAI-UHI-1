# main.py

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="ThermalGen — RGB to Thermal Image Translation"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "generate"],
        help="What to do: train | eval | generate"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yml",
        help="Path to config file (default: configs/config.yml)"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="(generate mode only) Path to input RGB image"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generated_thermal.png",
        help="(generate mode only) Where to save the output"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="(eval/generate mode) Path to a specific checkpoint .pt file"
    )

    return parser.parse_args()


def run_train(args):
    print("=" * 50)
    print("MODE: Training on Boson Night (STGL dataset)")
    print(f"Config: {args.config}")
    print("=" * 50)
    from src.train import train
    train(args.config)


def run_eval(args):
    print("=" * 50)
    print("MODE: Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 50)
    print("\nEvaluation mode coming soon.")
    print("Validation loss is already printed during training.")


def run_generate(args):
    print("=" * 50)
    print("MODE: Generate thermal image from RGB")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("=" * 50)

    if args.input is None:
        print("ERROR: Please provide an input image with --input")
        print("Example: python main.py --mode generate --input photo.png")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    import yaml
    import torch
    from PIL import Image
    import torchvision.transforms as transforms

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])

    from src.train import load_model, AlphaEarthProjection
    model   = load_model(cfg).to(device)
    ae_proj = AlphaEarthProjection(
        ae_dim=cfg["alphaearth_dim"],
        model_dim=384
    ).to(device)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        ae_proj.load_state_dict(ckpt["ae_proj"])
        print(f"Restored from epoch {ckpt['epoch']}")
    else:
        print("No checkpoint — using pretrained HF weights.")

    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std= [0.5, 0.5, 0.5])
    ])

    rgb_tensor = transform(
        Image.open(args.input).convert("RGB")
    ).unsqueeze(0).to(device)

    ae_embed = torch.zeros(1, cfg["alphaearth_dim"]).to(device)

    model.eval()
    print("Generating thermal image...")

    with torch.no_grad():
        z     = torch.randn_like(rgb_tensor)
        steps = 50
        for i in range(steps):
            t_val    = 1.0 - (i / steps)
            t        = torch.tensor([t_val]).to(device)
            style    = torch.zeros(1, dtype=torch.long).to(device)
            try:
                velocity = model(z, rgb_tensor, t, style)
            except TypeError:
                velocity = model(rgb_tensor)
            z = z + velocity * (1.0 / steps)

    output = (z.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    transforms.ToPILImage()(output).save(args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "generate":
        run_generate(args)
