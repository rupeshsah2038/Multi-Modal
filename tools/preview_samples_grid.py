#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import random
import textwrap
from typing import Dict, List, Optional, Tuple

import jsonlines
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def _wrap(text: str, width: int = 45, max_lines: int = 6) -> str:
    text = (text or "").strip()
    if not text:
        return "(no text)"
    wrapped = textwrap.fill(text, width=width)
    lines = wrapped.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + " …"
    return "\n".join(lines)


def _pick_n(items: List, n: int, seed: int) -> List:
    if n <= 0:
        return []
    n = min(n, len(items))
    rng = random.Random(seed)
    return rng.sample(items, n)


def _find_image(path_no_ext: str) -> Optional[str]:
    candidates = [
        path_no_ext,
        path_no_ext + ".png",
        path_no_ext + ".jpg",
        path_no_ext + ".jpeg",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_medpix_items(root: str, split: str) -> List[Dict]:
    split = split.lower()
    desc_path = os.path.join(root, f"splitted_dataset/descriptions_{split}.jsonl")
    data_path = os.path.join(root, f"splitted_dataset/data_{split}.jsonl")
    image_dir = os.path.join(root, "images")

    if not os.path.exists(desc_path):
        raise FileNotFoundError(f"Missing file: {desc_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing file: {data_path}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing directory: {image_dir}")

    case_by_uid: Dict[str, Dict] = {}
    with jsonlines.open(data_path, "r") as f:
        for obj in f:
            uid = obj.get("U_id")
            if uid:
                case_by_uid[str(uid)] = obj

    items: List[Dict] = []
    with jsonlines.open(desc_path, "r") as f:
        for obj in f:
            uid = str(obj.get("U_id", ""))
            image_key = obj.get("image")
            if not image_key:
                continue

            img_path = _find_image(os.path.join(image_dir, str(image_key)))
            caption = (
                (obj.get("Description") or {}).get("Caption")
                if isinstance(obj.get("Description"), dict)
                else None
            )

            case = case_by_uid.get(uid) or {}
            history = (((case.get("Case") or {}).get("History")) if isinstance(case.get("Case"), dict) else None)

            items.append(
                {
                    "img_path": img_path,
                    "caption": caption or "",
                    "history": history or "",
                    "uid": uid,
                }
            )

    return items


def load_wound_items(root: str, split: str) -> List[Dict]:
    split = split.lower()
    csv_path = os.path.join(root, f"metadata_{split}.csv")
    image_dir = os.path.join(root, "images")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing directory: {image_dir}")

    df = pd.read_csv(csv_path)

    # Prefer WoundDataset defaults; also accept common alternatives
    filepath_col = "img_path" if "img_path" in df.columns else ("file_path" if "file_path" in df.columns else None)
    if filepath_col is None:
        raise KeyError(f"CSV must contain 'img_path' or 'file_path'. Found columns: {list(df.columns)}")

    desc_col = "description" if "description" in df.columns else None
    caption_col = "caption" if "caption" in df.columns else None

    items: List[Dict] = []
    for _, row in df.iterrows():
        rel = str(row[filepath_col])
        img_path = _find_image(os.path.join(image_dir, rel)) or _find_image(os.path.join(root, rel))

        description = str(row[desc_col]).strip() if desc_col and pd.notna(row.get(desc_col)) else ""
        caption = str(row[caption_col]).strip() if caption_col and pd.notna(row.get(caption_col)) else ""

        items.append(
            {
                "img_path": img_path,
                "description": description,
                "caption": caption,
                "type": str(row.get("type", "")) if pd.notna(row.get("type", "")) else "",
                "severity": str(row.get("severity", "")) if pd.notna(row.get("severity", "")) else "",
            }
        )

    return items


def render_grid(
    items: List[Dict],
    dataset: str,
    split: str,
    seed: int,
    out: Optional[str],
    include_history: bool,
) -> None:
    if not items:
        raise ValueError("No items found to display")

    picked = _pick_n(items, n=9, seed=seed)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(f"Samples: {dataset} ({split})", fontsize=14)

    for ax, item in zip(axes.flatten(), picked):
        img_path = item.get("img_path")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "(missing image)", ha="center", va="center")

        ax.axis("off")

        if dataset == "medpix":
            parts: List[str] = []
            cap = (item.get("caption") or "").strip()
            hist = (item.get("history") or "").strip()
            if cap:
                parts.append(f"Caption: {cap}")
            if include_history and hist:
                parts.append(f"History: {hist}")
            text = "\n\n".join(parts) if parts else "(no text)"
        else:
            parts = []
            t = (item.get("type") or "").strip()
            s = (item.get("severity") or "").strip()
            desc = (item.get("description") or "").strip()
            cap = (item.get("caption") or "").strip()
            if t:
                parts.append(f"Type: {t}")
            if s:
                parts.append(f"Severity: {s}")
            if desc:
                parts.append(f"Description: {desc}")
            if cap:
                parts.append(f"Caption: {cap}")
            text = "\n".join(parts) if parts else "(no text)"

        ax.text(
            0.5,
            -0.05,
            _wrap(text, width=45, max_lines=7),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    # Hide any unused axes (if fewer than 9)
    for ax in axes.flatten()[len(picked) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35)

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved grid to: {out}")
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Display sample images + text in a 3x3 grid")
    p.add_argument("--dataset", choices=["medpix", "wound"], required=True)
    p.add_argument("--root", required=True, help="Dataset root folder")
    p.add_argument("--split", default="train", choices=["train", "dev", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="Optional output image path (e.g. plots/samples.png)")
    p.add_argument("--include-history", action="store_true", help="(MedPix) include history text")
    args = p.parse_args()

    if args.dataset == "medpix":
        items = load_medpix_items(args.root, args.split)
    else:
        items = load_wound_items(args.root, args.split)

    render_grid(
        items,
        dataset=args.dataset,
        split=args.split,
        seed=args.seed,
        out=args.out,
        include_history=bool(args.include_history),
    )


if __name__ == "__main__":
    main()
