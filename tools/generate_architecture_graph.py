#!/usr/bin/env python3
"""Generate a Graphviz diagram of the full 2‑phase KD system architecture.

This script is based on the current code paths:
- trainer/engine.py (main, train_teacher, train_student)
- data/dataset.py (MedPixDataset, WoundDataset, get_num_classes)
- models/teacher.py, models/student.py
- models/fusion/* (fusion modules)
- losses/* (distillation losses)

It writes a DOT file describing:
- Config → dataset/dataloaders → teacher/student models
- Fusion modules and loss modules
- Phase 1 (teacher training) and Phase 2 (student distillation)

Usage (from repo root):

  python tools/generate_architecture_graph.py \
      --output-dir figures \
      --basename system_architecture \
      --render --format png

This will create:
  figures/system_architecture.dot
  figures/system_architecture.png   (if Graphviz "dot" is installed)
"""

import argparse
import subprocess
import shutil
from pathlib import Path


DOT_TEMPLATE = r"""digraph MedpixKD {
  rankdir=LR;
  fontsize=10;
  labelloc="t";
  label="Two-Stage Multi-Modal KD System Architecture";

  node [shape=box style=rounded fontsize=9];

  // ================= CONFIG =================
  subgraph cluster_config {
    label="Config (YAML → cfg dict)";
    style=filled;
    color=lightgrey;

    C_data   [label="data:\n- type (medpix/wound)\n- root, batch_size, num_workers\n- task1_label, task2_label"];
    C_teacher[label="teacher:\n- vision, text\n- fusion_dim, fusion_layers\n- fusion_heads, dropout"];
    C_student[label="student:\n- vision, text\n- fusion_dim, fusion_layers\n- fusion_heads, dropout"];
    C_fusion [label="fusion:\n- type\n- hidden_mult, p_img, p_txt,\n  dropout, ... (fusion_params)"];
    C_loss   [label="loss:\n- type ∈ {vanilla, combined,\n  crd, rkd, mmd}\n- alpha, beta, gamma, T, ..."];
    C_device [label="device:\n- cuda:X or cpu"];
  }

  // ================= DATASETS =================
  subgraph cluster_data {
    label="Datasets & Dataloaders (data/dataset.py, trainer/engine.py)";
    style=filled;
    color=lightyellow;

    D_type   [label="dataset_type\n(medpix / wound)" shape=ellipse];
    D_mp     [label="MedPixDataset\n(data_*.jsonl, descriptions_*.jsonl, images/)"];
    D_wd     [label="WoundDataset\n(metadata_*.csv, images/)"];
    D_nc     [label="get_num_classes()\n(modality, location)"];
    ImgProc  [label="ViTImageProcessor\n(vit-base, resize+norm)"];
    Toks     [label="Teacher & Student Tokenizers\n(AutoTokenizer from backbones)"];
    DL_train [label="DataLoader(train)"];
    DL_dev   [label="DataLoader(dev)"];
    DL_test  [label="DataLoader(test)"];
  }

  // ================= TEACHER =================
  subgraph cluster_teacher {
    label="Teacher Model (models/teacher.py)";
    style=filled;
    color=lightblue;

    T_vis   [label="Vision Backbone\n(get_vision_backbone)"];
    T_txt   [label="Text Backbone\n(get_text_backbone)"];
    T_proj  [label="Projection Layers\nproj_vis, proj_txt → fusion_dim"];
    T_fuse  [label="Fusion Module\n(Simple, ConcatMLP, CrossAttention, Gated,\n TransformerConcat, ModalityDropout, FiLM,\n EnergyAwareAdaptive, SHoMR)"];
    T_heads [label="Classification Heads\n(modality, location)"];
    T_out   [label="Teacher outputs dict:\n- logits_modality\n- logits_location\n- img_raw/txt_raw\n- img_proj/txt_proj" shape=note];

    T_vis -> T_proj;
    T_txt -> T_proj;
    T_proj -> T_fuse -> T_heads -> T_out;
  }

  // ================= STUDENT =================
  subgraph cluster_student {
    label="Student Model (models/student.py)";
    style=filled;
    color=lightcyan;

    S_vis   [label="Vision Backbone\n(smaller)"];
    S_txt   [label="Text Backbone\n(smaller)"];
    S_proj  [label="Projection Layers\nproj_vis, proj_txt → fusion_dim"];
    S_fuse  [label="Fusion Module\n(same interface, own params)"];
    S_heads [label="Classification Heads\n(modality, location)"];
    S_out   [label="Student outputs dict:\n- logits_modality\n- logits_location\n- img_raw/txt_raw\n- img_proj/txt_proj" shape=note];

    S_vis -> S_proj;
    S_txt -> S_proj;
    S_proj -> S_fuse -> S_heads -> S_out;
  }

  // ================= LOSSES =================
  subgraph cluster_losses {
    label="KD / Loss Modules (losses/*)";
    style=filled;
    color=mistyrose;

    L_ce   [label="Supervised CE\n(teacher or student logits)"];
    L_van  [label="DistillationLoss\n(vanilla: CE + KL + MSE)"];
    L_comb [label="MedKDCombinedLoss\n(CE + KL + MSE + CRD)"];
    L_crd  [label="CRDLoss"];
    L_rkd  [label="RKDLoss"];
    L_mmd  [label="MMDLoss"];
  }

  // ================= TRAINER =================
  subgraph cluster_trainer {
    label="Trainer (trainer/engine.py)";
    style=filled;
    color=lightpink;

    Tr_main   [label="main(cfg)\n- parse config\n- build datasets/models\n- select loss\n- run phases"];
    Tr_Ttrain [label="Phase 1: train_teacher()\n(supervised teacher training)"];
    Tr_Strain [label="Phase 2: train_student()\n(student KD + supervised)"];
    Tr_eval   [label="evaluate_detailed()\n(dev/test metrics)"];
    Tr_log    [label="MetricsLogger / ResultsLogger"];
  }

  // ===== CONFIG → CONSTRUCTION =====
  C_data   -> D_type;
  C_data   -> DL_train;
  C_data   -> DL_dev;
  C_data   -> DL_test;
  C_teacher-> T_vis;
  C_teacher-> T_txt;
  C_teacher-> T_proj;
  C_teacher-> T_fuse;
  C_student-> S_vis;
  C_student-> S_txt;
  C_student-> S_proj;
  C_student-> S_fuse;
  C_fusion -> T_fuse;
  C_fusion -> S_fuse;
  C_loss   -> L_van;
  C_loss   -> L_comb;
  C_loss   -> L_crd;
  C_loss   -> L_rkd;
  C_loss   -> L_mmd;
  C_device -> Tr_main;

  // ===== DATA FLOW =====
  D_type -> D_mp [label="if medpix"];
  D_type -> D_wd [label="if wound"];
  D_mp   -> D_nc;
  D_wd   -> D_nc;

  D_mp   -> ImgProc;
  D_wd   -> ImgProc;
  ImgProc -> DL_train;
  ImgProc -> DL_dev;
  ImgProc -> DL_test;

  Toks   -> DL_train;
  Toks   -> DL_dev;
  Toks   -> DL_test;

  // ===== PHASE 1: TEACHER TRAINING =====
  DL_train -> Tr_Ttrain;
  Tr_Ttrain -> T_out [label="batches:\n(pv, ids_teacher, mask_teacher, labels)"];
  T_out -> L_ce;
  L_ce -> Tr_Ttrain;

  // ===== PHASE 2: STUDENT DISTILLATION =====
  DL_train -> Tr_Strain;
  Tr_Strain -> T_out [label="teacher(pv, ids_t, mask_t)\n(no grad)"];
  Tr_Strain -> S_out [label="student(pv, ids_s, mask_s)"];

  T_out -> L_van;
  S_out -> L_van;
  T_out -> L_comb;
  S_out -> L_comb;
  T_out -> L_crd;
  S_out -> L_crd;
  T_out -> L_rkd;
  S_out -> L_rkd;
  T_out -> L_mmd;
  S_out -> L_mmd;

  {L_van L_comb L_crd L_rkd L_mmd} -> Tr_Strain [label="KD loss backprop"];

  // ===== EVAL & LOGGING =====
  DL_dev  -> Tr_eval;
  DL_test -> Tr_eval;
  Tr_Ttrain -> Tr_eval;
  Tr_Strain -> Tr_eval;
  Tr_eval -> Tr_log;
}
"""


def generate_dot(output_path: Path) -> None:
  """Write the DOT graph to the given path."""
  output_path.write_text(DOT_TEMPLATE)
  print(f"[generate_architecture_graph] Wrote DOT file to: {output_path}")


def render_with_dot(dot_path: Path, out_path: Path, fmt: str) -> None:
  """Render a DOT file to an image using the Graphviz 'dot' CLI, if available."""
  dot_exe = shutil.which("dot")
  if dot_exe is None:
    print("[generate_architecture_graph] 'dot' executable not found. Skipping rendering.")
    print("Install Graphviz (e.g., 'sudo apt-get install graphviz') and re-run with --render.")
    return

  cmd = [dot_exe, f"-T{fmt}", str(dot_path), "-o", str(out_path)]
  print(f"[generate_architecture_graph] Running: {' '.join(cmd)}")
  try:
    subprocess.run(cmd, check=True)
  except subprocess.CalledProcessError as e:
    print(f"[generate_architecture_graph] dot command failed: {e}")
  else:
    print(f"[generate_architecture_graph] Rendered {fmt} to: {out_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate Graphviz diagram of the 2-phase KD system architecture.")
  parser.add_argument("--output-dir", type=Path, default=Path("figures"),
                      help="Directory to place the generated files (default: figures)")
  parser.add_argument("--basename", type=str, default="system_architecture",
                      help="Base name for outputs (default: system_architecture)")
  parser.add_argument("--render", action="store_true",
                      help="If set, also render to an image using Graphviz 'dot'.")
  parser.add_argument("--format", type=str, default="png",
                      help="Image format for rendering (default: png; e.g., pdf, svg)")

  args = parser.parse_args()
  out_dir: Path = args.output_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  dot_path = out_dir / f"{args.basename}.dot"
  generate_dot(dot_path)

  if args.render:
    img_path = out_dir / f"{args.basename}.{args.format}"
    render_with_dot(dot_path, img_path, args.format)


if __name__ == "__main__":
  main()
