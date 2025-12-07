#!/usr/bin/env python3
"""
Script to split Wound-1-0 metadata.csv into train/dev/test splits.

Usage:
    python tools/split_wound_dataset.py --input Wound-1-0/metadata.csv \
                                        --output Wound-1-0 \
                                        --train 0.7 --dev 0.15 --test 0.15
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_wound_dataset(input_csv, output_dir, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split wound dataset metadata into train/dev/test sets.
    
    Args:
        input_csv: Path to input metadata.csv
        output_dir: Directory to save split CSV files
        train_ratio: Proportion for training set
        dev_ratio: Proportion for dev set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    # Validate ratios
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")
    
    # First split: train vs (dev+test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(dev_ratio + test_ratio),
        random_state=seed,
        stratify=df['type'] if 'type' in df.columns else None  # Stratify by type if available
    )
    
    # Second split: dev vs test
    dev_ratio_adjusted = dev_ratio / (dev_ratio + test_ratio)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - dev_ratio_adjusted),
        random_state=seed,
        stratify=temp_df['type'] if 'type' in temp_df.columns else None
    )
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'metadata_train.csv')
    dev_path = os.path.join(output_dir, 'metadata_dev.csv')
    test_path = os.path.join(output_dir, 'metadata_test.csv')
    
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSplit complete:")
    print(f"  Train: {len(train_df)} samples -> {train_path}")
    print(f"  Dev:   {len(dev_df)} samples -> {dev_path}")
    print(f"  Test:  {len(test_df)} samples -> {test_path}")
    
    # Print class distribution
    if 'type' in df.columns:
        print(f"\nType distribution:")
        print(f"  Train: {dict(train_df['type'].value_counts())}")
        print(f"  Dev:   {dict(dev_df['type'].value_counts())}")
        print(f"  Test:  {dict(test_df['type'].value_counts())}")
    
    if 'severity' in df.columns:
        print(f"\nSeverity distribution:")
        print(f"  Train: {dict(train_df['severity'].value_counts())}")
        print(f"  Dev:   {dict(dev_df['severity'].value_counts())}")
        print(f"  Test:  {dict(test_df['severity'].value_counts())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Wound-1-0 dataset into train/dev/test")
    parser.add_argument('--input', type=str, required=True, help='Path to input metadata.csv')
    parser.add_argument('--output', type=str, required=True, help='Output directory for split files')
    parser.add_argument('--train', type=float, default=0.7, help='Train ratio (default: 0.7)')
    parser.add_argument('--dev', type=float, default=0.15, help='Dev ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15, help='Test ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_wound_dataset(
        input_csv=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        dev_ratio=args.dev,
        test_ratio=args.test,
        seed=args.seed
    )
