#!/usr/bin/env python3
"""
Quick verification script to check Wound-1-0 dataset setup.
Run this before training to ensure your dataset is properly configured.

Usage:
    python tools/verify_wound_dataset.py Wound-1-0
"""
import argparse
import os
import sys
import pandas as pd

def verify_wound_dataset(root_dir):
    """Verify wound dataset structure and contents"""
    print(f"Verifying Wound dataset in: {root_dir}\n")
    
    errors = []
    warnings = []
    
    # Check directory exists
    if not os.path.exists(root_dir):
        errors.append(f"Directory not found: {root_dir}")
        return errors, warnings
    
    # Check images directory
    images_dir = os.path.join(root_dir, 'images')
    if not os.path.exists(images_dir):
        errors.append(f"Images directory not found: {images_dir}")
    else:
        num_images = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✓ Images directory found: {num_images} images")
    
    # Check for split CSV files
    required_splits = ['train', 'dev', 'test']
    split_files = {}
    
    for split in required_splits:
        csv_path = os.path.join(root_dir, f'metadata_{split}.csv')
        if not os.path.exists(csv_path):
            errors.append(f"Missing split file: {csv_path}")
        else:
            split_files[split] = csv_path
    
    if errors:
        print("❌ ERRORS FOUND:")
        for err in errors:
            print(f"  - {err}")
        print("\nRun the dataset splitter first:")
        print(f"  python tools/split_wound_dataset.py --input {root_dir}/metadata.csv --output {root_dir}")
        return errors, warnings
    
    # Analyze CSV files
    print("\nAnalyzing split files:")
    all_types = set()
    all_severities = set()
    
    for split, csv_path in split_files.items():
        df = pd.read_csv(csv_path)
        print(f"\n  {split.upper()}:")
        print(f"    Samples: {len(df)}")
        
        # Check required columns
        required_cols = ['file_path', 'type', 'severity', 'description']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"{split} CSV missing columns: {missing_cols}")
            continue
        
        # Check for missing values
        for col in required_cols:
            null_count = df[col].isna().sum()
            if null_count > 0:
                warnings.append(f"{split} has {null_count} missing values in '{col}' column")
        
        # Collect unique values
        types_in_split = set(df['type'].dropna().unique())
        severities_in_split = set(df['severity'].dropna().unique())
        all_types.update(types_in_split)
        all_severities.update(severities_in_split)
        
        print(f"    Types: {sorted(types_in_split)}")
        print(f"    Severities: {sorted(severities_in_split)}")
        
        # Check if images exist
        if os.path.exists(images_dir):
            missing_images = []
            for idx, row in df.head(5).iterrows():  # Check first 5
                img_path = os.path.join(images_dir, row['file_path'])
                if not os.path.exists(img_path):
                    missing_images.append(row['file_path'])
            
            if missing_images:
                warnings.append(f"{split} has missing images (checked first 5): {missing_images}")
    
    # Summary
    print(f"\n{'='*60}")
    print("DATASET SUMMARY:")
    print(f"  Total type classes: {len(all_types)} - {sorted(all_types)}")
    print(f"  Total severity classes: {len(all_severities)} - {sorted(all_severities)}")
    
    # Check for class consistency across splits
    for split, csv_path in split_files.items():
        df = pd.read_csv(csv_path)
        split_types = set(df['type'].dropna().unique())
        split_severities = set(df['severity'].dropna().unique())
        
        missing_types = all_types - split_types
        missing_severities = all_severities - split_severities
        
        if missing_types:
            warnings.append(f"{split} missing type classes: {missing_types}")
        if missing_severities:
            warnings.append(f"{split} missing severity classes: {missing_severities}")
    
    # Print warnings
    if warnings:
        print(f"\n⚠️  WARNINGS:")
        for warn in warnings:
            print(f"  - {warn}")
    
    # Final verdict
    print(f"\n{'='*60}")
    if errors:
        print("❌ VERIFICATION FAILED")
        print("Please fix the errors above before training.")
        return errors, warnings
    elif warnings:
        print("⚠️  VERIFICATION PASSED WITH WARNINGS")
        print("You can proceed with training, but review warnings above.")
    else:
        print("✅ VERIFICATION PASSED")
        print("Dataset is ready for training!")
    
    # Suggest config
    print(f"\nSuggested config/wound.yaml settings:")
    print(f"  data:")
    print(f"    type: 'wound'")
    print(f"    root: '{root_dir}'")
    print(f"    # Will detect {len(all_types)} type classes and {len(all_severities)} severity classes")
    
    return errors, warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Wound-1-0 dataset setup")
    parser.add_argument('dataset_dir', type=str, help='Path to Wound-1-0 directory')
    
    args = parser.parse_args()
    
    errors, warnings = verify_wound_dataset(args.dataset_dir)
    
    if errors:
        sys.exit(1)
    else:
        sys.exit(0)
