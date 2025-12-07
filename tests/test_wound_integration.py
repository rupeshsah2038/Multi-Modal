#!/usr/bin/env python3
"""
Test script to verify Wound dataset integration.
Creates a minimal mock wound dataset and runs a quick training loop.
"""
import os
import sys
import pandas as pd
import yaml
from PIL import Image
import numpy as np

# Add repo root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def create_mock_wound_dataset(output_dir='test_wound_data'):
    """Create a minimal mock wound dataset for testing"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Create mock images (simple colored squares)
    for i in range(30):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(os.path.join(output_dir, 'images', f'wound_{i:03d}.jpg'))
    
    # Create mock metadata
    types = ['burn', 'laceration', 'ulcer'] * 10
    severities = ['mild', 'moderate', 'severe'] * 10
    descriptions = [f'Sample wound description {i}' for i in range(30)]
    filepaths = [f'wound_{i:03d}.jpg' for i in range(30)]
    
    df = pd.DataFrame({
        'file_path': filepaths,
        'type': types,
        'severity': severities,
        'description': descriptions
    })
    
    # Save as single metadata file
    df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    # Split into train/dev/test
    train_df = df[:20]
    dev_df = df[20:25]
    test_df = df[25:]
    
    train_df.to_csv(os.path.join(output_dir, 'metadata_train.csv'), index=False)
    dev_df.to_csv(os.path.join(output_dir, 'metadata_dev.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'metadata_test.csv'), index=False)
    
    print(f"✓ Created mock wound dataset in {output_dir}/")
    print(f"  - {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test samples")
    print(f"  - Types: {list(df['type'].unique())}")
    print(f"  - Severities: {list(df['severity'].unique())}")
    
    return output_dir

def test_wound_dataset_loading():
    """Test that WoundDataset can load data correctly"""
    from data.dataset import get_dataset, get_num_classes
    from transformers import AutoTokenizer
    
    print("\n=== Testing Wound Dataset Loading ===")
    
    # Create mock data
    data_dir = create_mock_wound_dataset()
    
    # Load tokenizers
    tokenizer_t = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer_s = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create dataset
    dataset = get_dataset(
        dataset_type='wound',
        csv_file=os.path.join(data_dir, 'metadata_train.csv'),
        image_dir=os.path.join(data_dir, 'images'),
        tokenizer_teacher=tokenizer_t,
        tokenizer_student=tokenizer_s,
    )
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Test get_num_classes
    num_classes = get_num_classes(dataset_type='wound', dataset_root=data_dir)
    print(f"✓ Class counts: {num_classes}")
    
    # Test __getitem__
    sample = dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")
    print(f"  - pixel_values shape: {sample['pixel_values'].shape}")
    print(f"  - modality (type) label: {sample['modality'].item()}")
    print(f"  - location (severity) label: {sample['location'].item()}")
    
    print("\n✅ Wound dataset loading test PASSED")
    return data_dir

def test_wound_training_integration():
    """Test that wound dataset works with training pipeline"""
    print("\n=== Testing Wound Training Integration ===")
    
    # Use the mock dataset
    data_dir = 'test_wound_data'
    
    # Create minimal config
    config = {
        'data': {
            'type': 'wound',
            'root': data_dir,
            'batch_size': 4,
            'num_workers': 0,
        },
        'teacher': {
            'vision': 'vit-base',
            'text': 'distilbert',
            'fusion_layers': 1,
            'fusion_dim': 128,
        },
        'student': {
            'vision': 'deit-base',
            'text': 'distilbert',
            'fusion_layers': 1,
            'fusion_dim': 128,
        },
        'training': {
            'teacher_epochs': 1,
            'student_epochs': 1,
            'teacher_lr': 1e-4,
            'student_lr': 1e-4,
            'alpha': 1.0,
            'beta': 10.0,
            'T': 2.0,
        },
        'logging': {
            'log_dir': 'logs/test_wound',
        },
        'fusion': {
            'type': 'simple',
        },
        'loss': {
            'type': 'vanilla',
        },
        'device': 'cpu',  # Use CPU for testing
    }
    
    # Save config
    os.makedirs('logs', exist_ok=True)
    with open('logs/test_wound_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("✓ Config created")
    
    # Run training (import here to avoid early import issues)
    from trainer.engine import main
    
    try:
        main(config)
        print("\n✅ Wound training integration test PASSED")
    except Exception as e:
        print(f"\n❌ Wound training integration test FAILED: {e}")
        raise

if __name__ == "__main__":
    try:
        # Test dataset loading
        data_dir = test_wound_dataset_loading()
        
        # Test training integration
        test_wound_training_integration()
        
        print("\n" + "="*60)
        print("🎉 All Wound dataset tests PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
