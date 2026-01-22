#!/usr/bin/env python
"""
Aggregate teacher metrics and McNemar test results from logs/ultra-edge-hp-tuned-all/*.

Outputs:
- Teacher metrics tables (averaged per dataset)
- McNemar test tables (per model, per dataset)
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_results(logs_dir: str) -> List[Dict]:
    """Load all results.json files from subdirectories."""
    results = []
    logs_path = Path(logs_dir)
    
    for run_dir in sorted(logs_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                data['_run_name'] = run_dir.name
                results.append(data)
        except Exception as e:
            print(f"Warning: failed to load {results_file}: {e}")
    
    return results


def extract_teacher_metrics(results: List[Dict]) -> pd.DataFrame:
    """Extract teacher test metrics from results."""
    rows = []
    
    for res in results:
        run_name = res.get('_run_name', 'unknown')
        dataset = res.get('config', {}).get('data', {}).get('type', 'unknown')
        
        teacher_test = res.get('metrics', {}).get('teacher', {}).get('test', {})
        if not teacher_test:
            continue
        
        row = {
            'run_name': run_name,
            'dataset': dataset,
        }
        
        # Extract all teacher_test_* metrics
        for key, val in teacher_test.items():
            if key.startswith('teacher_test_'):
                row[key] = val
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_mcnemar_metrics(results: List[Dict]) -> pd.DataFrame:
    """Extract McNemar test results from results."""
    rows = []
    
    for res in results:
        run_name = res.get('_run_name', 'unknown')
        dataset = res.get('config', {}).get('data', {}).get('type', 'unknown')
        
        test_metrics = res.get('metrics', {}).get('test', {})
        if not test_metrics:
            continue
        
        # Extract McNemar fields
        mcnemar_data = {k: v for k, v in test_metrics.items() if 'mcnemar' in k}
        
        if not mcnemar_data:
            continue
        
        row = {
            'model': run_name,
            'dataset': dataset,
        }
        row.update(mcnemar_data)
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_teacher_summary(df: pd.DataFrame, output_dir: str):
    """Generate averaged teacher metrics tables per dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group by dataset
    for dataset in df['dataset'].unique():
        df_dataset = df[df['dataset'] == dataset]
        
        # Compute mean for all numeric columns (excluding run_name and dataset)
        numeric_cols = df_dataset.select_dtypes(include='number').columns
        avg_metrics = df_dataset[numeric_cols].mean()
        
        # Create summary dataframe
        summary = pd.DataFrame({
            'metric': avg_metrics.index,
            'average_value': avg_metrics.values,
        })
        
        # Save CSV
        csv_path = output_path / f"teacher_metrics_{dataset}_avg.csv"
        summary.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Saved teacher metrics for {dataset}: {csv_path}")
        
        # Save markdown table (manual formatting to avoid tabulate dependency)
        md_path = output_path / f"teacher_metrics_{dataset}_avg.md"
        with open(md_path, 'w') as f:
            f.write(f"# Teacher Metrics — {dataset.upper()} (Average across all models)\n\n")
            f.write("| Metric | Average Value |\n")
            f.write("|--------|---------------|\n")
            for idx, row in summary.iterrows():
                f.write(f"| {row['metric']} | {row['average_value']:.4f} |\n")
            f.write(f"\n\nNumber of models: {len(df_dataset)}\n")
        print(f"Saved teacher metrics markdown for {dataset}: {md_path}")


def generate_mcnemar_tables(df: pd.DataFrame, output_dir: str):
    """Generate McNemar test tables per dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group by dataset
    for dataset in df['dataset'].unique():
        df_dataset = df[df['dataset'] == dataset].copy()
        
        if df_dataset.empty:
            print(f"No McNemar data for {dataset}")
            continue
        
        # Save CSV
        csv_path = output_path / f"mcnemar_{dataset}.csv"
        df_dataset.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"Saved McNemar results for {dataset}: {csv_path}")
        
        # Save markdown table (manual formatting to avoid tabulate dependency)
        md_path = output_path / f"mcnemar_{dataset}.md"
        with open(md_path, 'w') as f:
            f.write(f"# McNemar Test Results — {dataset.upper()}\n\n")
            # Create header
            cols = df_dataset.columns.tolist()
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
            # Write rows
            for _, row in df_dataset.iterrows():
                vals = []
                for col in cols:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        vals.append(f"{val:.6f}" if isinstance(val, float) else str(val))
                    else:
                        vals.append(str(val))
                f.write("| " + " | ".join(vals) + " |\n")
            f.write(f"\n\nNumber of models: {len(df_dataset)}\n")
        print(f"Saved McNemar markdown for {dataset}: {md_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Aggregate teacher metrics and McNemar results from ultra-edge-hp-tuned-all"
    )
    parser.add_argument(
        '--logs-dir',
        default='logs/ultra-edge-hp-tuned-all',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output-dir',
        default='docs/aggregated_results',
        help='Output directory for generated tables'
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.logs_dir}")
    results = load_results(args.logs_dir)
    print(f"Found {len(results)} runs")
    
    if not results:
        print("No results found. Exiting.")
        return
    
    # Extract teacher metrics
    print("\nExtracting teacher metrics...")
    teacher_df = extract_teacher_metrics(results)
    print(f"Extracted teacher metrics from {len(teacher_df)} runs")
    
    # Extract McNemar metrics
    print("\nExtracting McNemar test results...")
    mcnemar_df = extract_mcnemar_metrics(results)
    print(f"Extracted McNemar data from {len(mcnemar_df)} runs")
    
    # Generate teacher summary tables
    if not teacher_df.empty:
        print("\nGenerating teacher metrics tables...")
        generate_teacher_summary(teacher_df, args.output_dir)
    
    # Generate McNemar tables
    if not mcnemar_df.empty:
        print("\nGenerating McNemar tables...")
        generate_mcnemar_tables(mcnemar_df, args.output_dir)
    
    print(f"\n✓ All tables saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
