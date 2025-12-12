# Model Size Summary

This summary aggregates parameter sizes across all experiments analyzed by `tools/model_size_report.py` and the consolidated CSV at `logs/model_size_report.csv`.

## Aggregates
- Teacher sizes:
  - Range: 196.14M – 417.65M
  - Mean: 371.19M
- Student sizes:
  - Range: 29.21M – 198.65M
  - Mean: 159.17M
- Compression ratios:
  - Range: 1.89x – 6.71x
  - Mean: 2.55x
- Reduction percentages:
  - Range: 47.1% – 85.1%
  - Mean: 57.7%

## Sources
- Per-experiment report: `logs/model_size_report.csv` (includes an `AGGREGATE_MEAN` footer row)
- Stats CSV: `logs/model_size_report_stats.csv` (min/max/mean for teacher, student, compression, reduction)

## Regenerate
Run the reporter to refresh all outputs:

```bash
conda activate fedenv
python tools/model_size_report.py
```

This will update both CSVs and print the same aggregates in the terminal.
