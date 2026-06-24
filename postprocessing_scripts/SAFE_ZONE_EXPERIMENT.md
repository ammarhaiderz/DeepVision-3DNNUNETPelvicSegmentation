# Complete Safe-Zone Experiment

Run the complete experiment from the repository root:

```bash
./nnunet_env/bin/python \
  postprocessing_scripts/run_complete_safe_zone_experiment.py \
  --distance_mm 5 10 15 20 30 40
```

This uses existing LH, RH, and sacrum predictions. It does not run neural
network inference.

For a faster metrics-and-masks run without HTML generation:

```bash
./nnunet_env/bin/python \
  postprocessing_scripts/run_complete_safe_zone_experiment.py \
  --distance_mm 5 10 15 20 30 40 \
  --skip_visualizations
```

Generate all outputs again, replacing existing files:

```bash
./nnunet_env/bin/python \
  postprocessing_scripts/run_complete_safe_zone_experiment.py \
  --distance_mm 5 10 15 20 30 40 \
  --overwrite
```

Outputs are grouped by physical safe-zone distance:

```text
safe_zone_experiments/all_distances/
├── all_distances_summary.csv
├── experiment_manifest.json
├── 5mm/
│   ├── LH/
│   ├── RH/
│   ├── S/
│   └── merged/
├── 10mm/
├── 15mm/
├── 20mm/
├── 30mm/
└── 40mm/
```

Each binary bone folder contains:

```text
original/
filtered/
largest/
safe_zone/
removed/
evaluation/metrics_comparison.csv
visualizations/
```

Each merged folder contains the same masks, a multiclass comparison CSV with
per-class and mean metrics, and merged interactive visualizations.

The comparison CSVs report Dice, IoU, HD95, ASSD, false-positive voxels,
false-negative voxels, prediction volume, and before/after changes.
