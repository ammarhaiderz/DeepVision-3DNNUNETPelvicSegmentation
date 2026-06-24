# Pelvic1K to PENGWIN LH Smoke Test

This experiment trains one model in two stages:

1. Multiclass pretraining on CTPelvic1K:
   - `0`: background
   - `1`: sacrum
   - `2`: right hip
   - `3`: left hip
   - `4`: lumbar vertebra
2. Binary fine-tuning on PENGWIN:
   - `0`: background
   - `1`: left hip
   - RH and sacrum CT voxels are replaced with `-1024 HU` by default.

The final binary model is run on two versions of the same PENGWIN test cases:

- erased input: RH and sacrum are replaced with `-1024 HU`
- normal input: the original CT contains all bones

Both predictions are evaluated against the same binary LH ground truth.

The CTPelvic1K pretraining phase has its own held-out test set. That test set
is evaluated before any PENGWIN fine-tuning.

## Pelvic1K data

The official project is:

https://github.com/MIRACLE-Center/CTPelvic1K

The project provides annotations and clinical data through Zenodo. For public
sub-datasets 1-5, the original images must be obtained from their original
sources. The official annotation filenames are `*_mask_4label.nii.gz`; matching
images are `*_data.nii.gz`.

For this experiment, extract the downloaded dataset 6 and dataset 7 image and
mask archives below one folder:

```text
/data/home/ue704/Pelvic1k/extracted/
├── CTPelvic1K_dataset6_data/
├── ipcai2021_dataset6_Anonymized/
├── CTPelvic1K_dataset7_data/
└── dataset7_masks/
```

Nested folders are accepted. Dataset 6 contributes 103 labeled cases. Dataset
7 contains 75 CT images but only 14 released masks, so only those 14 labeled
cases are included. The remaining 61 images are recorded as ignored in
`preparation_summary.json`.

## Activate the environment

Run all commands from the main repository:

```bash
cd /data/home/ue704/DeepVision-3DNNUNETPelvicSegmentation
source nnunet_env/bin/activate
```

## 1. Prepare Pelvic1K pretraining data

For a quick subset smoke test:

```bash
python lh_pretraining_smoke_test/prepare_pelvic1k_dataset.py \
  --data_root /data/home/ue704/Pelvic1k/extracted \
  --max_cases 20
```

Remove `--max_cases 20` to use all 117 labeled cases. The script validates
image/label geometry and label values, creates source-preserving case IDs, and
writes a deterministic source-stratified train/validation/test split.

The full split uses seed 42:

- training: 95 cases
- validation: 11 cases
- held-out test: 11 cases

Training and validation cases are stored in `imagesTr` and `labelsTr`. Test
cases are physically separated into `imagesTs` and `labelsTs`, so they cannot
be sampled by the trainer.

This creates:

```text
nnUNet/nnUNet_raw/Dataset016_CTPelvic1K_pretrain
```

## 2. Pretrain the multiclass model

The environment uses PyTorch 2.5.1 with its CUDA 12.1 runtime so it remains
compatible with the cluster's CUDA 12.2 driver. The launcher also checks CUDA
before preprocessing and treats `--gpu` as an index into GPUs allocated by
Slurm.

```bash
python lh_pretraining_smoke_test/train_pelvic1k_pretrain.py \
  --gpu 0 \
  --num_gpus 1
```

This plans, preprocesses, and trains Dataset016.

The dedicated pretraining trainer writes:

```text
nnUNet/nnUNet_results/Dataset016_CTPelvic1K_pretrain/
└── nnUNetTrainer_PelvicPretrain__nnUNetPlans__3d_fullres/
    └── fold_0/
        ├── checkpoint_best.pth
        ├── checkpoint_final.pth
        ├── checkpoint_latest.pth
        ├── experiment_manifest.json
        ├── progress.png
        ├── training_log_*.txt
        └── tensorboard/
```

`checkpoint_latest.pth` is updated every 10 epochs and retained after training.
Resume an interrupted run with:

```bash
python lh_pretraining_smoke_test/train_pelvic1k_pretrain.py \
  --gpu 0 \
  --num_gpus 1 \
  --skip_preprocess \
  --continue_training
```

If preprocessing completed but training never created a checkpoint, restart
without `--continue_training`:

```bash
python lh_pretraining_smoke_test/train_pelvic1k_pretrain.py \
  --gpu 0 \
  --num_gpus 1 \
  --skip_preprocess
```

## 3. Track training in VS Code

Start TensorBoard in another cluster terminal:

```bash
python lh_pretraining_smoke_test/view_tensorboard.py --port 6006
```

In VS Code, open the **Ports** panel and forward port `6006`, then open the
forwarded address. TensorBoard shows train/validation loss, mean Dice,
per-class Dice, learning rate, and epoch timing.

## 4. Evaluate the held-out CTPelvic1K test set

After training completes:

```bash
python lh_pretraining_smoke_test/evaluate_pelvic1k_test.py \
  --gpu 0 \
  --checkpoint checkpoint_best.pth
```

Predictions and the nnU-Net metric summary are written below:

```text
lh_pretraining_smoke_test/artifacts/pelvic1k_test/checkpoint_best/
```

## 5. Prepare binary erased-LH PENGWIN data

```bash
python lh_pretraining_smoke_test/prepare_pengwin_lh_dataset.py \
  --background_value -1024
```

This creates:

```text
nnUNet/nnUNet_raw/Dataset017_PENGWIN_erased_LH
lh_pretraining_smoke_test/artifacts/normal_test_images
```

Dataset017 `imagesTs` contains erased test CTs. `normal_test_images` contains
the original versions of exactly the same cases.

## 6. Fine-tune from Pelvic1K

```bash
python lh_pretraining_smoke_test/finetune_lh_from_pelvic1k.py \
  --gpu 0 \
  --num_gpus 1
```

This:

1. creates the PENGWIN train/validation split
2. extracts Dataset017's fingerprint
3. transfers Dataset016's plans to Dataset017
4. preprocesses Dataset017 with the transferred plans
5. initializes matching network weights from Dataset016
6. fine-tunes a new binary segmentation head

## 7. Predict both test inputs

```bash
python lh_pretraining_smoke_test/predict_both_test_inputs.py --gpu 0
```

Predictions are written to:

```text
lh_pretraining_smoke_test/artifacts/predictions/erased_input
lh_pretraining_smoke_test/artifacts/predictions/normal_input
```

## 8. Compare both tests

```bash
python lh_pretraining_smoke_test/evaluate_both_test_inputs.py
```

The result is:

```text
lh_pretraining_smoke_test/artifacts/evaluation_comparison.csv
```

It contains Dice, IoU, HD95, ASSD, false-positive voxels, and false-negative
voxels for both input conditions, plus the Dice difference between normal and
erased inputs.

## Important interpretation

Erasing RH and sacrum uses ground-truth labels, so it is privileged information.
This is suitable for the proposed smoke test, but it is not a deployable
inference pipeline unless another method supplies those erase masks.
