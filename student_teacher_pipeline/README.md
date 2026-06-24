# Student-Teacher Erased-Bone nnU-Net Pipeline

This folder contains an experimental pipeline for privileged-information
student-teacher training.

The idea:

1. Train three binary teacher models.
   - each teacher sees a CT where the other anatomy bones are erased
   - each teacher predicts one target bone only
2. Use the teachers to generate probability priors for the training cases.
3. Train one normal anatomy student model on the original CT and original
   multi-class labels.
4. At inference, use only the student model.

Implementation detail: the student dataset stores teacher priors as channels
1-3 so they are cropped and augmented exactly like the CT. The custom trainer
zeros those channels before forwarding through the network, then uses them only
for the auxiliary soft distillation loss. Test images get zero prior channels,
so inference does not run teacher models.

The student distillation trainer is provided separately from the stock pipeline
so the existing anatomy and binary-bone experiments remain unchanged.

## Dataset IDs

The scripts use these dataset IDs by default:

- `Dataset012_teacher_erased_LH`
- `Dataset013_teacher_erased_RH`
- `Dataset014_teacher_erased_S`
- `Dataset015_student_anatomy_distilled`

## Typical Order

Run from the `NNUNETPipeline` project root.

```bash
python student_teacher_pipeline/run_student_teacher_pipeline.py --gpu 0 --teacher_num_gpus 1 --student_num_gpus 4 --parallel_teacher_train
```

This trains the three teachers concurrently on GPUs `0`, `1`, and `2`, then
trains the student with four GPUs starting at GPU `0`.

For step-by-step execution:

```bash
python student_teacher_pipeline/scripts/generate_erased_bone_teacher_datasets.py
python train_val_test_nnunet_scripts/create_train_val_split.py --dataset_id 12
python train_val_test_nnunet_scripts/create_train_val_split.py --dataset_id 13
python train_val_test_nnunet_scripts/create_train_val_split.py --dataset_id 14
python train_val_test_nnunet_scripts/preprocess.py --dataset_id 12
python train_val_test_nnunet_scripts/preprocess.py --dataset_id 13
python train_val_test_nnunet_scripts/preprocess.py --dataset_id 14
python train_val_test_nnunet_scripts/train.py --dataset_id 12 --trainer nnUNetTrainer_no_rotation --fold 0 --gpu 0 --num_gpus 4
python train_val_test_nnunet_scripts/train.py --dataset_id 13 --trainer nnUNetTrainer_no_rotation --fold 0 --gpu 0 --num_gpus 4
python train_val_test_nnunet_scripts/train.py --dataset_id 14 --trainer nnUNetTrainer_no_rotation --fold 0 --gpu 0 --num_gpus 4
python student_teacher_pipeline/scripts/predict_teacher_priors.py --dataset_id 12 --split train --save_probabilities --gpu 0
python student_teacher_pipeline/scripts/predict_teacher_priors.py --dataset_id 13 --split train --save_probabilities --gpu 0
python student_teacher_pipeline/scripts/predict_teacher_priors.py --dataset_id 14 --split train --save_probabilities --gpu 0
python student_teacher_pipeline/scripts/generate_student_dataset.py
python student_teacher_pipeline/scripts/build_student_teacher_prior_manifest.py
python train_val_test_nnunet_scripts/create_train_val_split.py --dataset_id 15
python train_val_test_nnunet_scripts/preprocess.py --dataset_id 15
python student_teacher_pipeline/scripts/train_student.py --gpu 0 --num_gpus 4
```

Before using `nnUNetTrainer_student_teacher`, copy it into the active nnU-Net
environment:

```bash
python student_teacher_pipeline/set_up_env/install_student_teacher_trainer.py
```

The trainer expects teacher `.npz` probability maps listed in
`student_teacher_pipeline/teacher_prior_manifest.json`.
