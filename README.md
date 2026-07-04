
# CuFP: Curriculum-Enhanced Flow Policies for Generalizable Robotic Manipulation

This repository contains the course project code for **CuFP: Curriculum-Enhanced Flow Policies for Generalizable Robotic Manipulation**.

The code is built on top of `robomimic` and `robosuite`. It provides training and evaluation scripts for offline visuomotor policies on the `Lift` and `Square` tasks under clean and perturbed environments.

## Setup

Build the Docker image:

```bash
docker build -t robomimic-robustness .
````

Start a container:

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    robomimic-robustness
```

Enter the workspace:

```bash
cd /workspace
```

Before training or evaluation, check that the dataset paths in the config files are correct.

## Training

Training is launched with robomimic config files.

Example:

```bash
python robomimic/scripts/train.py \
    --config configs/lift_image_flow_matching_video_eval.json
```

Common configs:

```bash
# Lift
python robomimic/scripts/train.py --config configs/lift_image_bc_video_eval.json
python robomimic/scripts/train.py --config configs/lift_image_diffusion_policy_video_eval.json
python robomimic/scripts/train.py --config configs/lift_image_flow_matching_video_eval.json
python robomimic/scripts/train.py --config configs/lift_image_flow_matching_x_video_eval.json

# Square
python robomimic/scripts/train.py --config configs/square_image_bc_video_eval.json
python robomimic/scripts/train.py --config configs/square_image_diffusion_policy_video_eval.json
python robomimic/scripts/train.py --config configs/square_image_flow_matching_video_eval.json
python robomimic/scripts/train.py --config configs/square_image_flow_matching_x_video_eval.json
```

Curriculum fine-tuning:

```bash
bash eval_scripts/train_rwr.sh
```

## Evaluation

Evaluation scripts are provided in `eval_scripts/`.

```bash
# Lift perturbation evaluation
bash eval_scripts/run_obj.sh

# Square perturbation evaluation
bash eval_scripts/run_obj_square.sh
```

Single-model evaluation:

```bash
bash eval_scripts/eval_one_model.sh
bash eval_scripts/eval_one_model_square.sh
bash eval_scripts/eval_one_model_square_parallel.sh
```

## Result Files

The main result summaries are stored in:

```text
success_rate_summary.md
best_success_rate_table.md
```

Figures used in the report are stored in:

```text
figures/
```

## Notes

* Please verify dataset paths before running scripts.
* Checkpoint paths may need to be adjusted in the evaluation scripts.
* The code depends on the modified `robomimic` and `robosuite` code included in this repository.

## Acknowledgements

This project is based on the open-source `robomimic` and `robosuite` frameworks.


