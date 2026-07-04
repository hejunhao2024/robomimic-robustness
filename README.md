
# CuFP: Curriculum-Enhanced Flow Policies for Generalizable Robotic Manipulation

This repository is the course project code for **CuFP: Curriculum-Enhanced Flow Policies for Generalizable Robotic Manipulation**.

We study the robustness of vision-based robotic manipulation policies under clean and perturbed environments. The project is built on top of **robomimic** and **robosuite**, and evaluates several representative offline policy learning methods on the `Lift` and `Square` tasks.

---

## Overview

Most robot imitation learning benchmarks evaluate policies only in the same environment used for training. In this project, we focus on a more practical question:

> Can a policy trained from offline demonstrations still work when object appearance, camera view, object properties, or task dynamics change?

We compare multiple policy families:

- Behavior Cloning (`BC`)
- Recurrent Behavior Cloning (`BC-RNN`)
- Transformer Behavior Cloning (`BC-Transformer`)
- Conservative Q-Learning (`CQL`)
- Diffusion Policy
- Flow Matching Policy
- Flow-X, a clean-action prediction variant of Flow Matching

We also propose a curriculum-enhanced Flow policy to improve robustness on the more difficult `Square` task.

---

## Tasks

We use two robosuite manipulation tasks through the robomimic pipeline.

| Task       | Description                                                                                                             |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| `Lift`   | Grasp a cube and lift it. This task mainly tests object localization and stable grasping.                               |
| `Square` | Pick up a square nut and insert it onto a peg. This task requires more precise alignment and contact-sensitive control. |

Each policy uses image observations and robot proprioception.

---

## Environment Perturbations

Policies are trained on clean demonstrations and evaluated under several controlled shifts.

| Perturbation         | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| `ObjectPerturb`    | Changes object size, geometry, or initialization.                           |
| `ColorPerturb`     | Changes object and table appearance.                                        |
| `CameraPerturb`    | Changes camera pose or field of view.                                       |
| `VisualOOD`        | Combines visual appearance and camera changes.                              |
| `TaskDynamicsHard` | Uses harder initialization, lower friction, or stricter success conditions. |

---

## Method

### Flow Policy

The Flow policy generates action chunks conditioned on recent image observations and robot proprioception.

Given an expert action chunk 

$$
x_1
$$

, Gaussian noise 
$$
x_0
$$

, and interpolation time $$t$$, we construct:

$$
x_t = (1 - t)x_0 + tx_1
$$

The velocity target is:

$$
v^* = x_1 - x_0
$$

The Flow Matching objective is:

$$
\mathcal{L}_{flow}
=
\mathbb{E}
\left[
\|v_\theta(x_t,t,c_t)-v^*\|_2^2
\right]
$$

At inference time, the policy starts from Gaussian noise and integrates the learned velocity field to generate an action chunk.

### Flow-X

Flow-X is a clean-action prediction variant. Instead of predicting the velocity directly, it predicts the clean action chunk:

$$
\mathcal{L}_{x}
=
\mathbb{E}
\left[
\|x_\theta(x_t,t,c_t)-x_1\|_2^2
\right]
$$

The predicted clean action is then converted into a velocity during sampling.

### Visual Data Augmentation

To improve visual robustness, expert trajectories are replayed in perturbed simulator environments. The RGB observations are re-rendered with changed object colors, table textures, lighting, background, or camera poses, while the robot states and expert actions are kept unchanged.

This expands the visual distribution without changing the original control supervision.

### Curriculum Learning

For the `Square` task, we decompose the task into ordered stages:

```text
grasp -> lift -> align -> success
```

During curriculum fine-tuning, online rollouts are divided into stage-level segments. Successful segments are retained and weighted according to:

- current training progress,
- stage difficulty,
- stage-wise success rate,
- trajectory segment quality.

The final objective combines the original demonstration loss and the weighted online loss:

$$
\mathcal{L}
=
\mathcal{L}_{demo}
+
\lambda \mathcal{L}_{online}
$$

The goal is to improve robustness while preserving the clean-task ability of the pretrained policy.

---

## Repository Structure

```text
.
├── configs/
│   ├── lift_image_bc_video_eval.json
│   ├── lift_image_bc_rnn_video_eval.json
│   ├── lift_image_proprio_bc_transformer_video_eval.json
│   ├── lift_image_diffusion_policy_video_eval.json
│   ├── lift_image_flow_matching_video_eval.json
│   ├── lift_image_flow_matching_x_video_eval.json
│   ├── lift_image_cql_video_eval.json
│   ├── square_image_bc_video_eval.json
│   ├── square_image_bc_rnn_video_eval.json
│   ├── square_image_proprio_bc_transformer_video_eval.json
│   ├── square_image_diffusion_policy_video_eval.json
│   ├── square_image_flow_matching_video_eval.json
│   ├── square_image_flow_matching_x_video_eval.json
│   ├── square_image_cql_video_eval.json
│   ├── final_square_flow_augdata.json
│   ├── final_square_flow_x.json
│   ├── final_square_flow_x_augdata.json
│   ├── final_square_flow_rwr.json
│   └── train_rwr.sh
│
├── eval_scripts/
│   ├── eval_one_model.sh
│   ├── eval_one_model_square.sh
│   ├── eval_one_model_square_parallel.sh
│   ├── run_obj.sh
│   └── run_obj_square.sh
│
├── figures/
│   ├── lift_success_rate_plot.png
│   ├── square_success_rate_plot.png
│   └── perturbed.png
│
├── robomimic/
│   └── Modified robomimic codebase.
│
├── third_party/
│   └── robosuite/
│       └── Modified robosuite environments.
│
├── Dockerfile
├── command.md
├── success_rate_summary.md
├── best_success_rate_table.md
└── README.md
```

---

## Installation

A Dockerfile is provided for environment setup.

```bash
docker build -t robomimic-robustness .
```

Start a container:

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    robomimic-robustness
```

Inside the container:

```bash
cd /workspace
```

Please check the dataset paths in the config files before training.

---

## Training

Training is launched with robomimic config files.

Example:

```bash
python robomimic/scripts/train.py \
    --config configs/lift_image_flow_matching_video_eval.json
```

Other commonly used configs:

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

For curriculum training, see:

```bash
bash configs/train_rwr.sh
```

---

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

Please modify checkpoint paths and dataset paths according to your local environment.

---

## Main Results

The following table reports the best success rate among selected checkpoints.

| Task            |   BC | BC-RNN | BC-T | DiffPol | Flow | Flow-X |  CQL |
| --------------- | ---: | -----: | ---: | ------: | ---: | -----: | ---: |
| Lift            | 0.96 |   0.68 | 0.92 |    1.00 | 1.00 |   1.00 | 0.00 |
| Lift + Obj.     | 0.52 |   0.52 | 0.48 |    0.92 | 0.92 |   0.96 | 0.00 |
| Lift + Color    | 0.00 |   0.00 | 0.04 |    0.88 | 1.00 |   1.00 | 0.00 |
| Lift + Camera   | 0.40 |   0.08 | 0.56 |    0.88 | 0.96 |   1.00 | 0.00 |
| Lift + Visual   | 0.08 |   0.00 | 0.04 |    0.76 | 0.84 |   1.00 | 0.00 |
| Lift + Env      | 0.64 |   0.52 | 0.56 |    0.72 | 0.92 |   0.96 | 0.00 |
| Square          | 0.28 |   0.32 | 0.48 |    0.76 | 0.84 |   0.64 | 0.00 |
| Square + Obj.   | 0.08 |   0.04 | 0.36 |    0.60 | 0.72 |   0.52 | 0.00 |
| Square + Color  | 0.00 |   0.00 | 0.00 |    0.00 | 0.00 |   0.00 | 0.00 |
| Square + Camera | 0.00 |   0.00 | 0.00 |    0.20 | 0.00 |   0.00 | 0.00 |
| Square + Visual | 0.00 |   0.00 | 0.00 |    0.00 | 0.00 |   0.00 | 0.00 |
| Square + Env    | 0.08 |   0.04 | 0.16 |    0.52 | 0.40 |   0.40 | 0.00 |

More details are available in:

```text
success_rate_summary.md
best_success_rate_table.md
```

---

## Curriculum Results on Square

We further compare Flow, Flow with visual data augmentation, and curriculum-enhanced Flow on `Square`.

| Task            |  Flow | Data Aug. | Curriculum |
| --------------- | ----: | --------: | ---------: |
| Square          |  0.80 |      0.54 |       0.66 |
| Square + Obj.   |  0.56 |      0.58 |       0.54 |
| Square + Color  |  0.00 |      0.16 |       0.22 |
| Square + Camera |  0.00 |      0.24 |       0.24 |
| Square + Visual |  0.00 |      0.08 |       0.12 |
| Square + Env    |  0.52 |      0.26 |       0.44 |
| Average         | 0.313 |     0.310 |      0.370 |

---

## Visualizations

### Lift Success Rate

![Lift Success Rate](figures/lift_success_rate_plot.png)

### Square Success Rate

![Square Success Rate](figures/square_success_rate_plot.png)

### Perturbed Environments

![Perturbed Environments](figures/perturbed.png)

---

## Key Findings

1. Generative action policies are generally stronger than standard BC baselines.
2. Diffusion Policy and Flow Matching achieve better robustness on `Lift`, especially under visual perturbations.
3. `Square` is much harder than `Lift`. Even strong policies can fail under color and camera shifts because insertion requires accurate visual alignment.
4. Flow-X performs very well on `Lift`, but it is not universally better than standard Flow Matching on `Square`.
5. CQL obtains zero success in our current image-based setting. This should be interpreted as a limitation of this specific setup, not as a general conclusion about offline reinforcement learning.
6. Visual data augmentation improves robustness to visual shifts, but may reduce clean-environment performance.
7. Curriculum learning gives the best average robustness on `Square`, but overly hard training can still hurt performance.

---

## Author Contributions

- **Binghao Cai** implemented the Flow-policy algorithms, contributed to the experimental design and project organization, and participated in experiment analysis and report writing.
- **Junhao He** conducted the benchmark and curriculum-learning experiments, designed and implemented the curriculum-learning method, and participated in result analysis and report writing.
- **Peijian Chen** investigated the task and environment settings, developed the evaluation and environment-modification pipeline, designed the data-augmentation method, and participated in experiment analysis and report writing.

---

## Limitations

- The success rates are estimated from a limited number of rollouts.
- The benchmark uses best-checkpoint selection, while the curriculum study uses a separate matched evaluation protocol.
- The perturbation environments are manually designed and may not cover all real-world distribution shifts.
- Stage-aware curriculum learning depends on task-specific stage detectors.
- CQL may require different hyperparameters or training settings to work well in this image-based setup.

---

## Acknowledgements

This project is based on the open-source `robomimic` and `robosuite` frameworks.

```bibtex
@inproceedings{mandlekar2021matters,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Mandlekar, Ajay and Xu, Danfei and Wong, Josiah and Nasiriany, Soroush and Wang, Chen and Kulkarni, Rohun and Fei-Fei, Li and Savarese, Silvio and Zhu, Yuke and Martin-Martin, Roberto},
  booktitle={Conference on Robot Learning},
  year={2021}
}

@inproceedings{zhu2020robosuite,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Zhu, Yuke and Wong, Josiah and Mandlekar, Ajay and Martin-Martin, Roberto and Joshi, Abhishek and Nasiriany, Soroush and Zhu, Yifeng},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
