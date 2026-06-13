"""
Online staged-reward-guided reward-weighted flow finetuning.
"""
import argparse
import atexit
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import traceback
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.python_utils as PyUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import RolloutPolicy, algo_factory
from robomimic.config import config_factory
from robomimic.envs.reward_wrappers import StagedRewardWrapper
from robomimic.utils.flow_rollout_buffer import FlowRolloutBuffer
from robomimic.utils.log_utils import DataLogger, PrintLogger, flush_warnings


STAGE_NAMES = {
    0: "none",
    1: "stable_grasp",
    2: "lift",
    3: "hover",
    4: "success",
}


def _print_stage_interval_debug(buffer, iter_idx, max_segments=12):
    """
    Validate and display stage-interval credit assignment.

    Each printed sample should contain actions from exactly one target interval:
    start -> grasp, grasp -> lift, lift -> hover, or hover -> success.
    """
    if iter_idx > 2:
        return

    print("\n========== Stage-interval curriculum debug ==========")
    for episode in buffer.episodes:
        print(
            "[episode]",
            "id=", episode.get("episode_id"),
            "num_steps=", episode.get("num_steps", len(episode.get("steps", []))),
            "highest_stage=", episode.get("highest_stage_name", "unknown"),
            "achievement_steps=", episode.get("stage_achievement_steps", {}),
            "success=", bool(episode.get("success", False)),
        )

    shown = 0
    for stage_id in range(1, 5):
        stage_samples = [
            seg for seg in buffer.segments
            if int(seg.get("credited_stage_id", 0)) == stage_id
            and float(seg.get("weight", 0.0)) > 0.0
        ]
        # Prefer boundary-crossing / partially masked examples.
        stage_samples.sort(
            key=lambda seg: float(
                np.asarray(seg["action_mask"]).sum()
                / max(
                    float(
                        np.asarray(
                            seg.get("original_action_mask", seg["action_mask"])
                        ).sum()
                    ),
                    1.0,
                )
            )
        )
        for seg in stage_samples[:3]:
            mask = np.asarray(seg["action_mask"], dtype=np.float32)
            original_mask = np.asarray(
                seg.get("original_action_mask", mask),
                dtype=np.float32,
            )
            step_indices = np.asarray(
                seg["action_step_indices"],
                dtype=np.int64,
            )
            interval_start = int(seg["stage_interval_start"])
            interval_end = int(seg["stage_interval_end"])
            credited_steps = step_indices[mask > 0.0]

            assert len(credited_steps) > 0
            assert np.all(credited_steps >= interval_start)
            assert np.all(credited_steps <= interval_end)
            assert np.all(
                mask[
                    (step_indices < interval_start)
                    | (step_indices > interval_end)
                ] == 0.0
            )
            if len(mask) > 0:
                assert mask[0] == 0.0

            print(
                "[stage sample]",
                "episode=", int(seg["episode_id"]),
                "timestep=", int(seg["timestep"]),
                "target_stage=", seg["credited_stage_name"],
                "interval=", [interval_start, interval_end],
                "valid_actions=", int(mask.sum()),
                "return=", round(float(seg["return"]), 4),
                "advantage=", round(float(seg["advantage"]), 4),
                "stage_priority=", round(float(seg["stage_priority"]), 4),
                "weight=", round(float(seg["weight"]), 4),
            )
            print("  steps         =", step_indices.tolist())
            print("  original_mask =", original_mask.astype(np.int32).tolist())
            print("  stage_mask    =", mask.astype(np.int32).tolist())
            shown += 1
            if shown >= int(max_segments):
                break
        if shown >= int(max_segments):
            break

    print(
        "[curriculum summary]",
        "source_segments=", buffer.num_source_segments,
        "stage_samples=", len(buffer.segments),
        "positive_stage_samples=", len([
            seg for seg in buffer.segments
            if float(seg.get("weight", 0.0)) > 0.0
        ]),
        "stage_priorities=", {
            buffer.STAGE_NAMES[s]: round(
                float(buffer.stage_priorities[s]),
                4,
            )
            for s in range(1, 5)
        },
    )
    print("=====================================================\n")



def _print_epoch_stage_report(rollout_stats, iter_idx):
    """Print fine-grained task-stage completion statistics every epoch."""
    labels = {
        "stable_grasp": "stable_grasp",
        "lift": "lift",
        "hover": "peg_align",
        "success": "success",
    }

    completion = {}
    for internal_name, display_name in labels.items():
        completion[display_name] = {
            "reached": int(
                rollout_stats.get(
                    f"num_episodes_reaching_{internal_name}", 0
                )
            ),
            "rate": float(
                rollout_stats.get(
                    f"rate_reaching_{internal_name}", 0.0
                )
            ),
            "ended_here": int(
                rollout_stats.get(
                    f"num_episodes_ending_at_{internal_name}", 0
                )
            ),
            "mean_first_step": float(
                rollout_stats.get(
                    f"mean_first_step_{internal_name}", -1.0
                )
            ),
            "median_first_step": float(
                rollout_stats.get(
                    f"median_first_step_{internal_name}", -1.0
                )
            ),
            "stage_samples": int(
                rollout_stats.get(
                    f"num_stage_segments_{internal_name}", 0
                )
            ),
            "trainable_actions": int(
                rollout_stats.get(
                    f"num_trainable_actions_{internal_name}", 0
                )
            ),
            "mean_weight": float(
                rollout_stats.get(
                    f"mean_weight_{internal_name}", 0.0
                )
            ),
            "curriculum_priority": float(
                rollout_stats.get(
                    f"stage_priority_{internal_name}", 0.0
                )
            ),
        }

    report = {
        "epoch": int(iter_idx),
        "episodes": int(rollout_stats.get("num_episodes", 0)),
        "no_progress": int(
            rollout_stats.get("num_episodes_no_progress", 0)
        ),
        "completion": completion,
        "conditional_conversion": {
            "lift_given_grasp": float(
                rollout_stats.get(
                    "lift_given_stable_grasp", 0.0
                )
            ),
            "peg_align_given_lift": float(
                rollout_stats.get("hover_given_lift", 0.0)
            ),
            "success_given_peg_align": float(
                rollout_stats.get("success_given_hover", 0.0)
            ),
        },
        "geometry": {
            "mean_xy_distance_at_first_peg_align_m": float(
                rollout_stats.get(
                    "mean_hover_xy_dist_at_first_hover", -1.0
                )
            )
        },
        "online_data": {
            "source_segments": int(
                rollout_stats.get("num_source_segments", 0)
            ),
            "eligible_source_segments": int(
                rollout_stats.get(
                    "num_eligible_source_segments", 0
                )
            ),
            "stage_samples": int(
                rollout_stats.get("num_stage_segments", 0)
            ),
            "positive_stage_samples": int(
                rollout_stats.get(
                    "num_positive_weight_segments", 0
                )
            ),
            "sample_uses": int(
                rollout_stats.get(
                    "consumed_online_samples", 0
                )
            ),
            "expected_sample_uses": int(
                rollout_stats.get(
                    "expected_online_sample_uses", 0
                )
            ),
        },
    }

    print("\n[StageCompletion]")
    print(json.dumps(report, sort_keys=True, indent=4))



def _config_get(section, key, default):
    """
    Read an optional robomimic Config value without requiring every new RWR
    field to exist in older JSON files.
    """
    try:
        return section.get(key, default)
    except Exception:
        try:
            return section[key]
        except Exception:
            return default



def _as_plain_mapping(value):
    if value is None:
        return {}
    try:
        return {
            key: value[key]
            for key in value.keys()
        }
    except Exception:
        try:
            return dict(value)
        except Exception:
            return {}


def _read_stage_weight_map(section, key, default):
    raw = _config_get(section, key, None)
    raw = _as_plain_mapping(raw)
    result = {}
    for stage_id, stage_name in STAGE_NAMES.items():
        if stage_id == 0:
            continue
        result[stage_id] = float(
            raw.get(stage_name, default[stage_id])
        )
    return result


def compute_stage_curriculum(config, iter_idx, num_iters):
    """
    Smoothly decay early-stage priorities while keeping final success high.

    Defaults:
        early: grasp=.8, lift=2.0, hover=1.4, success=2.5
        late:  grasp=.25, lift=.5, hover=1.0, success=2.5
    """
    rwr_config = config.algo.rwr
    curriculum = _config_get(rwr_config, "curriculum", None)

    default_early = {
        1: 0.8,
        2: 2.0,
        3: 1.4,
        4: 2.5,
    }
    default_late = {
        1: 0.25,
        2: 0.5,
        3: 1.0,
        4: 2.5,
    }

    if curriculum is None:
        enabled = True
        schedule = "cosine"
        decay_iters = max(1, int(round(0.6 * num_iters)))
        early = default_early
        late = default_late
    else:
        enabled = bool(_config_get(curriculum, "enabled", True))
        schedule = str(_config_get(curriculum, "schedule", "cosine"))
        decay_iters = int(
            _config_get(
                curriculum,
                "decay_iters",
                max(1, int(round(0.6 * num_iters))),
            )
        )
        early = _read_stage_weight_map(
            curriculum,
            "early_stage_weights",
            default_early,
        )
        late = _read_stage_weight_map(
            curriculum,
            "late_stage_weights",
            default_late,
        )

    if not enabled:
        priorities = {
            stage_id: float(early[stage_id])
            for stage_id in range(1, 5)
        }
        return priorities, 0.0

    progress = np.clip(
        (int(iter_idx) - 1) / max(int(decay_iters) - 1, 1),
        0.0,
        1.0,
    )
    if schedule == "cosine":
        alpha = 0.5 * (1.0 + np.cos(np.pi * progress))
    elif schedule == "linear":
        alpha = 1.0 - progress
    else:
        raise ValueError(
            "curriculum schedule must be 'cosine' or 'linear', got "
            f"{schedule}"
        )

    priorities = {
        stage_id: float(
            late[stage_id]
            + alpha * (early[stage_id] - late[stage_id])
        )
        for stage_id in range(1, 5)
    }
    return priorities, float(progress)


def _extract_new_stage_ids(info, previous_max_stage_id):
    """
    Normalize stage-transition metadata emitted by StagedRewardWrapper.

    The fallback based on max_stage_id keeps the collector compatible with a
    partially upgraded wrapper, while the preferred path uses the explicit
    newly_achieved_stage_ids field.
    """
    explicit_ids = info.get("newly_achieved_stage_ids", ())
    if explicit_ids is None:
        explicit_ids = ()
    if isinstance(explicit_ids, (int, np.integer)):
        explicit_ids = (int(explicit_ids),)

    stage_ids = []
    for stage_id in explicit_ids:
        stage_id = int(stage_id)
        if stage_id > previous_max_stage_id:
            stage_ids.append(stage_id)

    if len(stage_ids) == 0:
        max_stage_id = int(info.get("max_stage_id", previous_max_stage_id))
        if max_stage_id > previous_max_stage_id:
            stage_ids = list(range(previous_max_stage_id + 1, max_stage_id + 1))

    return stage_ids


def _summarize_steps(steps, global_step_indices):
    """
    Summarize raw staged-state scores and one-time transition rewards over the
    valid future part of one prediction window.
    """
    valid_steps = [
        steps[int(step_idx)]
        for step_idx in global_step_indices
        if 0 <= int(step_idx) < len(steps)
    ]
    if len(valid_steps) == 0:
        return {}

    staged_infos = [step["staged"] for step in valid_steps]
    summary = {}

    max_keys = (
        "r_reach",
        "r_grasp",
        "r_lift",
        "r_hover",
        "success",
        "current_stage_id",
        "max_stage_id",
        "stable_grasp",
        "lifted",
        "hovering",
    )
    sum_keys = (
        "r_grasp_transition",
        "r_lift_transition",
        "r_hover_transition",
        "r_success",
        "reward_progress",
        "stage_transition_reward",
        "reward_total",
        "base_reward",
    )

    for key in max_keys:
        summary[key] = float(max(float(info.get(key, 0.0)) for info in staged_infos))
    for key in sum_keys:
        summary[key] = float(sum(float(info.get(key, 0.0)) for info in staged_infos))

    return summary


def _build_prediction_segments(
    steps,
    chunk_records,
    policy_actions,
    episode_id,
    observation_horizon,
    prediction_horizon,
    stage_achievement_steps,
    highest_stage_id,
):
    """
    Reconstruct online training targets with the same temporal alignment as the
    offline SequenceDataset.

    For observation_horizon=2 and a chunk beginning at environment step s:

        target[0] = a_{s-1}   (history context, never trained)
        target[1] = a_s
        ...
        target[15] = a_{s+14}

    The future actions may come from later receding-horizon replans. This is
    intentional: they are the actions that were actually executed in the
    environment, matching the semantics of an offline demonstration trajectory.
    """
    if len(policy_actions) == 0:
        return []

    policy_actions = np.asarray(policy_actions, dtype=np.float32)
    action_dim = int(policy_actions.shape[-1])
    num_steps = int(policy_actions.shape[0])
    execution_start_index = int(observation_horizon) - 1

    if execution_start_index < 0 or execution_start_index >= prediction_horizon:
        raise ValueError(
            "observation_horizon - 1 must lie inside prediction_horizon, got "
            f"observation_horizon={observation_horizon}, "
            f"prediction_horizon={prediction_horizon}"
        )

    segments = []
    for record in chunk_records:
        start_step = int(record["start_step"])

        action_chunk = np.zeros(
            (prediction_horizon, action_dim),
            dtype=np.float32,
        )
        action_mask = np.zeros((prediction_horizon,), dtype=np.float32)
        action_step_indices = np.full(
            (prediction_horizon,),
            fill_value=-1,
            dtype=np.int64,
        )

        for position in range(prediction_horizon):
            global_step = start_step + position - execution_start_index

            if 0 <= global_step < num_steps:
                action_chunk[position] = policy_actions[global_step]
                action_step_indices[position] = global_step

                # Position(s) before execution_start_index are history context.
                # They match the frame-stacked dataset format but must not
                # receive online credit.
                if position >= execution_start_index:
                    action_mask[position] = 1.0
            elif global_step < 0:
                # Match padded sequence behavior at the beginning. This value is
                # context only because its mask remains zero.
                action_chunk[position] = policy_actions[0]
            else:
                # Match padded sequence behavior near the episode tail. These
                # unavailable future positions also remain masked out.
                action_chunk[position] = policy_actions[-1]

        trainable_global_steps = action_step_indices[
            action_mask.astype(bool)
        ]
        end_step = (
            int(trainable_global_steps.max()) + 1
            if len(trainable_global_steps) > 0
            else start_step
        )

        reward_seq = np.zeros((prediction_horizon,), dtype=np.float32)
        for position, global_step in enumerate(action_step_indices):
            if action_mask[position] > 0.0 and 0 <= global_step < len(steps):
                reward_seq[position] = float(steps[int(global_step)]["reward"])

        staged_summary = _summarize_steps(
            steps=steps,
            global_step_indices=trainable_global_steps,
        )
        window_success = any(
            bool(steps[int(global_step)]["success"])
            for global_step in trainable_global_steps
            if 0 <= int(global_step) < len(steps)
        )

        segments.append({
            "obs": deepcopy(record["obs"]),
            "action_chunk": action_chunk,
            "action_mask": action_mask,
            "action_step_indices": action_step_indices,
            "execution_start_index": execution_start_index,
            "episode_id": int(episode_id),
            "timestep": start_step,
            "start_step": start_step,
            "end_step": end_step,
            "executed_chunk_end_step": int(record["executed_chunk_end_step"]),
            "segment_reward": float(reward_seq.sum()),
            "reward_seq": reward_seq,
            "staged_summary": staged_summary,
            "done": bool(record["done"]),
            "success": bool(window_success),
            "highest_stage_id": int(highest_stage_id),
            "highest_stage_name": STAGE_NAMES.get(
                int(highest_stage_id),
                f"stage_{highest_stage_id}",
            ),
            "stage_achievement_steps": deepcopy(stage_achievement_steps),
        })

    return segments


def tensor_action_chunk_to_numpy(rollout_policy, action_chunk):
    if torch.is_tensor(action_chunk):
        action_chunk = action_chunk.detach().cpu().numpy()

    if rollout_policy.action_normalization_stats is None:
        return action_chunk

    action_keys = rollout_policy.policy.global_config.train.action_keys
    action_shapes = {
        key: rollout_policy.action_normalization_stats[key]["offset"].shape[1:]
        for key in rollout_policy.action_normalization_stats
    }
    action_dict = PyUtils.vector_to_action_dict(
        action_chunk,
        action_shapes=action_shapes,
        action_keys=action_keys,
    )
    action_dict = ObsUtils.unnormalize_dict(
        action_dict,
        normalization_stats=rollout_policy.action_normalization_stats,
    )
    action_config = rollout_policy.policy.global_config.train.action_config
    for key, value in action_dict.items():
        this_format = action_config[key].get("format", None)
        if this_format == "rot_6d":
            rot_6d = torch.from_numpy(value)
            conversion_format = action_config[key].get("convert_at_runtime", "rot_axis_angle")
            if conversion_format == "rot_axis_angle":
                rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).numpy()
            elif conversion_format == "rot_euler":
                rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d, convention="XYZ").numpy()
            else:
                raise ValueError("unknown rotation conversion format: {}".format(conversion_format))
            action_dict[key] = rot
    return PyUtils.action_dict_to_vector(action_dict, action_keys=action_keys)



def sample_action_chunk(rollout_policy, model, obs):
    """
    Sample one executable action chunk.

    Returns both:
      - policy_action_chunk: actions in the model / training representation;
      - env_action_chunk: actions converted to the environment representation.

    They are identical for the current config (no action normalization), but
    keeping both prevents online targets from silently switching spaces when
    normalization or runtime rotation conversion is enabled later.
    """
    obs_t = rollout_policy._prepare_observation(obs, batched_ob=False)
    with torch.no_grad():
        action_chunk = model.sample_action_chunk(obs_dict=obs_t)

    policy_action_chunk = action_chunk[0].detach().cpu().numpy().astype(np.float32)
    env_action_chunk = tensor_action_chunk_to_numpy(
        rollout_policy,
        action_chunk[0],
    ).astype(np.float32)
    return policy_action_chunk, env_action_chunk



def sample_action_chunks_batched(rollout_policy, model, observations):
    """
    Batch policy inference across several independent environments.

    Environment stepping still uses independent env instances, but the expensive
    observation encoder and every Flow ODE function evaluation run with batch
    size B instead of B=1.
    """
    if len(observations) == 0:
        return [], []

    prepared = [
        rollout_policy._prepare_observation(obs, batched_ob=False)
        for obs in observations
    ]
    keys = prepared[0].keys()
    obs_batch = {
        key: torch.cat([item[key] for item in prepared], dim=0)
        for key in keys
    }

    with torch.no_grad():
        action_chunks = model.sample_action_chunk(obs_dict=obs_batch)

    if action_chunks.shape[0] != len(observations):
        raise RuntimeError(
            "batched policy output has wrong batch dimension: "
            f"{action_chunks.shape[0]} != {len(observations)}"
        )

    policy_chunks = (
        action_chunks.detach().cpu().numpy().astype(np.float32)
    )
    env_chunks = [
        tensor_action_chunk_to_numpy(
            rollout_policy,
            action_chunks[index],
        ).astype(np.float32)
        for index in range(len(observations))
    ]
    return list(policy_chunks), env_chunks



def maybe_wrap_env_for_online_training(env, config):
    env = EnvUtils.wrap_env_from_config(env, config=config)
    rwr_config = config.algo.rwr
    env = StagedRewardWrapper(
        env,
        use_staged_reward=bool(rwr_config.use_staged_reward),
        success_bonus=float(rwr_config.success_bonus),
        grasp_bonus=float(_config_get(rwr_config, "grasp_bonus", 1.0)),
        lift_bonus=float(_config_get(rwr_config, "lift_bonus", 1.0)),
        hover_bonus=float(_config_get(rwr_config, "hover_bonus", 1.0)),
        stable_grasp_steps=int(
            _config_get(rwr_config, "stable_grasp_steps", 3)
        ),
        stable_hover_steps=int(
            _config_get(rwr_config, "stable_hover_steps", 3)
        ),
        grasp_threshold=float(
            _config_get(rwr_config, "grasp_threshold", 0.30)
        ),
        lift_threshold=float(
            _config_get(rwr_config, "lift_threshold", 0.40)
        ),
        hover_threshold=float(
            _config_get(rwr_config, "hover_threshold", 0.60)
        ),
        use_geometric_hover=bool(
            _config_get(rwr_config, "use_geometric_hover", True)
        ),
        hover_xy_threshold=float(
            _config_get(rwr_config, "hover_xy_threshold", 0.05)
        ),
    )
    return env


def create_one_env(config, env_meta, shape_meta, env_name):
    local_meta = deepcopy(env_meta)
    local_meta["lang"] = None
    env = EnvUtils.create_env_from_metadata(
        env_meta=local_meta,
        env_name=env_name,
        render=False,
        render_offscreen=(
            config.experiment.render_video
            or shape_meta["use_images"]
            or shape_meta["use_depths"]
        ),
        use_image_obs=(
            shape_meta["use_images"] or shape_meta["use_depths"]
        ),
        use_depth_obs=shape_meta["use_depths"],
    )
    return maybe_wrap_env_for_online_training(env, config)


def create_envs(config, env_meta, shape_meta):
    env_name = (
        config.train.online.env_name
        if config.train.online.env_name is not None
        else env_meta["env_name"]
    )
    env_names = [env_name]
    if config.experiment.additional_envs is not None:
        env_names.extend(config.experiment.additional_envs)

    return OrderedDict(
        (
            name,
            create_one_env(
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                env_name=name,
            ),
        )
        for name in env_names
    )


def create_train_env_pool(
    config,
    env_meta,
    shape_meta,
    first_env,
    num_envs,
):
    num_envs = max(1, int(num_envs))
    env_name = (
        config.train.online.env_name
        if config.train.online.env_name is not None
        else env_meta["env_name"]
    )
    pool = [first_env]
    for _ in range(1, num_envs):
        pool.append(
            create_one_env(
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                env_name=env_name,
            )
        )
    return pool



_STAGE_INFO_KEYS = (
    "success",
    "current_stage_id",
    "max_stage_id",
    "new_stage",
    "new_stage_id",
    "newly_achieved_stage_ids",
    "r_reach",
    "r_grasp",
    "r_lift",
    "r_hover",
    "r_grasp_transition",
    "r_lift_transition",
    "r_hover_transition",
    "r_success",
    "reward_progress",
    "stage_transition_reward",
    "reward_total",
    "base_reward",
    "grasp_contact",
    "grasp_streak",
    "stable_grasp",
    "lifted",
    "peg_aligned_now",
    "hovering",
    "hover_streak",
    "hover_detection_source",
    "geometry_available",
    "hover_xy_dist",
    "hover_xy_threshold",
    "nut_height_above_table",
    "nut_x",
    "nut_y",
    "nut_z",
    "peg_x",
    "peg_y",
    "peg_z",
)


def _to_plain_value(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return tuple(_to_plain_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_to_plain_value(item) for item in value)
    return value


def _select_stage_info(info):
    info = dict(info)
    return {
        key: _to_plain_value(info[key])
        for key in _STAGE_INFO_KEYS
        if key in info
    }


def _rebuild_worker_config(config_dict):
    config = config_factory(config_dict["algo_name"])
    config.unlock()
    config.update(config_dict)
    config.lock()
    return config


def _subprocess_env_worker(
    connection,
    worker_id,
    config_dict,
    env_meta,
    shape_meta,
):
    """Own one robosuite environment inside an isolated spawned process."""
    env = None
    current_obs = None
    try:
        torch.set_num_threads(1)
        config = _rebuild_worker_config(config_dict)

        # ``spawn`` starts a fresh Python interpreter. Robomimic observation
        # modality mappings are process-local globals, so the initialization
        # performed in the parent process is not inherited by this worker.
        # This must happen before creating / resetting the environment.
        ObsUtils.initialize_obs_utils_with_config(config)

        worker_seed = int(config.train.seed) + 100003 * int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        env_name = (
            config.train.online.env_name
            if config.train.online.env_name is not None
            else env_meta["env_name"]
        )
        env = create_one_env(
            config=config,
            env_meta=env_meta,
            shape_meta=shape_meta,
            env_name=env_name,
        )
        try:
            if hasattr(env, "seed"):
                env.seed(worker_seed)
        except Exception:
            pass

        connection.send((
            "ready",
            {
                "worker_id": int(worker_id),
                "seed": int(worker_seed),
            },
        ))

        while True:
            command, payload = connection.recv()

            if command == "reset":
                current_obs = env.reset()
                connection.send(("ok", current_obs))
                continue

            if command == "step_chunk":
                env_action_chunk = np.asarray(
                    payload["env_action_chunk"],
                    dtype=np.float32,
                )
                max_steps = max(0, int(payload["max_steps"]))
                terminate_on_success = bool(
                    payload["terminate_on_success"]
                )

                transitions = []
                done = max_steps <= 0
                for env_action in env_action_chunk[:max_steps]:
                    next_obs, reward, env_done, info = env.step(env_action)
                    info = _select_stage_info(info)
                    step_success = bool(info.get("success", False))
                    done = bool(
                        env_done
                        or (
                            terminate_on_success
                            and step_success
                        )
                        or (
                            len(transitions) + 1 >= max_steps
                        )
                    )
                    transitions.append({
                        "reward": float(reward),
                        "env_done": bool(env_done),
                        "success": bool(step_success),
                        "done": bool(done),
                        "info": info,
                    })
                    current_obs = next_obs
                    if done:
                        break

                connection.send((
                    "ok",
                    {
                        "obs": current_obs,
                        "transitions": transitions,
                        "done": bool(done),
                    },
                ))
                continue

            if command == "close":
                try:
                    if env is not None and hasattr(env, "close"):
                        env.close()
                finally:
                    connection.send(("closed", None))
                    break

            raise ValueError(f"unknown worker command: {command}")

    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            connection.close()
        except Exception:
            pass


class SubprocessEnvPool:
    """
    Spawned robosuite workers with central batched GPU policy inference.

    All commands are sent before any result is received, so the environments
    execute their action chunks concurrently.
    """

    def __init__(self, config, env_meta, shape_meta, num_workers):
        self.num_workers = int(num_workers)
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")

        self._closed = False
        self._ctx = mp.get_context("spawn")
        self._connections = []
        self._processes = []

        config_dict = json.loads(json.dumps(config))

        for worker_id in range(self.num_workers):
            parent_connection, child_connection = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_subprocess_env_worker,
                args=(
                    child_connection,
                    worker_id,
                    config_dict,
                    env_meta,
                    shape_meta,
                ),
                daemon=True,
            )
            process.start()
            child_connection.close()
            self._connections.append(parent_connection)
            self._processes.append(process)

        for worker_id, connection in enumerate(self._connections):
            status, payload = connection.recv()
            if status != "ready":
                raise RuntimeError(
                    f"environment worker {worker_id} failed to start:\n"
                    f"{payload}"
                )

    def _receive(self, worker_id):
        status, payload = self._connections[int(worker_id)].recv()
        if status == "error":
            raise RuntimeError(
                f"environment worker {worker_id} failed:\n{payload}"
            )
        if status not in ("ok", "closed"):
            raise RuntimeError(
                f"unexpected worker status {status!r}"
            )
        return payload

    def reset(self, worker_ids):
        worker_ids = [int(worker_id) for worker_id in worker_ids]
        for worker_id in worker_ids:
            self._connections[worker_id].send(("reset", None))
        return {
            worker_id: self._receive(worker_id)
            for worker_id in worker_ids
        }

    def step_chunks(
        self,
        worker_ids,
        env_action_chunks,
        max_steps,
        terminate_on_success,
    ):
        worker_ids = [int(worker_id) for worker_id in worker_ids]
        for worker_id, env_action_chunk, this_max_steps in zip(
            worker_ids,
            env_action_chunks,
            max_steps,
        ):
            self._connections[worker_id].send((
                "step_chunk",
                {
                    "env_action_chunk": np.asarray(
                        env_action_chunk,
                        dtype=np.float32,
                    ),
                    "max_steps": int(this_max_steps),
                    "terminate_on_success": bool(
                        terminate_on_success
                    ),
                },
            ))

        return {
            worker_id: self._receive(worker_id)
            for worker_id in worker_ids
        }

    def close(self):
        if self._closed:
            return
        self._closed = True

        for connection, process in zip(
            self._connections,
            self._processes,
        ):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except Exception:
                    pass

        for worker_id, (connection, process) in enumerate(zip(
            self._connections,
            self._processes,
        )):
            if process.is_alive():
                try:
                    self._receive(worker_id)
                except Exception:
                    pass
                process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            try:
                connection.close()
            except Exception:
                pass


def _new_parallel_rollout_state(worker_id, episode_id, obs):
    return {
        "worker_id": int(worker_id),
        "episode_id": int(episode_id),
        "obs": obs,
        "steps": [],
        "chunk_records": [],
        "policy_actions": [],
        "stage_achievement_steps": {
            "stable_grasp": None,
            "lift": None,
            "hover": None,
            "success": None,
        },
        "highest_stage_id": 0,
        "timestep": 0,
        "success": False,
        "done": False,
    }


def _parallel_staged_record(
    info,
    reward,
    step_success,
    current_stage_id,
    max_stage_id,
    new_stage_ids,
):
    record = {
        "r_reach": float(info.get("r_reach", 0.0)),
        "r_grasp": float(info.get("r_grasp", 0.0)),
        "r_lift": float(info.get("r_lift", 0.0)),
        "r_hover": float(info.get("r_hover", 0.0)),
        "r_grasp_transition": float(
            info.get("r_grasp_transition", 0.0)
        ),
        "r_lift_transition": float(
            info.get("r_lift_transition", 0.0)
        ),
        "r_hover_transition": float(
            info.get("r_hover_transition", 0.0)
        ),
        "r_success": float(info.get("r_success", 0.0)),
        "reward_progress": float(
            info.get("reward_progress", reward)
        ),
        "stage_transition_reward": float(
            info.get("stage_transition_reward", reward)
        ),
        "reward_total": float(
            info.get("reward_total", reward)
        ),
        "base_reward": float(
            info.get("base_reward", reward)
        ),
        "success": float(step_success),
        "current_stage_id": float(current_stage_id),
        "max_stage_id": float(max_stage_id),
        "new_stage": float(bool(info.get("new_stage", False))),
        "new_stage_id": float(
            info.get("new_stage_id", max_stage_id)
        ),
        "newly_achieved_stage_ids": tuple(
            int(stage_id) for stage_id in new_stage_ids
        ),
        "grasp_contact": float(
            bool(info.get("grasp_contact", False))
        ),
        "grasp_streak": float(info.get("grasp_streak", 0)),
        "stable_grasp": float(
            bool(info.get("stable_grasp", False))
        ),
        "lifted": float(bool(info.get("lifted", False))),
        "peg_aligned_now": float(
            bool(info.get("peg_aligned_now", False))
        ),
        "hovering": float(bool(info.get("hovering", False))),
        "hover_streak": float(info.get("hover_streak", 0)),
        "geometry_available": float(
            bool(info.get("geometry_available", False))
        ),
        "hover_xy_dist": float(
            info.get("hover_xy_dist", float("nan"))
        ),
        "hover_xy_threshold": float(
            info.get("hover_xy_threshold", float("nan"))
        ),
        "nut_height_above_table": float(
            info.get("nut_height_above_table", float("nan"))
        ),
    }
    return record


def _consume_parallel_chunk(
    state,
    policy_action_chunk,
    env_action_chunk,
    result,
):
    chunk_obs = deepcopy(state["obs"])
    chunk_start_step = int(state["timestep"])
    previous_max_stage_id = int(state["highest_stage_id"])
    transitions = result.get("transitions", [])

    for local_index, transition in enumerate(transitions):
        global_step = int(state["timestep"])
        reward = float(transition["reward"])
        step_success = bool(transition["success"])
        done = bool(transition["done"])
        info = dict(transition["info"])

        current_stage_id = int(info.get("current_stage_id", 0))
        max_stage_id = int(
            info.get(
                "max_stage_id",
                max(previous_max_stage_id, current_stage_id),
            )
        )
        new_stage_ids = _extract_new_stage_ids(
            info=info,
            previous_max_stage_id=previous_max_stage_id,
        )

        for stage_id in new_stage_ids:
            stage_name = STAGE_NAMES.get(stage_id)
            if (
                stage_name in state["stage_achievement_steps"]
                and state["stage_achievement_steps"][stage_name] is None
            ):
                state["stage_achievement_steps"][stage_name] = global_step

        if (
            step_success
            and state["stage_achievement_steps"]["success"] is None
        ):
            state["stage_achievement_steps"]["success"] = global_step
            max_stage_id = max(max_stage_id, 4)

        state["highest_stage_id"] = max(
            int(state["highest_stage_id"]),
            max_stage_id,
        )
        previous_max_stage_id = max(
            previous_max_stage_id,
            max_stage_id,
        )

        policy_action_np = np.asarray(
            policy_action_chunk[local_index],
            dtype=np.float32,
        ).copy()
        env_action_np = np.asarray(
            env_action_chunk[local_index],
            dtype=np.float32,
        ).copy()

        state["steps"].append({
            "reward": reward,
            "done": done,
            "success": step_success,
            "staged": _parallel_staged_record(
                info=info,
                reward=reward,
                step_success=step_success,
                current_stage_id=current_stage_id,
                max_stage_id=max_stage_id,
                new_stage_ids=new_stage_ids,
            ),
            "action": policy_action_np,
            "env_action": env_action_np,
        })
        state["policy_actions"].append(policy_action_np)
        state["timestep"] += 1
        state["success"] = bool(
            state["success"] or step_success
        )

    executed_count = len(transitions)
    if executed_count > 0:
        state["chunk_records"].append({
            "obs": deepcopy(chunk_obs),
            "start_step": int(chunk_start_step),
            "executed_chunk_end_step": int(
                chunk_start_step + executed_count
            ),
            "done": bool(result.get("done", False)),
        })

    state["obs"] = result.get("obs", state["obs"])
    state["done"] = bool(
        result.get("done", False) or executed_count == 0
    )


def collect_rollout_episodes_parallel(
    env_pool,
    rollout_policy,
    model,
    num_episodes,
    horizon,
    terminate_on_success,
    progress_desc="Rollout",
):
    """
    Collect rollout waves using:
      1. one spawned robosuite process per environment;
      2. one central batched Flow inference call for all active workers.
    """
    observation_horizon = int(
        model.algo_config.horizon.observation_horizon
    )
    action_horizon = int(
        model.algo_config.horizon.action_horizon
    )
    prediction_horizon = int(
        model.algo_config.horizon.prediction_horizon
    )

    num_episodes = int(num_episodes)
    if num_episodes <= 0:
        return []

    episodes = []
    next_episode_id = 0
    model.reset()

    inference_rounds = 0
    policy_batch_sizes = []

    with tqdm(
        total=num_episodes,
        desc=str(progress_desc),
        dynamic_ncols=True,
        leave=True,
        unit="episode",
    ) as rollout_progress:
        while next_episode_id < num_episodes:
            wave_size = min(
                env_pool.num_workers,
                num_episodes - next_episode_id,
            )
            worker_ids = list(range(wave_size))
            initial_obs = env_pool.reset(worker_ids)

            states = [
                _new_parallel_rollout_state(
                    worker_id=worker_id,
                    episode_id=next_episode_id + local_index,
                    obs=initial_obs[worker_id],
                )
                for local_index, worker_id in enumerate(worker_ids)
            ]
            next_episode_id += wave_size

            while any(not state["done"] for state in states):
                active_states = [
                    state for state in states
                    if not state["done"]
                ]
                active_batch_size = len(active_states)
                policy_batch_sizes.append(active_batch_size)
                inference_rounds += 1

                observations = [
                    state["obs"] for state in active_states
                ]

                policy_chunks, env_chunks = sample_action_chunks_batched(
                    rollout_policy=rollout_policy,
                    model=model,
                    observations=observations,
                )

                active_worker_ids = [
                    state["worker_id"] for state in active_states
                ]
                max_steps = [
                    int(horizon) - int(state["timestep"])
                    for state in active_states
                ]

                results = env_pool.step_chunks(
                    worker_ids=active_worker_ids,
                    env_action_chunks=env_chunks,
                    max_steps=max_steps,
                    terminate_on_success=terminate_on_success,
                )

                newly_completed = 0
                for state, policy_chunk, env_chunk in zip(
                    active_states,
                    policy_chunks,
                    env_chunks,
                ):
                    was_done = bool(state["done"])
                    _consume_parallel_chunk(
                        state=state,
                        policy_action_chunk=policy_chunk,
                        env_action_chunk=env_chunk,
                        result=results[state["worker_id"]],
                    )
                    if (not was_done) and bool(state["done"]):
                        newly_completed += 1

                if newly_completed > 0:
                    rollout_progress.update(newly_completed)

                mean_batch = float(np.mean(policy_batch_sizes))
                rollout_progress.set_postfix(
                    active_batch=active_batch_size,
                    mean_batch=f"{mean_batch:.2f}",
                    infer_rounds=inference_rounds,
                    refresh=False,
                )

            episodes.extend([
                _finalize_rollout_state(
                    state=state,
                    observation_horizon=observation_horizon,
                    prediction_horizon=prediction_horizon,
                )
                for state in states
            ])

    episodes.sort(key=lambda episode: episode["episode_id"])

    if policy_batch_sizes:
        parallel_stats = {
            "workers": int(env_pool.num_workers),
            "episodes": int(num_episodes),
            "inference_rounds": int(inference_rounds),
            "max_policy_batch": int(max(policy_batch_sizes)),
            "min_policy_batch": int(min(policy_batch_sizes)),
            "mean_policy_batch": float(np.mean(policy_batch_sizes)),
            "batch_utilization": float(
                np.mean(policy_batch_sizes)
                / max(float(env_pool.num_workers), 1.0)
            ),
        }
    else:
        parallel_stats = {
            "workers": int(env_pool.num_workers),
            "episodes": int(num_episodes),
            "inference_rounds": 0,
            "max_policy_batch": 0,
            "min_policy_batch": 0,
            "mean_policy_batch": 0.0,
            "batch_utilization": 0.0,
        }

    print("\n[ParallelRollout]")
    print(json.dumps(parallel_stats, sort_keys=True, indent=4))
    return episodes



def build_model(config, shape_meta, device, ckpt_dict):
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.deserialize(ckpt_dict["model"])
    return model



def collect_rollout_episode(
    env,
    rollout_policy,
    model,
    episode_id,
    horizon,
    terminate_on_success,
):
    """
    Collect one closed-loop rollout and reconstruct prediction-horizon training
    windows from the actions that were actually executed.

    The policy still replans every action_horizon steps. After the episode is
    complete, each chunk observation is paired with a prediction_horizon action
    target aligned exactly like the offline frame-stacked SequenceDataset.
    """
    observation_horizon = int(model.algo_config.horizon.observation_horizon)
    action_horizon = int(model.algo_config.horizon.action_horizon)
    prediction_horizon = int(model.algo_config.horizon.prediction_horizon)

    if observation_horizon < 1:
        raise ValueError("observation_horizon must be positive")
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    if prediction_horizon < observation_horizon - 1 + action_horizon:
        raise ValueError(
            "prediction_horizon is too short for the executable window: "
            f"To={observation_horizon}, Ta={action_horizon}, "
            f"Tp={prediction_horizon}"
        )

    obs = env.reset()
    model.reset()

    steps = []
    chunk_records = []
    policy_actions = []

    stage_achievement_steps = {
        "stable_grasp": None,
        "lift": None,
        "hover": None,
        "success": None,
    }
    highest_stage_id = 0
    timestep = 0
    success = False

    while timestep < horizon:
        chunk_obs = deepcopy(obs)
        chunk_start_step = timestep

        policy_action_chunk, env_action_chunk = sample_action_chunk(
            rollout_policy=rollout_policy,
            model=model,
            obs=chunk_obs,
        )

        if policy_action_chunk.shape[0] != env_action_chunk.shape[0]:
            raise RuntimeError(
                "policy-space and environment-space action chunks have "
                "different lengths"
            )
        if policy_action_chunk.shape[0] > action_horizon:
            raise RuntimeError(
                "sampled action chunk is longer than configured action_horizon: "
                f"{policy_action_chunk.shape[0]} > {action_horizon}"
            )

        done = False
        previous_max_stage_id = highest_stage_id
        executed_count = 0

        for policy_action, env_action in zip(
            policy_action_chunk,
            env_action_chunk,
        ):
            global_step = timestep
            next_obs, reward, env_done, info = env.step(env_action)
            info = dict(info)

            step_success = bool(info.get("success", False))
            done = bool(
                env_done
                or (terminate_on_success and step_success)
                or ((timestep + 1) >= horizon)
            )

            current_stage_id = int(info.get("current_stage_id", 0))
            max_stage_id = int(
                info.get(
                    "max_stage_id",
                    max(previous_max_stage_id, current_stage_id),
                )
            )
            new_stage_ids = _extract_new_stage_ids(
                info=info,
                previous_max_stage_id=previous_max_stage_id,
            )
            for stage_id in new_stage_ids:
                stage_name = STAGE_NAMES.get(stage_id)
                if (
                    stage_name in stage_achievement_steps
                    and stage_achievement_steps[stage_name] is None
                ):
                    stage_achievement_steps[stage_name] = global_step

            # Success remains a hard fallback in case an older wrapper does not
            # emit stage 4 metadata.
            if step_success and stage_achievement_steps["success"] is None:
                stage_achievement_steps["success"] = global_step
                max_stage_id = max(max_stage_id, 4)

            highest_stage_id = max(highest_stage_id, max_stage_id)
            previous_max_stage_id = max(
                previous_max_stage_id,
                max_stage_id,
            )

            newly_achieved_stage_ids = tuple(
                int(stage_id) for stage_id in new_stage_ids
            )
            staged_info = {
                # Raw robosuite staged-state scores.
                "r_reach": float(info.get("r_reach", 0.0)),
                "r_grasp": float(info.get("r_grasp", 0.0)),
                "r_lift": float(info.get("r_lift", 0.0)),
                "r_hover": float(info.get("r_hover", 0.0)),

                # One-time transition rewards.
                "r_grasp_transition": float(
                    info.get("r_grasp_transition", 0.0)
                ),
                "r_lift_transition": float(
                    info.get("r_lift_transition", 0.0)
                ),
                "r_hover_transition": float(
                    info.get("r_hover_transition", 0.0)
                ),
                "r_success": float(info.get("r_success", 0.0)),
                "reward_progress": float(
                    info.get("reward_progress", reward)
                ),
                "stage_transition_reward": float(
                    info.get("stage_transition_reward", reward)
                ),
                "reward_total": float(
                    info.get("reward_total", reward)
                ),
                "base_reward": float(
                    info.get("base_reward", reward)
                ),

                # Physical and historical stage state.
                "success": float(step_success),
                "current_stage_id": float(current_stage_id),
                "max_stage_id": float(max_stage_id),
                "new_stage": float(bool(info.get("new_stage", False))),
                "new_stage_id": float(
                    info.get("new_stage_id", max_stage_id)
                ),
                "newly_achieved_stage_ids": newly_achieved_stage_ids,
                "grasp_contact": float(
                    bool(info.get("grasp_contact", False))
                ),
                "grasp_streak": float(info.get("grasp_streak", 0)),
                "stable_grasp": float(
                    bool(info.get("stable_grasp", False))
                ),
                "lifted": float(bool(info.get("lifted", False))),
                "hovering": float(bool(info.get("hovering", False))),
            }

            policy_action_np = np.asarray(
                policy_action,
                dtype=np.float32,
            ).copy()
            env_action_np = np.asarray(
                env_action,
                dtype=np.float32,
            ).copy()

            steps.append({
                "reward": float(reward),
                "done": bool(done),
                "success": bool(step_success),
                "staged": staged_info,
                "action": policy_action_np,
                "env_action": env_action_np,
            })
            policy_actions.append(policy_action_np)

            obs = next_obs
            timestep += 1
            executed_count += 1
            success = success or step_success

            if done:
                break

        if executed_count == 0:
            break

        chunk_records.append({
            "obs": deepcopy(chunk_obs),
            "start_step": int(chunk_start_step),
            "executed_chunk_end_step": int(
                chunk_start_step + executed_count
            ),
            "done": bool(done),
        })

        if done:
            break

    segments = _build_prediction_segments(
        steps=steps,
        chunk_records=chunk_records,
        policy_actions=policy_actions,
        episode_id=episode_id,
        observation_horizon=observation_horizon,
        prediction_horizon=prediction_horizon,
        stage_achievement_steps=stage_achievement_steps,
        highest_stage_id=highest_stage_id,
    )

    return {
        "episode_id": int(episode_id),
        "steps": steps,
        "segments": segments,
        "success": bool(success),
        "highest_stage_id": int(highest_stage_id),
        "highest_stage_name": STAGE_NAMES.get(
            int(highest_stage_id),
            f"stage_{highest_stage_id}",
        ),
        "stage_achievement_steps": stage_achievement_steps,
        "num_steps": int(len(steps)),
        "num_chunks": int(len(chunk_records)),
    }



def _new_rollout_state(env, episode_id):
    return {
        "env": env,
        "episode_id": int(episode_id),
        "obs": env.reset(),
        "steps": [],
        "chunk_records": [],
        "policy_actions": [],
        "stage_achievement_steps": {
            "stable_grasp": None,
            "lift": None,
            "hover": None,
            "success": None,
        },
        "highest_stage_id": 0,
        "timestep": 0,
        "success": False,
        "done": False,
    }


def _execute_rollout_chunk(
    state,
    policy_action_chunk,
    env_action_chunk,
    horizon,
    terminate_on_success,
    action_horizon,
):
    if policy_action_chunk.shape[0] != env_action_chunk.shape[0]:
        raise RuntimeError(
            "policy-space and environment-space action chunks have "
            "different lengths"
        )
    if policy_action_chunk.shape[0] > action_horizon:
        raise RuntimeError(
            "sampled action chunk is longer than action_horizon"
        )

    chunk_obs = deepcopy(state["obs"])
    chunk_start_step = int(state["timestep"])
    previous_max_stage_id = int(state["highest_stage_id"])
    executed_count = 0
    done = False

    for policy_action, env_action in zip(
        policy_action_chunk,
        env_action_chunk,
    ):
        global_step = int(state["timestep"])
        next_obs, reward, env_done, info = state["env"].step(env_action)
        info = dict(info)

        step_success = bool(info.get("success", False))
        done = bool(
            env_done
            or (terminate_on_success and step_success)
            or ((state["timestep"] + 1) >= horizon)
        )

        current_stage_id = int(info.get("current_stage_id", 0))
        max_stage_id = int(
            info.get(
                "max_stage_id",
                max(previous_max_stage_id, current_stage_id),
            )
        )
        new_stage_ids = _extract_new_stage_ids(
            info=info,
            previous_max_stage_id=previous_max_stage_id,
        )
        for stage_id in new_stage_ids:
            stage_name = STAGE_NAMES.get(stage_id)
            if (
                stage_name in state["stage_achievement_steps"]
                and state["stage_achievement_steps"][stage_name] is None
            ):
                state["stage_achievement_steps"][stage_name] = global_step

        if (
            step_success
            and state["stage_achievement_steps"]["success"] is None
        ):
            state["stage_achievement_steps"]["success"] = global_step
            max_stage_id = max(max_stage_id, 4)

        state["highest_stage_id"] = max(
            int(state["highest_stage_id"]),
            max_stage_id,
        )
        previous_max_stage_id = max(
            previous_max_stage_id,
            max_stage_id,
        )

        newly_achieved_stage_ids = tuple(
            int(stage_id) for stage_id in new_stage_ids
        )
        staged_info = {
            "r_reach": float(info.get("r_reach", 0.0)),
            "r_grasp": float(info.get("r_grasp", 0.0)),
            "r_lift": float(info.get("r_lift", 0.0)),
            "r_hover": float(info.get("r_hover", 0.0)),
            "r_grasp_transition": float(
                info.get("r_grasp_transition", 0.0)
            ),
            "r_lift_transition": float(
                info.get("r_lift_transition", 0.0)
            ),
            "r_hover_transition": float(
                info.get("r_hover_transition", 0.0)
            ),
            "r_success": float(info.get("r_success", 0.0)),
            "reward_progress": float(
                info.get("reward_progress", reward)
            ),
            "stage_transition_reward": float(
                info.get("stage_transition_reward", reward)
            ),
            "reward_total": float(info.get("reward_total", reward)),
            "base_reward": float(info.get("base_reward", reward)),
            "success": float(step_success),
            "current_stage_id": float(current_stage_id),
            "max_stage_id": float(max_stage_id),
            "new_stage": float(bool(info.get("new_stage", False))),
            "new_stage_id": float(
                info.get("new_stage_id", max_stage_id)
            ),
            "newly_achieved_stage_ids": newly_achieved_stage_ids,
            "grasp_contact": float(
                bool(info.get("grasp_contact", False))
            ),
            "grasp_streak": float(info.get("grasp_streak", 0)),
            "stable_grasp": float(
                bool(info.get("stable_grasp", False))
            ),
            "lifted": float(bool(info.get("lifted", False))),
            "hovering": float(bool(info.get("hovering", False))),
        }

        policy_action_np = np.asarray(
            policy_action,
            dtype=np.float32,
        ).copy()
        env_action_np = np.asarray(
            env_action,
            dtype=np.float32,
        ).copy()

        state["steps"].append({
            "reward": float(reward),
            "done": bool(done),
            "success": bool(step_success),
            "staged": staged_info,
            "action": policy_action_np,
            "env_action": env_action_np,
        })
        state["policy_actions"].append(policy_action_np)
        state["obs"] = next_obs
        state["timestep"] += 1
        executed_count += 1
        state["success"] = bool(state["success"] or step_success)

        if done:
            break

    if executed_count > 0:
        state["chunk_records"].append({
            "obs": deepcopy(chunk_obs),
            "start_step": int(chunk_start_step),
            "executed_chunk_end_step": int(
                chunk_start_step + executed_count
            ),
            "done": bool(done),
        })
    state["done"] = bool(done or executed_count == 0)


def _finalize_rollout_state(
    state,
    observation_horizon,
    prediction_horizon,
):
    highest_stage_id = int(state["highest_stage_id"])
    segments = _build_prediction_segments(
        steps=state["steps"],
        chunk_records=state["chunk_records"],
        policy_actions=state["policy_actions"],
        episode_id=state["episode_id"],
        observation_horizon=observation_horizon,
        prediction_horizon=prediction_horizon,
        stage_achievement_steps=state["stage_achievement_steps"],
        highest_stage_id=highest_stage_id,
    )
    return {
        "episode_id": int(state["episode_id"]),
        "steps": state["steps"],
        "segments": segments,
        "success": bool(state["success"]),
        "highest_stage_id": highest_stage_id,
        "highest_stage_name": STAGE_NAMES.get(
            highest_stage_id,
            f"stage_{highest_stage_id}",
        ),
        "stage_achievement_steps": state["stage_achievement_steps"],
        "num_steps": int(len(state["steps"])),
        "num_chunks": int(len(state["chunk_records"])),
    }


def collect_rollout_episodes_batched(
    env_pool,
    rollout_policy,
    model,
    num_episodes,
    horizon,
    terminate_on_success,
):
    """
    Collect episodes in waves with batched Flow inference.

    This is a low-risk first level of rollout parallelism: environments remain
    independent, and policy inference for all active envs is batched. MuJoCo
    stepping is still performed per environment, avoiding multiprocessing / EGL
    synchronization hazards.
    """
    observation_horizon = int(
        model.algo_config.horizon.observation_horizon
    )
    action_horizon = int(model.algo_config.horizon.action_horizon)
    prediction_horizon = int(
        model.algo_config.horizon.prediction_horizon
    )

    if len(env_pool) == 0:
        raise ValueError("env_pool must contain at least one env")

    episodes = []
    next_episode_id = 0
    model.reset()

    while next_episode_id < int(num_episodes):
        wave_size = min(
            len(env_pool),
            int(num_episodes) - next_episode_id,
        )
        states = [
            _new_rollout_state(
                env=env_pool[index],
                episode_id=next_episode_id + index,
            )
            for index in range(wave_size)
        ]
        next_episode_id += wave_size

        while any(not state["done"] for state in states):
            active_states = [
                state for state in states
                if not state["done"]
            ]
            observations = [
                state["obs"] for state in active_states
            ]
            policy_chunks, env_chunks = sample_action_chunks_batched(
                rollout_policy=rollout_policy,
                model=model,
                observations=observations,
            )

            for state, policy_chunk, env_chunk in zip(
                active_states,
                policy_chunks,
                env_chunks,
            ):
                _execute_rollout_chunk(
                    state=state,
                    policy_action_chunk=policy_chunk,
                    env_action_chunk=env_chunk,
                    horizon=int(horizon),
                    terminate_on_success=bool(
                        terminate_on_success
                    ),
                    action_horizon=action_horizon,
                )

        episodes.extend([
            _finalize_rollout_state(
                state=state,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon,
            )
            for state in states
        ])

    episodes.sort(key=lambda episode: episode["episode_id"])
    return episodes


def get_data_loaders(config, shape_meta):
    trainset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    obs_norm_stats = trainset.get_obs_normalization_stats() if config.train.hdf5_normalize_obs else None
    action_norm_stats = trainset.get_action_normalization_stats()
    return trainset, obs_norm_stats, action_norm_stats


def make_demo_loader(trainset, batch_size, num_workers):
    if batch_size <= 0:
        return None
    sampler = trainset.get_dataset_sampler()
    return DataLoader(
        dataset=trainset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        drop_last=True,
    )


def train(config, device):
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(2)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print("\n============= New Flow RWR Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config, resume=False)
    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    ObsUtils.initialize_obs_utils_with_config(config)

    if isinstance(config.train.data, str):
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]

    dataset_cfg = config.train.data[0]
    dataset_path = os.path.expanduser(dataset_cfg["path"])
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env_meta["lang"] = None
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=dataset_cfg,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=True,
    )

    ckpt_path = config.train.online.checkpoint_path or config.experiment.ckpt_path
    if ckpt_path is None:
        raise ValueError("must provide a pretrained flow checkpoint via train.online.checkpoint_path or experiment.ckpt_path")
    ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=ckpt_path)

    trainset, obs_normalization_stats, action_normalization_stats = get_data_loaders(config, shape_meta)

    # Mirror the offline train.py behavior so optimizer / scheduler construction
    # gets concrete step counts instead of empty Config placeholders.
    with config.values_unlocked():
        if "optim_params" in config.algo:
            for key in config.algo.optim_params:
                config.algo.optim_params[key]["num_train_batches"] = int(config.train.online.num_train_steps_per_iter)
                config.algo.optim_params[key]["num_epochs"] = int(config.train.online.num_iters)

    model = build_model(config, shape_meta, device, ckpt_dict)

    demo_batch_ratio = float(config.train.online.demo_batch_ratio)
    total_batch_size = int(config.train.batch_size)
    demo_batch_size = int(round(total_batch_size * demo_batch_ratio))
    demo_batch_size = max(0, min(total_batch_size, demo_batch_size))
    online_batch_size = max(0, total_batch_size - demo_batch_size)
    demo_loader = make_demo_loader(trainset, demo_batch_size, config.train.num_data_workers) if demo_batch_size > 0 else None
    demo_iter = iter(demo_loader) if demo_loader is not None else None

    train_env_name = (
        config.train.online.env_name
        if config.train.online.env_name is not None
        else env_meta["env_name"]
    )
    num_rollout_workers = int(
        _config_get(
            config.train.online,
            "num_rollout_workers",
            _config_get(
                config.train.online,
                "num_rollout_envs",
                8,
            ),
        )
    )
    parallel_env_pool = SubprocessEnvPool(
        config=config,
        env_meta=env_meta,
        shape_meta=shape_meta,
        num_workers=num_rollout_workers,
    )
    atexit.register(parallel_env_pool.close)

    rollout_envs = OrderedDict()
    print(
        "[rollout pool]",
        "num_workers=", parallel_env_pool.num_workers,
        "mode=subprocess_envs+central_batched_flow_inference",
    )

    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )

    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)
    print("")
    print("*" * 50)
    flush_warnings()
    print("*" * 50)
    print("")

    best_success_rate = -1.0
    best_return = -np.inf
    rollout_policy = RolloutPolicy(
        model,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats,
    )

    num_iters = int(config.train.online.num_iters)
    variable_state = {"iter": 0, "best_success_rate": best_success_rate, "best_return": best_return}
    for iter_idx in range(1, num_iters + 1):
        iter_wall_start = time.perf_counter()
        collection_start = time.perf_counter()

        model.set_eval()
        stage_priorities, curriculum_progress = (
            compute_stage_curriculum(
                config=config,
                iter_idx=iter_idx,
                num_iters=num_iters,
            )
        )

        buffer = FlowRolloutBuffer(
            action_horizon=model.algo_config.horizon.prediction_horizon,
            topk_fraction=float(config.algo.rwr.topk_fraction),
            use_segment_level_weighting=bool(
                config.algo.rwr.use_segment_level_weighting
            ),
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)

        collected_episodes = collect_rollout_episodes_parallel(
            env_pool=parallel_env_pool,
            rollout_policy=rollout_policy,
            model=model,
            num_episodes=int(
                config.train.online.num_rollout_episodes_per_iter
            ),
            horizon=int(config.train.online.rollout_horizon),
            terminate_on_success=bool(
                config.train.online.terminate_on_success
            ),
            progress_desc="Epoch {} rollout".format(iter_idx),
        )
        for episode in collected_episodes:
            buffer.add_episode(episode)

        collection_time_sec = time.perf_counter() - collection_start
        buffer_processing_start = time.perf_counter()

        buffer.compute_returns(
            gamma=float(config.algo.rwr.gamma),
            stage_priorities=stage_priorities,
        )
        buffer.normalize_advantages(
            eps=float(config.algo.rwr.advantage_eps),
            group_by_stage=bool(
                _config_get(
                    config.algo.rwr,
                    "group_advantages_by_stage",
                    True,
                )
            ),
        )
        buffer.compute_weights(
            temperature=float(config.algo.rwr.reward_temperature),
            min_weight=float(config.algo.rwr.min_weight),
            max_weight=float(config.algo.rwr.max_weight),
            topk_fraction=float(config.algo.rwr.topk_fraction),
            stage_priorities=stage_priorities,
        )
        rollout_stats = buffer.get_stats()
        rollout_stats["curriculum_progress"] = float(
            curriculum_progress
        )

        buffer_processing_time_sec = (
            time.perf_counter() - buffer_processing_start
        )

        _print_stage_interval_debug(
            buffer=buffer,
            iter_idx=iter_idx,
            max_segments=12,
        )

        train_start = time.perf_counter()
        model.set_train()
        train_logs = []

        num_online_epochs = int(
            _config_get(
                config.train.online,
                "num_online_epochs_per_iter",
                2,
            )
        )
        online_updates = buffer.num_batches(
            batch_size=online_batch_size,
            num_epochs=num_online_epochs,
        ) if online_batch_size > 0 else 0

        def next_demo_batch():
            nonlocal demo_iter
            if demo_loader is None:
                return None
            try:
                raw = next(demo_iter)
            except StopIteration:
                demo_iter = iter(demo_loader)
                raw = next(demo_iter)
            processed = model.process_batch_for_training(raw)
            return model.postprocess_batch_for_training(
                processed,
                obs_normalization_stats=obs_normalization_stats,
            )

        consumed_online_samples = 0
        actual_train_updates = 0

        if online_updates > 0:
            online_iterator = buffer.iter_batches(
                batch_size=online_batch_size,
                num_epochs=num_online_epochs,
                shuffle=True,
                seed=int(config.train.seed) + 100000 * iter_idx,
                only_positive_weights=True,
            )
            train_progress = tqdm(
                online_iterator,
                total=int(online_updates),
                desc="Epoch {} train".format(iter_idx),
                dynamic_ncols=True,
                leave=True,
                unit="update",
            )
            for online_batch_raw in train_progress:
                expected_horizon = int(
                    model.algo_config.horizon.prediction_horizon
                )
                if (
                    online_batch_raw["actions"].shape[1]
                    != expected_horizon
                ):
                    raise RuntimeError(
                        "online action target must use "
                        "prediction_horizon"
                    )
                if (
                    online_batch_raw["action_mask"].shape[1]
                    != expected_horizon
                ):
                    raise RuntimeError(
                        "online action mask must use "
                        "prediction_horizon"
                    )

                demo_batch = next_demo_batch()
                online_batch = model.process_online_batch_for_training(
                    online_batch_raw
                )
                online_batch = model.postprocess_batch_for_training(
                    online_batch,
                    obs_normalization_stats=obs_normalization_stats,
                )

                info = model.train_on_mixed_batch(
                    demo_batch=demo_batch,
                    online_batch=online_batch,
                    epoch=iter_idx,
                    validate=False,
                )
                model.on_gradient_step()
                step_log = model.log_info(info)
                train_logs.append(step_log)
                progress_values = {}
                for key, label in (
                    ("Loss", "loss"),
                    ("Demo_Loss", "demo"),
                    ("Online_Loss", "online"),
                ):
                    if key in step_log:
                        progress_values[label] = (
                            f"{float(step_log[key]):.4f}"
                        )
                if progress_values:
                    train_progress.set_postfix(
                        progress_values,
                        refresh=False,
                    )
                consumed_online_samples += int(
                    online_batch_raw["actions"].shape[0]
                )
                actual_train_updates += 1
        else:
            demo_only_steps = int(
                _config_get(
                    config.train.online,
                    "demo_only_steps_when_no_online",
                    1,
                )
            )
            demo_progress = tqdm(
                range(max(demo_only_steps, 0)),
                total=max(demo_only_steps, 0),
                desc="Epoch {} demo-only train".format(iter_idx),
                dynamic_ncols=True,
                leave=True,
                unit="update",
            )
            for _ in demo_progress:
                demo_batch = next_demo_batch()
                if demo_batch is None:
                    break
                info = model.train_on_mixed_batch(
                    demo_batch=demo_batch,
                    online_batch=None,
                    epoch=iter_idx,
                    validate=False,
                )
                model.on_gradient_step()
                train_logs.append(model.log_info(info))
                actual_train_updates += 1

        rollout_stats["num_online_epochs"] = int(
            num_online_epochs
        )
        rollout_stats["planned_online_updates"] = int(
            online_updates
        )
        rollout_stats["actual_train_updates"] = int(
            actual_train_updates
        )
        rollout_stats["consumed_online_samples"] = int(
            consumed_online_samples
        )
        rollout_stats["expected_online_sample_uses"] = int(
            len([
                seg for seg in buffer.segments
                if float(seg.get("weight", 0.0)) > 0.0
            ]) * max(num_online_epochs, 0)
        )

        train_time_sec = time.perf_counter() - train_start

        _print_epoch_stage_report(
            rollout_stats=rollout_stats,
            iter_idx=iter_idx,
        )

        mean_train_log = {}
        if len(train_logs) > 0:
            log_keys = train_logs[0].keys()
            for key in log_keys:
                mean_train_log[key] = float(np.mean([log[key] for log in train_logs]))

        print("Iter {}".format(iter_idx))
        print(json.dumps({
            "rollout": rollout_stats,
            "train": mean_train_log,
        }, sort_keys=True, indent=4))

        for key, value in rollout_stats.items():
            data_logger.record("Rollout/{}".format(key), value, iter_idx)
        for key, value in mean_train_log.items():
            data_logger.record("Train/{}".format(key), value, iter_idx)

        eval_time_sec = 0.0
        eval_interval = int(config.train.online.eval_interval)
        if eval_interval > 0:
            raise ValueError(
                "This formal trainer is configured for no in-training "
                "evaluation. Set train.online.eval_interval to 0 and "
                "evaluate saved checkpoints separately."
            )

        variable_state = {
            "iter": iter_idx,
            "best_success_rate": best_success_rate,
            "best_return": best_return,
        }

        iter_wall_time_sec = time.perf_counter() - iter_wall_start
        timing_stats = {
            "collection_time_sec": float(collection_time_sec),
            "collection_time_per_episode_equivalent_sec": float(
                collection_time_sec
                / max(
                    int(
                        config.train.online
                        .num_rollout_episodes_per_iter
                    ),
                    1,
                )
            ),
            "buffer_processing_time_sec": float(
                buffer_processing_time_sec
            ),
            "train_time_sec": float(train_time_sec),
            "eval_time_sec": float(eval_time_sec),
            "iter_wall_time_before_save_sec": float(
                iter_wall_time_sec
            ),
        }
        if torch.cuda.is_available():
            timing_stats["gpu_peak_allocated_gb"] = float(
                torch.cuda.max_memory_allocated(device=device)
                / (1024 ** 3)
            )
            timing_stats["gpu_peak_reserved_gb"] = float(
                torch.cuda.max_memory_reserved(device=device)
                / (1024 ** 3)
            )
        print("[Timing]")
        print(json.dumps(timing_stats, sort_keys=True, indent=4))
        for key, value in timing_stats.items():
            data_logger.record("Timing/{}".format(key), value, iter_idx)

        save_interval = int(config.train.online.save_interval)
        if save_interval > 0 and (iter_idx % save_interval == 0):
            checkpoint_path = os.path.join(
                ckpt_dir,
                "model_iter_{}.pth".format(iter_idx),
            )
            print(
                "\nsaving periodic checkpoint at {}...\n".format(
                    checkpoint_path
                )
            )
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                variable_state=variable_state,
                ckpt_path=checkpoint_path,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, iter_idx)
        print("\nIter {} Memory Usage: {} MB\n".format(iter_idx, mem_usage))

    final_checkpoint_path = os.path.join(
        ckpt_dir,
        "model_final_iter_{}.pth".format(num_iters),
    )
    print(
        "\nsaving final checkpoint at {}...\n".format(
            final_checkpoint_path
        )
    )
    TrainUtils.save_model(
        model=model,
        config=config,
        env_meta=env_meta,
        shape_meta=shape_meta,
        variable_state=variable_state,
        ckpt_path=final_checkpoint_path,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats,
    )

    parallel_env_pool.close()
    data_logger.close()


def main(args):
    if args.config is not None:
        ext_cfg = json.load(open(args.config, "r"))
        config = config_factory(ext_cfg["algo_name"])
        # The curriculum / batched-rollout options are new keys that older
        # FlowRWRConfig classes may not declare. Fully unlock during update,
        # then lock again below.
        config.unlock()
        config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = [{"path": args.dataset}]
    if args.name is not None:
        config.experiment.name = args.name
    if args.ckpt_path is not None:
        config.train.online.checkpoint_path = args.ckpt_path

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    if args.debug:
        config.unlock()
        config.lock_keys()
        config.train.online.num_iters = 2
        config.train.online.num_rollout_episodes_per_iter = 2
        config.train.online.num_train_steps_per_iter = 2
        config.train.online.num_rollout_envs = 1
        config.train.online.num_online_epochs_per_iter = 1
        config.train.online.rollout_horizon = 16
        config.train.online.eval_interval = 1
        config.train.online.save_interval = 1
        config.train.online.num_eval_episodes = 1
        config.train.output_dir = "/tmp/tmp_flow_rwr"

    config.lock()

    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--algo", type=str, default="flow_rwr")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
