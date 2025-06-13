"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.
"""

from collections import defaultdict
import os
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal, Optional, Union
import torch
import wandb

import draccus
import numpy as np
# import tqdm
from tqdm import tqdm, trange

import pandas as pd

from libero.libero import benchmark


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video_given_path
)
from experiments.robot.openvla_utils import (
    get_processor,
    get_text_tokens
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.viz_utils import (
    PoseCumulator
)
from experiments.robot.unc_utils import (
    compute_token_uncertainty_metrics,
    compute_samples_uncertainty_metrics,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    
    n_samples: int = 1                               # Number of samples to draw from the model for each action step
    attn_implementation: str = "flash_attention_2"   # Only eager attention supports return_attentions, spda will fall back to eager to support it. Options: "flash_attention_2", "sdpa", "eager"
    output_logits: bool = True                       # Whether to output logits from the model
    output_attentions: bool = False                  # Whether to output attention weights from the model
    output_hidden_states: bool = False               # Whether to output hidden states from the model

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    task_start_index: Optional[int] = None                     # Start task index (inclusive)
    task_end_index: Optional[int] = None                       # End task index (inclusive)
    resume: bool = False                            # Resume from a previous run

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = "debug"             # Extra note to add in run ID for logging
    save_root: str = "./rollouts"                    # Root directory to save rollouts

    use_wandb: bool = True                           # Whether to also log results in Weights & Biases
    wandb_project: str = "openvla"                   # Name of W&B project to log to (use default!)
    wandb_entity: str = "qiaog"                      # Name of entity to log under
    wandb_dir: Optional[str] = None                  # Directory to save W&B logs
    save_logs: bool = True                          # Whether to dump W&B logs to a file

    seed: int = 7                                    # Random Seed (for reproducibility)
    
    #################################################################################################################
    # Plotting and Logging
    #################################################################################################################
    attn_avg_token: bool = True                      # Plot the attention map averaged over tokens

    # fmt: on
    
    

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    
    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_group = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}"
    if cfg.run_id_note is not None:
        run_group += f"--{cfg.run_id_note}"

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    
    # Formulate the path to save outputs
    save_root = Path(cfg.save_root)
    save_folder = save_root / f"{cfg.run_id_note}" /  f"{cfg.task_suite_name}"
    save_folder.mkdir(parents=True, exist_ok=True)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_start_index = cfg.task_start_index if cfg.task_start_index is not None else 0
    task_end_index = cfg.task_end_index + 1 if cfg.task_end_index is not None else num_tasks_in_suite
    for task_id in trange(task_start_index, task_end_index):
        if cfg.resume:
            # Check for existing saved results. 
            existing_results = list(save_folder.glob(f"task{task_id}--ep*--succ*.csv"))
            # Get the maximum episode index existing
            max_existing_episode = max([int(str(p).split("--ep")[1].split("--")[0]) for p in existing_results]) if existing_results else -1
            if max_existing_episode + 1 >= cfg.num_trials_per_task:
                print(f"Task {task_id} already has {cfg.num_trials_per_task} episodes. Skipping...")
                continue
            start_episode = max_existing_episode + 1
        else:
            start_episode = 0
        
        print(f"Starting from episode {start_episode} for task {task_id}")
        
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in trange(start_episode, cfg.num_trials_per_task):
            print(f"\nTask: {task_description}")
            
            if cfg.use_wandb:
                wandb_config = asdict(cfg)
                wandb_config.update({
                    "base_task": task.name,
                    "task_desc": task_description,
                })
                wandb.init(
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    group = run_group,
                    name = f"task-{task_id}--episode-{episode_idx}",
                    dir = cfg.wandb_dir,
                    config = wandb_config,
                )

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            
            # Setup
            t = 0
            replay_images = []
            if "libero_spatial" in cfg.task_suite_name:
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
                

            # Experiment loggers
            logs = defaultdict(list)
            logs_cum = defaultdict(float)
            pose_cumulator = PoseCumulator()
            hidden_states_episode = []
            logs_to_dump = []
            
            while t < max_steps + cfg.num_steps_wait:
                excepted = False
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    
                    if cfg.use_wandb and t == cfg.num_steps_wait:
                        # Log the initial observation
                        wandb.log({"obs/init": wandb.Image(img)})

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    
                    # Query model to get action
                    actions = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        n_samples=cfg.n_samples,
                    )
                    
                    if type(actions) is tuple:
                        actions, generated_outputs = actions
                        
                    else:
                        generated_outputs = {} # empty dict
                        
                    if cfg.output_hidden_states:
                        # If output_hidden_states is True, "hidden_states" exists in generated_outputs
                        # generated_outputs['hidden_states'] is a tuple of length 7 (number of generated tokens)
                        # Within which each element is a tuple of length 33 (number of layers)
                        # Each element in the inner tuple is a tensor of shape (bs=1, 1 or N, 4096)
                        # The final hidden states before decoding is generated_outputs['hidden_states'][0][-1][0, -1, :]
                        all_hidden_states = generated_outputs['hidden_states']
                        hidden_states_last_layer = [s[-1][0, -1, :] for s in all_hidden_states]
                        hidden_states_last_layer = torch.stack(hidden_states_last_layer, dim=0) # (7, 4096)
                        
                        # Save the hidden states for further analysis
                        hidden_states_episode.append(hidden_states_last_layer.detach().cpu())
                        

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    actions = normalize_gripper_action(actions, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        actions = invert_gripper_action(actions)
                        
                    to_be_logged = {
                        "action/timestep": t,
                    }
                        
                    if actions.ndim > 1: # mulit-forward uncertainty metrics
                        metrics, cluster_labels = compute_samples_uncertainty_metrics(actions)
                        for k, v in metrics.items():
                            logs[k].append(v)
                            to_be_logged[f"action/{k}"] = v
                            to_be_logged[f"action/cum_{k}"] = np.sum(logs[k])

                        # Use the first sampled action as the action to execute
                        action = actions[0]

                    else: # single-forward uncertainty metrics
                        metrics = compute_token_uncertainty_metrics(generated_outputs, model)
                        
                        # Log the metrics
                        for k, v in metrics.items():
                            logs[k].append(v)
                            to_be_logged[f"action/{k}"] = v
                            to_be_logged[f"action/cum_{k}"] = np.sum(logs[k])

                        action = actions
                        
                        
                    # Log the action itself
                    dpos = np.linalg.norm(action[:3])
                    drot = np.linalg.norm(action[3:6])
                    logs_cum["dpos"] += dpos
                    logs_cum["drot"] += drot
                    to_be_logged.update({
                        "action/dx": action[0],
                        "action/dy": action[1],
                        "action/dz": action[2],
                        "action/droll": action[3],
                        "action/dpitch": action[4],
                        "action/dyaw": action[5],
                        "action/dgripper": action[6],
                        "action/dpos": dpos,
                        "action/drot": drot,
                        "action/cum_dpos": logs_cum["dpos"],
                        "action/cum_drot": logs_cum["drot"],
                    })
                    
                    # Track how much the end effector has moved
                    pose_cumulator.update(obs["robot0_eef_pos"], obs["robot0_eef_quat"])
                    to_be_logged.update({
                        "pose/cum_pos": pose_cumulator.cum_pos,
                        "pose/cum_rot": pose_cumulator.cum_rot,
                    })
                    
                    # Convert the numpy values to float
                    for k, v in to_be_logged.items():
                        if type(v) is np.float64:
                            to_be_logged[k] = float(v)
                        
                    if cfg.use_wandb:
                        wandb.log(to_be_logged)
                        
                    if cfg.save_logs:
                        logs_to_dump.append(to_be_logged)
                        
                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())

                    # get and check goal state
                    goal_state = env.env.parsed_problem["goal_state"]
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    # print(f"Caught exception: {e}")
                    # log_file.write(f"Caught exception: {e}\n")
                    # excepted = True
                    # break
                    raise e
                    
            if cfg.use_wandb:
                if excepted:
                    wandb.log({
                        "ep/excepted": float(excepted),
                        "ep/success": 0.0,
                    })
                else:
                    wandb.log({
                        "ep/excepted": float(excepted),
                        "ep/success": float(done),
                        "obs/final": wandb.Image(img),
                    })
                    
                for k, v in logs.items():
                    wandb.log({
                        f"action/mean_{k}": np.mean(v),
                    })
                
                wandb.finish(quiet=True)

            task_episodes += 1
            total_episodes += 1
            
            
            mp4_path = save_folder / f"task{task_id}--ep{episode_idx}--succ{int(done)}.mp4"
            save_rollout_video_given_path(replay_images, mp4_path)

            # Save the hidden states
            if cfg.output_hidden_states:
                hidden_states_episode = torch.stack(hidden_states_episode, dim=0) # (T, 7, 4096)
                hidden_states_episode = hidden_states_episode
                hidden_states_path = mp4_path.with_suffix(".pkl")
                save_dict = {
                    "hidden_states": hidden_states_episode,
                    "task_suite_name": cfg.task_suite_name,
                    "task_id": task_id,
                    "task_description": task_description,
                    "eposide_idx": episode_idx,
                    "episode_success": done,
                    "mp4_path": str(mp4_path),
                }
                pickle.dump(save_dict, open(hidden_states_path, "wb"))
                print(f"Saved hidden states at path {hidden_states_path}")
                
            if cfg.save_logs:
                logs_path = mp4_path.with_suffix(".csv")
                df = pd.DataFrame(logs_to_dump)
                df.to_csv(logs_path, index=False)

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        
    env.close()


if __name__ == "__main__":
    eval_libero()
