"""Training, evaluation, metrics, and plotting utilities."""

import json
import platform
import random
from numbers import Integral
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from .agents import DEVICE, DDPGAgent, TD3Agent, PolicyNetwork
from .environment import AUVConstantDepthEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_FILENAMES = (
    "evaluation.json",
    "trajectories.npz",
    "depth_tracking.png",
)


def _validate_positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _prepare_output_directory(output_dir, artifact_filenames, overwrite):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    existing = [output_path / name for name in artifact_filenames if (output_path / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing artifacts in {output_path}: {names}"
        )
    return output_path


def default_run_directory(algorithm, seed):
    return PROJECT_ROOT / "runs" / f"{algorithm.lower()}_seed{seed}"


def set_seed(seed=12):
    """Seed Python, NumPy, and PyTorch for an experiment run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_training_rewards(rewards, algorithm):
    figure, axis = plt.subplots()
    axis.plot(rewards)
    axis.set_xlabel("Episode")
    axis.set_ylabel("Reward")
    axis.set_title(f"{algorithm.upper()} Training Rewards")
    return figure, axis


def plot_depth_trajectories(depths, dt, target_depth, label=None):
    figure, axis = plt.subplots()
    time = (np.arange(depths.shape[1]) + 1) * dt
    for episode, episode_depths in enumerate(depths, start=1):
        episode_label = label or f"Episode {episode}"
        if depths.shape[0] > 1 and label:
            episode_label = f"{label} {episode}"
        axis.plot(time, episode_depths, label=episode_label)
    axis.axhline(target_depth, color="k", linestyle="--", label="Target")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Depth (m)")
    axis.set_title("Depth vs Time")
    axis.legend()
    axis.grid(True)
    return figure, axis


def calculate_metrics(depths, thetas, actions, rewards, target_depth, dt):
    """Calculate descriptive metrics for evaluated controller rollouts."""
    final_depth_error = np.abs(depths[:, -1] - target_depth)
    overshoot = np.maximum(np.max(depths, axis=1) - target_depth, 0.0)
    action_effort = np.sum(actions**2, axis=(1, 2)) * dt
    action_variation = np.sum(np.abs(np.diff(actions, axis=1)), axis=(1, 2))
    return {
        "total_reward": rewards,
        "final_depth_error": final_depth_error,
        "maximum_depth_overshoot": overshoot,
        "maximum_absolute_pitch": np.max(np.abs(thetas), axis=1),
        "squared_action_effort": action_effort,
        "total_action_variation": action_variation,
    }


def train_agent(
    algorithm="ddpg",
    episodes=500,
    steps_per_episode=200,
    start_depth=2.0,
    target_depth=8.0,
    output_dir=None,
    plot=True,
    seed=12,
    overwrite=False,
):
    """Train DDPG or TD3 in fixed-horizon continuing-task rollouts.

    The environment has no terminal state or replay terminal mask; each episode
    is an orchestration-level truncation after ``steps_per_episode`` transitions.
    """
    _validate_positive_integer("episodes", episodes)
    _validate_positive_integer("steps_per_episode", steps_per_episode)
    algorithm = algorithm.lower()
    if algorithm == "td3":
        actor_filename = "td3_actor.pth"
        critic_filenames = ("td3_critic1.pth", "td3_critic2.pth")
    elif algorithm == "ddpg":
        actor_filename = "ddpg_actor.pth"
        critic_filenames = ("ddpg_critic.pth",)
    else:
        raise ValueError("algorithm must be 'ddpg' or 'td3'")

    rewards_filename = f"{algorithm}_rewards.npy"
    artifact_filenames = (
        actor_filename,
        *critic_filenames,
        rewards_filename,
        "run.json",
    )
    if output_dir is None:
        output_dir = default_run_directory(algorithm, seed)
    output_path = _prepare_output_directory(
        output_dir, artifact_filenames, overwrite=overwrite
    )

    set_seed(seed)
    env = AUVConstantDepthEnv()
    state_dim, action_dim = 5, 2
    max_action = env.max_tau

    if algorithm == "td3":
        agent = TD3Agent(state_dim, action_dim, max_action)
        exploration_noise_std = 0.1 * max_action
    else:
        agent = DDPGAgent(state_dim, action_dim, max_action)
        exploration_noise_std = 0.1
    batch_size = 64

    rewards = []
    for episode in range(1, episodes + 1):
        state = env.reset(zr=target_depth, z0=start_depth)
        episode_reward = 0.0

        for _ in range(steps_per_episode):
            if algorithm == "ddpg":
                action = agent.select_action(state, noise_scale=exploration_noise_std)
                action = np.clip(action, -max_action, max_action)
            else:
                action = agent.select_action(state)
                action = (
                    action
                    + np.random.normal(0, exploration_noise_std, size=action.shape)
                ).clip(-max_action, max_action)

            next_state, reward, _, _ = env.step(action)
            agent.replay.push(state, action, reward, next_state)
            agent.train(batch_size=batch_size)
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        print(
            f"{algorithm.upper()} Episode {episode}/{episodes}, "
            f"Reward: {episode_reward:.2f}"
        )

    torch.save(agent.actor.state_dict(), output_path / actor_filename)
    if algorithm == "ddpg":
        torch.save(agent.critic.state_dict(), output_path / critic_filenames[0])
    else:
        torch.save(agent.critic1.state_dict(), output_path / critic_filenames[0])
        torch.save(agent.critic2.state_dict(), output_path / critic_filenames[1])
    np.save(output_path / rewards_filename, np.array(rewards))

    hidden_sizes = [
        layer.out_features
        for layer in agent.actor.net
        if isinstance(layer, torch.nn.Linear)
    ][:-1]
    hyperparameters = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_sizes": hidden_sizes,
        "max_action": max_action,
        "gamma": agent.gamma,
        "tau": agent.tau,
        "batch_size": batch_size,
        "replay_capacity": agent.replay.buffer.maxlen,
        "actor_learning_rate": agent.actor_optimizer.param_groups[0]["lr"],
        "exploration_noise_std": exploration_noise_std,
    }
    if algorithm == "ddpg":
        hyperparameters["critic_learning_rate"] = agent.critic_optimizer.param_groups[0][
            "lr"
        ]
    else:
        hyperparameters.update(
            {
                "critic1_learning_rate": agent.critic1_optimizer.param_groups[0]["lr"],
                "critic2_learning_rate": agent.critic2_optimizer.param_groups[0]["lr"],
                "target_policy_noise_std": agent.policy_noise,
                "target_noise_clip": agent.noise_clip,
                "policy_delay": agent.policy_delay,
            }
        )

    manifest = {
        "algorithm": algorithm,
        "seed": seed,
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "start_depth": start_depth,
        "target_depth": target_depth,
        "device": str(DEVICE),
        "hyperparameters": hyperparameters,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": {
            "actor": actor_filename,
            "critics": list(critic_filenames),
            "rewards": rewards_filename,
            "manifest": "run.json",
        },
    }
    with (output_path / "run.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")

    if plot:
        plot_training_rewards(rewards, algorithm)
        plt.show()

    return agent.actor


def evaluate_controller(
    actor_path,
    episodes=1,
    steps_per_episode=2000,
    start_depth=2.0,
    target_depth=8.0,
    plot=True,
    label=None,
    output_dir=None,
    overwrite=False,
):
    """Evaluate a saved actor deterministically in the simplified AUV model.

    State trajectories contain the post-transition observation for each action,
    including the state produced by the final action.
    """
    _validate_positive_integer("episodes", episodes)
    _validate_positive_integer("steps_per_episode", steps_per_episode)
    output_path = None
    if output_dir is not None:
        output_path = _prepare_output_directory(
            output_dir, EVALUATION_FILENAMES, overwrite=overwrite
        )
    env = AUVConstantDepthEnv()
    actor = PolicyNetwork(5, 2, env.max_tau).to(DEVICE)
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE, weights_only=True))
    actor.eval()

    all_depths = np.zeros((episodes, steps_per_episode), dtype=np.float32)
    all_thetas = np.zeros((episodes, steps_per_episode), dtype=np.float32)
    all_ws = np.zeros((episodes, steps_per_episode), dtype=np.float32)
    all_qs = np.zeros((episodes, steps_per_episode), dtype=np.float32)
    all_actions = np.zeros((episodes, steps_per_episode, 2), dtype=np.float32)
    all_rewards = np.zeros(episodes, dtype=np.float32)

    with torch.no_grad():
        for episode in range(episodes):
            state = env.reset(zr=target_depth, z0=start_depth)
            total_reward = 0.0

            for step in range(steps_per_episode):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action = actor(state_tensor).cpu().numpy()[0]
                all_actions[episode, step] = action
                state, reward, _, _ = env.step(action)
                total_reward += reward

                dz, cos_theta, sin_theta, w, q = state
                all_depths[episode, step] = dz + target_depth
                all_thetas[episode, step] = np.arctan2(sin_theta, cos_theta)
                all_ws[episode, step] = w
                all_qs[episode, step] = q

            all_rewards[episode] = total_reward
            print(f"Episode {episode + 1} Total Reward: {total_reward:.2f}")

    figure = None
    if plot or output_path is not None:
        figure, _ = plot_depth_trajectories(
            all_depths, env.dt, target_depth, label=label
        )

    if output_path is not None:
        np.savez_compressed(
            output_path / "trajectories.npz",
            depth=all_depths,
            pitch=all_thetas,
            heave_velocity=all_ws,
            pitch_rate=all_qs,
            actions=all_actions,
            rewards=all_rewards,
        )
        metrics = calculate_metrics(
            all_depths,
            all_thetas,
            all_actions,
            all_rewards,
            target_depth,
            env.dt,
        )
        evaluation_record = {
            "metadata": {
                "controller_label": label or Path(actor_path).stem,
                "checkpoint_path": str(actor_path),
                "episodes": episodes,
                "steps_per_episode": steps_per_episode,
                "start_depth": start_depth,
                "target_depth": target_depth,
                "dt": env.dt,
                "device": str(DEVICE),
            },
            "metrics": {name: values.tolist() for name, values in metrics.items()},
        }
        with (output_path / "evaluation.json").open(
            "w", encoding="utf-8"
        ) as evaluation_file:
            json.dump(evaluation_record, evaluation_file, indent=2)
            evaluation_file.write("\n")
        figure.savefig(output_path / "depth_tracking.png", bbox_inches="tight")

    if plot:
        plt.show()
    elif figure is not None:
        plt.close(figure)

    return all_depths, all_thetas, all_ws, all_qs, all_actions, all_rewards
