"""Small regression tests for the extracted implementation."""

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auv_control.agents import DDPGAgent, ReplayBuffer, TD3Agent  # noqa: E402
from auv_control.environment import AUVConstantDepthEnv  # noqa: E402
import auv_control.experiment as experiment  # noqa: E402
from auv_control.experiment import evaluate_controller, train_agent  # noqa: E402
from auv_control.lqi import CLOSED_LOOP_EIGENVALUES, simulate_lqi  # noqa: E402


class EnvironmentTests(unittest.TestCase):
    def test_default_construction_and_reset(self):
        env = AUVConstantDepthEnv()
        state = env.reset()

        np.testing.assert_array_equal(state, np.array([-6, 1, 0, 0, 0], np.float32))

    def test_custom_constructor_values_are_used_by_reset(self):
        env = AUVConstantDepthEnv(zr=5.0, z0=1.0)
        state = env.reset()

        np.testing.assert_array_equal(state, np.array([-4, 1, 0, 0, 0], np.float32))
        self.assertEqual(env.zr, 5.0)
        self.assertEqual(env.z0, 1.0)

    def test_explicit_reset_values_update_stored_defaults(self):
        env = AUVConstantDepthEnv()
        state = env.reset(zr=12.0, z0=3.0)

        np.testing.assert_array_equal(state, np.array([-9, 1, 0, 0, 0], np.float32))
        self.assertEqual(env.zr, 12.0)
        self.assertEqual(env.z0, 3.0)

    def test_no_argument_reset_reuses_explicit_reset_values(self):
        env = AUVConstantDepthEnv()
        env.reset(zr=12.0, z0=3.0)
        env.step(np.array([10.0, -5.0]))
        state = env.reset()

        np.testing.assert_array_equal(state, np.array([-9, 1, 0, 0, 0], np.float32))

    def test_zero_action_step(self):
        env = AUVConstantDepthEnv()
        state = env.reset()
        next_state, reward, done, info = env.step(np.zeros(2))

        np.testing.assert_array_equal(next_state, state)
        self.assertAlmostEqual(reward, -43.2)
        self.assertFalse(done)
        self.assertEqual(info, {})

    def test_action_is_clipped(self):
        first = AUVConstantDepthEnv()
        second = AUVConstantDepthEnv()
        first.reset()
        second.reset()
        clipped_state = first.step(np.array([100.0, -100.0]))[0]
        large_state = second.step(np.array([1000.0, -1000.0]))[0]
        np.testing.assert_allclose(large_state, clipped_state)


class AgentTests(unittest.TestCase):
    def test_actor_shapes_and_bounds(self):
        state = np.array([-6, 1, 0, 0, 0], dtype=np.float32)
        for agent in (DDPGAgent(5, 2, 100.0), TD3Agent(5, 2, 100.0)):
            action = agent.select_action(state)
            self.assertEqual(action.shape, (2,))
            self.assertTrue(np.all(np.abs(action) <= 100.0))

    def test_training_update_is_finite(self):
        agent = DDPGAgent(5, 2, 100.0)
        state = np.array([-6, 1, 0, 0, 0], dtype=np.float32)
        for _ in range(64):
            agent.replay.push(state, np.zeros(2), -43.2, state)
        agent.train()
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.actor.parameters()))


class ExperimentTests(unittest.TestCase):
    @staticmethod
    def _save_test_actor(checkpoint_path):
        actor = experiment.PolicyNetwork(5, 2, 100.0)
        torch.save(actor.state_dict(), checkpoint_path)

    @staticmethod
    def _save_constant_test_actor(checkpoint_path, action):
        actor = experiment.PolicyNetwork(5, 2, 100.0)
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            normalized_action = torch.tensor(action, dtype=torch.float32) / 100.0
            actor.net[-2].bias.copy_(torch.atanh(normalized_action))
        torch.save(actor.state_dict(), checkpoint_path)

    def test_ddpg_stores_the_clipped_exploratory_action(self):
        transitions = []
        original_push = ReplayBuffer.push

        def capture_push(buffer, state, action, reward, next_state):
            transitions.append((state, action.copy(), reward, next_state))
            original_push(buffer, state, action, reward, next_state)

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(
                DDPGAgent,
                "select_action",
                return_value=np.array([150.0, -175.0]),
            ), patch.object(ReplayBuffer, "push", new=capture_push):
                train_agent(
                    "ddpg",
                    episodes=1,
                    steps_per_episode=1,
                    output_dir=output_dir,
                    plot=False,
                )

        self.assertEqual(len(transitions), 1)
        _, stored_action, stored_reward, stored_next_state = transitions[0]
        np.testing.assert_array_equal(stored_action, np.array([100.0, -100.0]))

        expected_env = AUVConstantDepthEnv()
        expected_env.reset()
        expected_next_state, expected_reward, _, _ = expected_env.step(stored_action)
        np.testing.assert_allclose(stored_next_state, expected_next_state)
        self.assertAlmostEqual(stored_reward, expected_reward)

    def test_training_rejects_zero_and_negative_lengths(self):
        for parameter in ("episodes", "steps_per_episode"):
            for value in (0, -1):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaisesRegex(ValueError, parameter):
                        train_agent(plot=False, **{parameter: value})

    def test_library_and_cli_training_step_defaults_match(self):
        library_default = inspect.signature(train_agent).parameters[
            "steps_per_episode"
        ].default
        script_path = PROJECT_ROOT / "scripts" / "train.py"
        spec = importlib.util.spec_from_file_location("train_script", script_path)
        train_script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_script)

        with patch.object(sys, "argv", ["train.py", "ddpg"]):
            cli_default = train_script.parse_args().steps_per_episode

        self.assertEqual(library_default, 200)
        self.assertEqual(cli_default, library_default)

    def test_evaluation_rejects_zero_and_negative_lengths(self):
        for parameter in ("episodes", "steps_per_episode"):
            for value in (0, -1):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaisesRegex(ValueError, parameter):
                        evaluate_controller("unused.pth", plot=False, **{parameter: value})

    def test_default_training_path_uses_algorithm_and_seed(self):
        with tempfile.TemporaryDirectory() as project_root:
            with patch.object(experiment, "PROJECT_ROOT", Path(project_root)):
                train_agent(
                    "ddpg",
                    episodes=1,
                    steps_per_episode=1,
                    seed=23,
                    plot=False,
                )

            run_directory = Path(project_root) / "runs" / "ddpg_seed23"
            self.assertTrue(run_directory.is_dir())
            self.assertTrue((run_directory / "ddpg_actor.pth").is_file())
            self.assertTrue((run_directory / "run.json").is_file())

    def test_run_directory_exists_before_agent_construction(self):
        class AgentConstructionReached(Exception):
            pass

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "new-run"

            def check_directory(*_args, **_kwargs):
                self.assertTrue(output_directory.is_dir())
                raise AgentConstructionReached

            with patch.object(experiment, "DDPGAgent", side_effect=check_directory):
                with self.assertRaises(AgentConstructionReached):
                    train_agent(
                        "ddpg",
                        episodes=1,
                        steps_per_episode=1,
                        output_dir=output_directory,
                        plot=False,
                    )

    def test_training_overwrite_requires_explicit_approval(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            actor_path = output_directory / "ddpg_actor.pth"
            actor_path.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "ddpg_actor.pth"):
                train_agent(
                    "ddpg",
                    episodes=1,
                    steps_per_episode=1,
                    output_dir=output_directory,
                    plot=False,
                )
            self.assertEqual(actor_path.read_text(encoding="utf-8"), "existing")

            train_agent(
                "ddpg",
                episodes=1,
                steps_per_episode=1,
                output_dir=output_directory,
                plot=False,
                overwrite=True,
            )
            state_dict = torch.load(actor_path, map_location="cpu", weights_only=True)
            self.assertIn("net.0.weight", state_dict)

    def test_training_manifest_records_effective_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            train_agent(
                "td3",
                episodes=1,
                steps_per_episode=2,
                start_depth=1.5,
                target_depth=7.5,
                output_dir=temporary_directory,
                plot=False,
                seed=37,
            )
            manifest = json.loads(
                (Path(temporary_directory) / "run.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["algorithm"], "td3")
        self.assertEqual(manifest["seed"], 37)
        self.assertEqual(manifest["episodes"], 1)
        self.assertEqual(manifest["steps_per_episode"], 2)
        self.assertEqual(manifest["start_depth"], 1.5)
        self.assertEqual(manifest["target_depth"], 7.5)
        self.assertIn("device", manifest)
        self.assertIn("hyperparameters", manifest)
        self.assertEqual(manifest["hyperparameters"]["policy_delay"], 2)
        self.assertEqual(manifest["hyperparameters"]["hidden_sizes"], [128, 128])
        self.assertEqual(
            set(manifest["runtime"]),
            {"python", "numpy", "torch", "scipy", "matplotlib"},
        )
        self.assertEqual(manifest["artifacts"]["actor"], "td3_actor.pth")
        self.assertEqual(
            manifest["artifacts"]["critics"],
            ["td3_critic1.pth", "td3_critic2.pth"],
        )
        self.assertEqual(manifest["artifacts"]["rewards"], "td3_rewards.npy")
        self.assertEqual(manifest["artifacts"]["manifest"], "run.json")

    def test_evaluation_without_output_directory_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            checkpoint_path = temporary_path / "actor.pth"
            self._save_test_actor(checkpoint_path)
            files_before = set(temporary_path.iterdir())

            result = evaluate_controller(
                checkpoint_path,
                episodes=1,
                steps_per_episode=2,
                plot=False,
            )

            self.assertEqual(set(temporary_path.iterdir()), files_before)
            self.assertEqual(result[0].shape, (1, 2))

    def test_evaluation_state_metrics_include_final_transition(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "constant_actor.pth"
            self._save_constant_test_actor(checkpoint_path, [50.0, 50.0])
            depths, thetas, _, _, actions, rewards = evaluate_controller(
                checkpoint_path,
                episodes=1,
                steps_per_episode=1,
                start_depth=8.0,
                target_depth=8.0,
                plot=False,
            )

        expected_env = AUVConstantDepthEnv(z0=8.0, zr=8.0)
        expected_env.reset()
        final_state, _, _, _ = expected_env.step(actions[0, 0])
        expected_depth = final_state[0] + expected_env.zr
        expected_pitch = np.arctan2(final_state[2], final_state[1])
        metrics = experiment.calculate_metrics(
            depths,
            thetas,
            actions,
            rewards,
            target_depth=8.0,
            dt=expected_env.dt,
        )

        self.assertAlmostEqual(depths[0, -1], expected_depth)
        self.assertAlmostEqual(thetas[0, -1], expected_pitch)
        self.assertAlmostEqual(
            metrics["final_depth_error"][0], abs(expected_depth - 8.0)
        )
        self.assertAlmostEqual(
            metrics["maximum_depth_overshoot"][0],
            max(expected_depth - 8.0, 0.0),
        )
        self.assertAlmostEqual(
            metrics["maximum_absolute_pitch"][0], abs(expected_pitch)
        )

    def test_persisted_evaluation_outputs_and_shapes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            checkpoint_path = temporary_path / "actor.pth"
            output_directory = temporary_path / "evaluation"
            self._save_test_actor(checkpoint_path)

            result = evaluate_controller(
                checkpoint_path,
                episodes=2,
                steps_per_episode=3,
                start_depth=1.0,
                target_depth=7.0,
                label="test-controller",
                plot=False,
                output_dir=output_directory,
            )

            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {"evaluation.json", "trajectories.npz", "depth_tracking.png"},
            )
            with np.load(output_directory / "trajectories.npz") as trajectories:
                self.assertEqual(trajectories["depth"].shape, (2, 3))
                self.assertEqual(trajectories["pitch"].shape, (2, 3))
                self.assertEqual(trajectories["heave_velocity"].shape, (2, 3))
                self.assertEqual(trajectories["pitch_rate"].shape, (2, 3))
                self.assertEqual(trajectories["actions"].shape, (2, 3, 2))
                self.assertEqual(trajectories["rewards"].shape, (2,))
                np.testing.assert_array_equal(trajectories["depth"], result[0])
                np.testing.assert_array_equal(trajectories["actions"], result[4])

            record = json.loads(
                (output_directory / "evaluation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["metadata"]["controller_label"], "test-controller")
            self.assertEqual(record["metadata"]["checkpoint_path"], str(checkpoint_path))
            self.assertEqual(record["metadata"]["episodes"], 2)
            self.assertEqual(record["metadata"]["steps_per_episode"], 3)
            self.assertEqual(record["metadata"]["start_depth"], 1.0)
            self.assertEqual(record["metadata"]["target_depth"], 7.0)
            self.assertIn("total_reward", record["metrics"])
            self.assertGreater((output_directory / "depth_tracking.png").stat().st_size, 0)

    def test_evaluation_overwrite_requires_explicit_approval(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            checkpoint_path = temporary_path / "actor.pth"
            output_directory = temporary_path / "evaluation"
            self._save_test_actor(checkpoint_path)
            evaluate_controller(
                checkpoint_path,
                episodes=1,
                steps_per_episode=1,
                plot=False,
                output_dir=output_directory,
            )

            with self.assertRaisesRegex(FileExistsError, "evaluation.json"):
                evaluate_controller(
                    checkpoint_path,
                    episodes=1,
                    steps_per_episode=1,
                    plot=False,
                    output_dir=output_directory,
                )

            evaluate_controller(
                checkpoint_path,
                episodes=1,
                steps_per_episode=1,
                plot=False,
                output_dir=output_directory,
                overwrite=True,
            )
            self.assertTrue((output_directory / "evaluation.json").is_file())


class LQITests(unittest.TestCase):
    def test_closed_loop_is_stable(self):
        self.assertTrue(np.all(np.real(CLOSED_LOOP_EIGENVALUES) < 0))

    def test_simulation_shapes(self):
        result = simulate_lqi(duration=1.0, num_points=11)
        self.assertEqual(result["solution"].shape, (11, 6))
        self.assertEqual(result["actions"].shape, (11, 2))


if __name__ == "__main__":
    unittest.main()
