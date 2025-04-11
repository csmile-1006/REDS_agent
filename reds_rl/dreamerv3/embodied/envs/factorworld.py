import functools
import os

import embodied
import numpy as np


class FactorWorld(embodied.Env):
    def __init__(self, env, seed=None, repeat=1, size=(64, 64), camera=None, reward_type="dense", **kwargs):
        # TODO: This env variable may be necessary when running on a headless GPU
        # but breaks when running on a CPU machine.
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"

        from metaworld_generalization.envs.env_dict import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        from metaworld_generalization.envs.factors.utils import make_env_with_factors

        # from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

        self._task = f"{env}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[self._task]
        env_kwargs = dict(
            camera_name=["corner2"],
            get_image_obs=True,
            image_obs_size=size,
            random_init=True,
            seed=seed,
        )
        factor_kwargs = dict(
            # camera_pos=dict(
            #     azimuth_range=[1.5707963267948966, 2.356194490192345],
            #     inclination_range=[1.0471975511965976, 1.308996938995747],
            #     radius_range=[1.25, 1.75],
            #     num_resets_per_randomize="default",
            #     seed=seed,
            # ),
            light=dict(diffuse_range=[0.3, 0.8], num_resets_per_randomize=1, seed=seed),
            table_pos=dict(
                x_range=[-0.025, 0.025],
                y_range=[-0.1, 0.1],
                z_range=[-0.08, 0.0],
                num_resets_per_randomize=1,
                seed=seed,
            ),
        )
        self._env = make_env_with_factors(env_cls, env_kwargs, factor_kwargs=factor_kwargs, use_train_xml=False)
        self._env._freeze_rand_vec = False

        # self._env = embodied.wrappers.ExpandScalars(self._env)
        self._repeat = repeat
        self._size = size
        self._camera = camera
        assert reward_type in [
            "sparse",
            "semi_sparse",
            "dense",
        ], f"{reward_type} not in ['sparse', 'semi_sparse', 'dense']"
        self._reward_type = reward_type

    @functools.cached_property
    def obs_space(self):
        return {
            "image": embodied.Space(np.uint8, self._size + (3,), low=0, high=255),
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
            "state": self._convert(self._env.observation_space["proprio"]),
            "success": embodied.Space(bool, low=0, high=1),
            "skill": embodied.Space(np.int32),
        }

    @functools.cached_property
    def act_space(self):
        return {
            "reset": embodied.Space(bool),
            "action": self._convert(self._env.action_space),
        }

    def step(self, action):
        if action["reset"]:
            return self.reset()

        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        success = 0.0
        skill = 0
        for _ in range(self._repeat):
            state, rew, done, info = self._env.step(action["action"])
            success += float(info["success"])
            skill = max(int(info.get("skill", 0)), skill)
            reward += rew or 0.0
        success = min(success, 1.0)
        assert success in [0.0, 1.0]

        if self._reward_type == "sparse":
            output_reward = success
        elif self._reward_type == "semi_sparse":
            output_reward = skill
        elif self._reward_type == "dense":
            output_reward = reward
        obs = {
            "reward": output_reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": state["image"]["corner2"],
            "state": state["proprio"],
            "success": success,
            "skill": skill,
        }
        return obs

    def reset(self):
        # if self._camera == "corner2":
        #     self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": state["image"]["corner2"],
            "state": state["proprio"],
            "success": False,
            "skill": 0,
        }
        return obs

    def _convert(self, space):
        if hasattr(space, "n"):
            return embodied.Space(np.int32, (), 0, space.n)
        return embodied.Space(space.dtype, space.shape, space.low, space.high)
