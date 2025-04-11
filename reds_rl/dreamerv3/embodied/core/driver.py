import collections

import numpy as np
import jax.numpy as jnp

from .basics import convert


class Driver:
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, env, env_fn, **kwargs):
        assert len(env) > 0
        self._env = env
        self._env_fn = env_fn
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.reset()

    def reset(self):
        self._acts = {
            k: convert(np.zeros((len(self._env),) + v.shape, v.dtype)) for k, v in self._env.act_space.items()
        }
        self._acts["reset"] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None

    def handle_sim_failure(self, i):
        # If reset failed, re-start env
        print(f"Resetting Env {i} due to sim failure")
        self._env._envs[i].close()
        self._env._envs[i] = self._env_fn()
        # NOTE: We need this to set a new action space
        # A bit hacky ..
        self._env._envs[i].act_space

        if self._state is not None:
            if len(self._env) == 1:
                # If not using parallel envs, let's just set state to None
                self._state = None
            else:
                # If using parallel envs, manually reset state = (latent, action)

                # Choose different index j != i, which is index of other envs
                j = i - 1 if i != 0 else i + 1

                # Set new empty latent
                (latent, action), task_state, expl_state = self._state
                new_latent = dict()
                for key in latent.keys():
                    new_latent[key] = jnp.concatenate(
                        [
                            latent[key][:i],
                            jnp.zeros_like(latent[key][j])[None],
                            latent[key][i + 1 :],
                        ],
                        axis=0,
                    )
                # Set new empty action
                new_action = jnp.concatenate(
                    [
                        action[:i],
                        jnp.zeros_like(action[j])[None],
                        action[i + 1 :],
                    ],
                    axis=0,
                )
                self._state = ((new_latent, new_action), task_state, expl_state)

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        assert all(len(x) == len(self._env) for x in self._acts.values())
        action = {k: v for k, v in self._acts.items() if not k.startswith("log_")}
        # obs = self._env.step(acts)
        obs = []
        for i in range(len(self._env)):
            act = {k: v[i] for k, v in action.items()}
            if act["reset"]:
                reset_success = False
                while not reset_success:
                    try:
                        ob = self._env._envs[i].step(act)
                        if self._env._parallel:
                            ob = ob()
                        reset_success = True
                    except Exception:
                        self.handle_sim_failure(i)
            else:
                try:
                    ob = self._env._envs[i].step(act)
                    if self._env._parallel:
                        ob = ob()
                except Exception as e:
                    print(f"Skipping step for Env {i} due to sim failure: {e}")
                    self.handle_sim_failure(i)
                    obs_keys = [key for key in self._env.obs_space.keys() if key not in ["density"]]
                    last_ob = {k: self._env.obs_space[k].sample() for k in obs_keys}
                    last_ob["reward"] = 0.0
                    last_ob["success"] = 0.0
                    last_ob["state"] = np.zeros_like(last_ob["state"])
                    last_ob["is_first"] = True
                    last_ob["is_last"] = True
                    last_ob["is_terminal"] = True
                    last_ob["skill"] = 0.0
                    ob = last_ob
            obs.append(ob)
        obs = {k: np.array([ob[k] for ob in obs]) for k in obs[0]}
        obs = {k: convert(v) for k, v in obs.items()}
        assert all(len(x) == len(self._env) for x in obs.values()), obs
        acts, self._state = policy(obs, self._state, **self._kwargs)
        acts = {k: convert(v) for k, v in acts.items()}
        if obs["is_last"].any():
            mask = 1 - obs["is_last"]
            acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
        acts["reset"] = obs["is_last"].copy()
        self._acts = acts
        trns = {**obs, **acts}
        if obs["is_first"].any():
            for i, first in enumerate(obs["is_first"]):
                if first:
                    self._eps[i].clear()
        for i in range(len(self._env)):
            trn = {k: v[i] for k, v in trns.items()}
            [self._eps[i][k].append(v) for k, v in trn.items()]
            [fn(trn, i, **self._kwargs) for fn in self._on_steps]
            step += 1
        if obs["is_last"].any():
            for i, done in enumerate(obs["is_last"]):
                if done:
                    ep = {k: convert(v) for k, v in self._eps[i].items()}
                    [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value
