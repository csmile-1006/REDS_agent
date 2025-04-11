from typing import Any, Union, Dict, Tuple, Optional, List
import numpy as np
import torch

from omegaconf import OmegaConf


class InvalidSequenceError(Exception):
    def __init__(self, message):
        super().__init__(message)


class VQDiffusionRewardModel:
    PRIVATE_LIKELIHOOD_KEY = "log_immutable_density"
    PUBLIC_LIKELIHOOD_KEY = "density"

    def __init__(
        self,
        task: str,
        vqdiffusion_path: str,
        camera_key: str = "image",
        reward_scale: Optional[Union[Dict[str, Tuple], Tuple]] = None,
        reward_model_device: int = 0,
    ):
        """VideoGPT likelihood model for reward computation.

        Args:
            task: Task name, used for conditioning when class_map.pkl exists in videogpt_path.
            vqgan_path: Path to vqgan weights.
            videogpt_path: Path to videogpt weights.
            camera_key: Key for camera observation.
            reward_scale: Range to scale logits from [0, 1].
            reward_model_device: Device to run reward model on.
            nll_reduce_sum: Whether to reduce sum over nll.
            compute_joint: Whether to compute joint likelihood or use conditional.
            minibatch_size: Minibatch size for VideoGPT.
            encoding_minibatch_size: Minibatch size for VQGAN.
        """
        self.domain, self.task = task.split("_", 1)
        self.camera_key = camera_key
        self.reward_scale = reward_scale

        # cfg = yaml.safe_load(open(vqdiffusion_path, "r"))
        cfg = OmegaConf.load(vqdiffusion_path)
        cfg.task_name = self.task + "-v2" if "metaworld" in task else self.task
        from diffusion_reward.models.reward_models import make_rm

        self.device = torch.device(cfg.device)
        self.vqdiffusion_reward_model = make_rm(cfg).to(self.device)

        self.seq_len = self.vqdiffusion_reward_model.subseq_len
        self.seq_len_steps = self.seq_len
        self.use_task_reward = True
        self.mask = None

        print(
            f"finished loading {self.__class__.__name__}:"
            f"\n\tseq_len: {self.seq_len}"
            f"\n\ttask: {self.task}"
            f"\n\tcamera_key: {self.camera_key}"
            f"\n\tseq_len_steps: {self.seq_len_steps}"
        )

    def __call__(self, seq, **kwargs):
        return self.process_seq(self.compute_reward(seq, **kwargs), **kwargs)

    def _reward_scaler(self, reward):
        if self.reward_scale:
            if isinstance(self.reward_scale, dict) and (self.task not in self.reward_scale):
                return reward
            rs = self.reward_scale[self.task] if isinstance(self.reward_scale, dict) else self.reward_scale
            reward = np.array(np.clip((reward - rs[0]) / (rs[1] - rs[0]), 0.0, 1.0))
            return reward
        else:
            return reward

    def compute_reward(self, seq: List[Dict[str, Any]], **kwargs):
        """Use VGPT model to compute likelihoods for input sequence.
        Args:
            seq: Input sequence of states.
        Returns:
            seq: Input sequence with additional keys in the state dict.
        """
        if len(seq) < self.seq_len_steps:
            raise InvalidSequenceError(
                f"Input sequence must be at least {self.seq_len_steps} steps long. Seq len is {len(seq)}"
            )

        # Where in sequence to start computing likelihoods. Don't perform redundant likelihood computations.
        start_idx = 0
        for i in range(len(seq)):
            if not self.is_step_processed(seq[i]):
                start_idx = i
                break
        start_idx = int(max(start_idx, 0))
        T = len(seq) - start_idx

        image_batch = np.stack([seq[i][self.camera_key] for i in range(start_idx, len(seq))])
        image_batch = self.process_images(image_batch)
        images = torch.Tensor(image_batch)[None]

        rewards = self.vqdiffusion_reward_model.calc_reward(images).cpu().numpy()
        if len(rewards.shape) <= 1:
            rewards = self._reward_scaler(rewards)
        if self.use_task_reward:
            rewards = rewards + np.array([seq[i]["reward"] for i in range(start_idx, len(seq))])

        assert len(rewards) == T, f"{len(rewards)} != {T}"
        for i, rew in enumerate(rewards):
            idx = start_idx + i
            assert not self.is_step_processed(seq[idx])
            seq[idx][VQDiffusionRewardModel.PRIVATE_LIKELIHOOD_KEY] = rew

        return seq

    def is_step_processed(self, step):
        return VQDiffusionRewardModel.PRIVATE_LIKELIHOOD_KEY in step.keys()

    def is_seq_processed(self, seq):
        for step in seq:
            if not self.is_step_processed(step):
                return False
        return True

    def process_images(self, image_batch):
        image_batch = np.array(image_batch).astype(np.uint8)
        image_batch = image_batch * self.mask if self.mask is not None else image_batch
        return image_batch.astype(np.float32) / 127.5 - 1.0

    def process_seq(self, seq, whole_output: bool = False):
        for step in seq:
            if not self.is_step_processed(step):
                continue
            step[VQDiffusionRewardModel.PUBLIC_LIKELIHOOD_KEY] = step[VQDiffusionRewardModel.PRIVATE_LIKELIHOOD_KEY]
        return seq[self.seq_len_steps - 1 :] if not whole_output else seq
