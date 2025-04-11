import os
from functools import partial

from reds_rl import CONFIG_PATH

from .videogpt_reward_model import VideoGPTRewardModel
from .vqdiffusion_reward_model import VQDiffusionRewardModel
from ... import CHECKPOINT_PATH

bind_videogpt_reward_model = lambda **kwargs: partial(VideoGPTRewardModel, **kwargs)
bind_vqdiffusion_reward_model = lambda **kwargs: partial(VQDiffusionRewardModel, **kwargs)

checkpoint_dir = os.getenv("VIPER_CHECKPOINT_DIR") or str(CHECKPOINT_PATH)
get_path = lambda dir: str(os.path.join(checkpoint_dir, dir))

LOAD_REWARD_MODEL_DICT = {
    # DeepMind Control Suite.
    "dmc_clen16_fskip1": bind_videogpt_reward_model(  # No frame skip
        videogpt_path=get_path("dmc_videogpt_l16_s1"), vqgan_path=get_path("dmc_vqgan")
    ),
    "dmc_clen16_fskip2": bind_videogpt_reward_model(  # Frame skip 2
        videogpt_path=get_path("dmc_videogpt_l8_s2"), vqgan_path=get_path("dmc_vqgan")
    ),
    "dmc_clen16_fskip4": bind_videogpt_reward_model(  # Frame skip 4
        videogpt_path=get_path("dmc_videogpt_l4_s4"), vqgan_path=get_path("dmc_vqgan")
    ),
    # Atari.
    "atari_clen16_fskip1": bind_videogpt_reward_model(  # No frame skip
        videogpt_path=get_path("atari_videogpt_l16_s1"), vqgan_path=get_path("atari_vqgan")
    ),
    "atari_clen16_fskip2": bind_videogpt_reward_model(  # Frame skip 2
        videogpt_path=get_path("atari_videogpt_l8_s2"), vqgan_path=get_path("atari_vqgan")
    ),
    "atari_clen16_fskip4": bind_videogpt_reward_model(  # Frame skip 4
        videogpt_path=get_path("atari_videogpt_l4_s4"), vqgan_path=get_path("atari_vqgan")
    ),
    # Metaworld.
    "metaworld_door-open_clen4_fskip4": bind_videogpt_reward_model(  # Frame skip 4
        videogpt_path=get_path("door-open_videogpt_l4_s4"), vqgan_path=get_path("door-open_vqgan")
    ),
    # Metaworld.
    "metaworld_lever-pull_clen4_fskip4": bind_videogpt_reward_model(  # Frame skip 4
        videogpt_path=get_path("lever-pull_videogpt_l4_s4"), vqgan_path=get_path("lever-pull_vqgan")
    ),
    # Metaworld.
    "metaworld_clen4_fskip4": bind_videogpt_reward_model(  # Frame skip 4
        videogpt_path=get_path("metaworld_videogpt_l4_s4"), vqgan_path=get_path("metaworld_vqgan")
    ),
    # RLBench.
    "rlbench_clen4_fskip4": bind_videogpt_reward_model(  # Frame skip 4
        videogpt_path=get_path("rlbench_videogpt_l4_s4"),
        vqgan_path=get_path("rlbench_vqgan"),
        camera_key="image_front",
    ),
    # Metaworld VQDiffusion
    "metaworld_vqdiffusion": bind_vqdiffusion_reward_model(
        vqdiffusion_path=CONFIG_PATH / "vqdiffusion" / "metaworld.yaml",
        camera_key="image",
    ),
}
