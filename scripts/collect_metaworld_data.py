import importlib
import pathlib
import sys
import warnings


warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*using stateful random seeds*")
warnings.filterwarnings("ignore", ".*is a deprecated alias for.*")
warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from reds_rl.dreamerv3 import embodied  # noqa: E402
from reds_rl.dreamerv3.embodied import wrappers  # noqa: E402


def main(argv=None):
    from reds_rl.dreamerv3 import agent as agt

    parsed, other = embodied.Flags(configs=["defaults"]).parse_known(argv)
    config = embodied.Config(agt.Agent.configs["defaults"])
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)
    print(config)

    cleanup = []
    try:
        env = make_env(config)
        reference_replay = make_replay(config, config.reference_dir, is_eval=False)
        embodied.metaworld_collect_demo(
            env,
            reference_replay,
            config.num_demos,
        )
    finally:
        for obj in cleanup:
            obj.close()


def make_replay(config, directory=None, is_eval=False, rate_limit=False, reward_model=None, **kwargs):
    assert config.replay == "chunks" or not rate_limit
    length = config.batch_length
    size = config.replay_size // 10 if is_eval else config.replay_size
    replay = embodied.replay.NaiveChunks(length, size, directory, chunks=500)
    return replay


def make_env(config, **overrides):
    # You can add custom environments by creating and returning the environment
    # instance here. Environments with different interfaces can be converted
    # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
    suite, task = config.task.split("_", 1)
    ctor = {
        "dummy": "embodied.envs.dummy:Dummy",
        "gym": "embodied.envs.from_gym:FromGym",
        "dm": "embodied.envs.from_dmenv:FromDM",
        "crafter": "embodied.envs.crafter:Crafter",
        "dmc": "embodied.envs.dmc:DMC",
        "rlbench": "embodied.envs.rlbench:RLBench",
        "metaworld": "embodied.envs.metaworld:MetaWorld",
        "dmcmulticam": "embodied.envs.dmcmulticam:DMCMultiCam",
        "atari": "embodied.envs.atari:Atari",
        "dmlab": "embodied.envs.dmlab:DMLab",
        "minecraft": "embodied.envs.minecraft:Minecraft",
        "loconav": "embodied.envs.loconav:LocoNav",
        "pinpad": "embodied.envs.pinpad:PinPad",
        "kitchen": "embodied.envs.kitchen:Kitchen",
        "cliport": "embodied.envs.cliport:Cliport",
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(":")
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def wrap_env(env, config):
    args = config.wrapper
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = wrappers.NormalizeAction(env, name)
    if args.density:
        env = wrappers.Density(env)
    env = wrappers.FlattenTwoDimObs(env)
    env = wrappers.ExpandScalars(env)
    if args.length:
        env = wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    return env


if __name__ == "__main__":
    main()
