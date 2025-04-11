import importlib
import pathlib
import sys
import warnings
from functools import partial as bind

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
    args = embodied.Config(**config.run, logdir=config.logdir, batch_steps=config.batch_size * config.batch_length)
    print(config)

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    config.save(logdir / "config.yaml")
    step = embodied.Counter()
    logger = make_logger(parsed, logdir, step, config)

    reward_model = None
    if config.reward_model == "REDS" and config.reward_model_path != "none":
        print(f"Loading our reward model from {config.reward_model_path}")
        from bpref_v2.reward_model.reds_reward_model import REDSRewardModel

        suite, task = config.task.split("_", 1)
        kwargs = config.env.get(suite, {})

        reward_model = REDSRewardModel(
            task=config.task,
            model_name=config.reward_model_type,
            rm_path=config.reward_model_path,
            encoding_minibatch_size=config.reward_model_batch_size,
            camera_keys=kwargs.get("camera_keys", "image"),
            window_size=config.reward_model_window_size,
            skip_frame=config.reward_model_skip_frame,
            use_task_reward=config.reward_model_use_task_reward,
            use_scale=config.reward_model_use_scale,
            clip_value=config.reward_model_clip,
            scale_value=config.reward_model_scale,
        )

    if config.reward_model_type == "VQDiffusion":
        print(f"Loading Diffusion Reward model from {config.reward_model_path}")
        from reds_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT

        suite, task = config.task.split("_", 1)
        kwargs = config.env.get(suite, {})

        reward_model = LOAD_REWARD_MODEL_DICT[config.reward_model](
            task=config.task,
            camera_key=kwargs.get("camera_keys", "image"),
        )

    if config.reward_model != "none" and config.reward_model_path == "none":
        print(f"Loading reward model {config.reward_model}")
        from reds_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT

        reward_model = LOAD_REWARD_MODEL_DICT[config.reward_model](
            task=config.task,
            compute_joint=config.reward_model_compute_joint,
            minibatch_size=config.reward_model_batch_size,
            encoding_minibatch_size=config.reward_model_batch_size,
            reward_model_device=config.jax.reward_model_device,
        )

    replay_kwargs = {"reward_model": reward_model}

    cleanup = []
    try:
        if args.script == "train":
            env = make_envs(config)
            replay = make_replay(config, logdir / "replay", **replay_kwargs)
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train(agent, env, replay, logger, args)

        elif args.script == "train_amp":
            replay_config = config.update({"replay": "uniform"})
            reference_replay = make_replay(replay_config, config.reference_dir, is_eval=False)
            # pre-collect expert demonstrations using scripted policy.
            dev_env = make_env(config)
            cleanup.append(dev_env)
            suite, task = config.task.split("_", 1)
            if suite == "metaworld":
                # delete cont head
                # config = config.update({"grad_heads": config.grad_heads[:-1]})
                embodied.metaworld_collect_demo(dev_env, reference_replay, config.num_demos)
                actions_min_max = None
            elif suite == "rlbench":
                actions_min_max = embodied.rlbench_collect_demo(
                    dev_env,
                    reference_replay,
                    config.num_demos,
                    config.env.rlbench.camera_keys,
                    reward_type=config.env.rlbench.reward_type,
                    use_rotation=config.env.rlbench.use_rotation,
                    randomize=config.use_randomize,
                )
            else:
                raise NotImplementedError(f"not available of collecting expert demonstrations for {suite}")

            print(f"Loaded reference data: {reference_replay.stats}")
            replay = make_replay(config, logdir / "replay", **replay_kwargs)
            env, env_fn = make_envs(config, actions_min_max=actions_min_max)
            eval_replay = make_replay(config, logdir / "eval_replay", **replay_kwargs, is_eval=True)
            eval_env, eval_env_fn = make_envs(config, actions_min_max=actions_min_max)  # mode='eval'
            cleanup += [env, eval_env]
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_amp(
                agent, env, eval_env, env_fn, eval_env_fn, replay, reference_replay, eval_replay, logger, args
            )

        elif args.script == "train_save":
            env = make_envs(config)
            replay = make_replay(config, logdir / "replay", **replay_kwargs)
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_save(agent, env, replay, logger, args)

        elif args.script == "train_eval":
            replay = make_replay(config, logdir / "replay", **replay_kwargs)
            eval_replay = make_replay(config, logdir / "eval_replay", **replay_kwargs, is_eval=True)
            suite, task = config.task.split("_", 1)
            # if suite == "metaworld":
            #     config.grad_heads = config.grad_heads[:-1]
            if suite == "rlbench" and config.num_demos > 0:
                demo_replay = make_replay(config, logdir / "demo_replay", **replay_kwargs)
                # demo_replay = None
                dev_env = make_env(config)
                cleanup.append(dev_env)
                actions_min_max = embodied.rlbench_collect_demo(
                    dev_env,
                    demo_replay,
                    config.num_demos,
                    config.env.rlbench.camera_keys,
                    reward_type=config.env.rlbench.reward_type,
                    use_rotation=config.env.rlbench.use_rotation,
                    randomize=config.use_randomize,
                )
                # print(f"Loaded reference data: {demo_replay.stats}")
                print(f"Loaded reference data: {replay.stats}")
            else:
                actions_min_max = None
                demo_replay = None
            env, env_fn = make_envs(config, actions_min_max=actions_min_max)
            eval_env, eval_env_fn = make_envs(config, actions_min_max=actions_min_max)  # mode='eval'
            cleanup += [env, eval_env]
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_eval(
                agent, env, eval_env, env_fn, eval_env_fn, replay, eval_replay, demo_replay, logger, args
            )

        elif args.script == "train_holdout":
            env = make_envs(config)
            replay = make_replay(config, logdir / "replay", **replay_kwargs)
            if config.eval_dir:
                assert not config.train.eval_fill
                eval_replay = make_replay(config, config.eval_dir, is_eval=True)
            else:
                assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
                eval_replay = make_replay(config, logdir / "eval_replay", is_eval=True, **replay_kwargs)
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_holdout(agent, env, replay, eval_replay, logger, args)

        elif args.script == "eval_only":
            env = make_envs(config)  # mode='eval'
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.eval_only(agent, env, logger, args)

        elif args.script == "eval_only_save":
            env = make_envs(config)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            cleanup.append(env)
            embodied.run.eval_only_save(agent, env, logger, args)

        elif args.script == "parallel":
            assert config.run.actor_batch <= config.envs.amount, (config.run.actor_batch, config.envs.amount)
            step = embodied.Counter()
            env = make_env(config)
            env.close()
            replay = make_replay(config, logdir / "replay", reward_model=reward_model, rate_limit=True, **replay_kwargs)
            agent = agt.Agent(env.obs_space, env.act_space, step, reward_model, config)
            embodied.run.parallel(agent, replay, logger, bind(make_env, config), num_envs=config.envs.amount, args=args)

        else:
            raise NotImplementedError(args.script)
    finally:
        for obj in cleanup:
            obj.close()


def make_logger(parsed, logdir, step, config):
    multiplier = config.env.get(config.task.split("_")[0], {}).get("repeat", 1)
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(config.filter),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.JSONLOutput(logdir, "scores.jsonl", "episode/score"),
            embodied.logger.JSONLOutput(logdir, "success.jsonl", "max_success"),
            embodied.logger.TensorBoardOutput(logdir),
            embodied.logger.WandBOutput(logdir, config),
            # embodied.logger.MLFlowOutput(logdir.name),
        ],
        multiplier,
    )
    return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False, reward_model=None, **kwargs):
    assert config.replay == "uniform" or config.replay == "uniform_relabel" or not rate_limit
    length = config.batch_length
    size = config.replay_size // 10 if is_eval else config.replay_size
    if config.replay == "uniform_relabel":
        kw = {"online": config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw["samples_per_insert"] = config.run.train_ratio / config.batch_length
            kw["tolerance"] = 10 * config.batch_size
            kw["min_size"] = config.batch_size
        assert reward_model is not None, "relabel requires reward model"
        replay = embodied.replay.UniformRelabel(
            length, reward_model, config.uniform_relabel_add_mode, size, directory, **kw
        )
    elif config.replay == "uniform" or is_eval:
        kw = {"online": config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw["samples_per_insert"] = config.run.train_ratio / config.batch_length
            kw["tolerance"] = 10 * config.batch_size
            kw["min_size"] = config.batch_size
        replay = embodied.replay.Uniform(length, size, None, **kw)
    elif config.replay == "reverb":
        replay = embodied.replay.Reverb(length, size, directory)
    elif config.replay == "chunks":
        replay = embodied.replay.NaiveChunks(length, size, directory)
    else:
        raise NotImplementedError(config.replay)
    return replay


def make_envs(config, **overrides):
    suite, task = config.task.split("_", 1)
    ctors = []
    for index in range(config.envs.amount):
        ctor = lambda: make_env(config, **overrides)
        if config.envs.parallel != "none":
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if config.envs.restart:
            ctor = bind(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs.parallel != "none")), ctors[0]


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
        "factorworld": "embodied.envs.factorworld:FactorWorld",
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
    if args.rank:
        env = wrappers.Rank(env)
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
