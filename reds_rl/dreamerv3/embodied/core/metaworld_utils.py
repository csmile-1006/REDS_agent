import gc
import numpy as np
from tqdm import tqdm

from metaworld import policies

POLICIES = {
    "assembly-v2-goal-observable": policies.SawyerAssemblyV2Policy,
    "basketball-v2-goal-observable": policies.SawyerBasketballV2Policy,
    "bin-picking-v2-goal-observable": policies.SawyerBinPickingV2Policy,
    "box-close-v2-goal-observable": policies.SawyerBoxCloseV2Policy,
    "button-press-v2-goal-observable": policies.SawyerButtonPressV2Policy,
    "button-press-topdown-v2-goal-observable": policies.SawyerButtonPressTopdownV2Policy,
    "button-press-topdown-wall-v2-goal-observable": policies.SawyerButtonPressTopdownWallV2Policy,
    "button-press-wall-v2-goal-observable": policies.SawyerButtonPressWallV2Policy,
    "coffee-button-v2-goal-observable": policies.SawyerCoffeeButtonV2Policy,
    "coffee-pull-v2-goal-observable": policies.SawyerCoffeePullV2Policy,
    "coffee-push-v2-goal-observable": policies.SawyerCoffeePushV2Policy,
    "dial-turn-v2-goal-observable": policies.SawyerDialTurnV2Policy,
    "disassemble-v2-goal-observable": policies.SawyerDisassembleV2Policy,
    "door-close-v2-goal-observable": policies.SawyerDoorCloseV2Policy,
    "door-lock-v2-goal-observable": policies.SawyerDoorLockV2Policy,
    "door-open-v2-goal-observable": policies.SawyerDoorOpenV2Policy,
    "door-unlock-v2-goal-observable": policies.SawyerDoorUnlockV2Policy,
    "drawer-close-v2-goal-observable": policies.SawyerDrawerCloseV2Policy,
    "drawer-open-v2-goal-observable": policies.SawyerDrawerOpenV2Policy,
    "faucet-close-v2-goal-observable": policies.SawyerFaucetCloseV2Policy,
    "faucet-open-v2-goal-observable": policies.SawyerFaucetOpenV2Policy,
    "hammer-v2-goal-observable": policies.SawyerHammerV2Policy,
    "hand-insert-v2-goal-observable": policies.SawyerHandInsertV2Policy,
    "handle-press-v2-goal-observable": policies.SawyerHandlePressV2Policy,
    "handle-pull-v2-goal-observable": policies.SawyerHandlePullV2Policy,
    "handle-pull-side-v2-goal-observable": policies.SawyerHandlePullSideV2Policy,
    "lever-pull-v2-goal-observable": policies.SawyerLeverPullV2Policy,
    "peg-insert-side-v2-goal-observable": policies.SawyerPegInsertionSideV2Policy,
    "pick-out-of-hole-v2-goal-observable": policies.SawyerPickOutOfHoleV2Policy,
    "pick-place-v2-goal-observable": policies.SawyerPickPlaceV2Policy,
    "pick-place-wall-v2-goal-observable": policies.SawyerPickPlaceWallV2Policy,
    "plate-slide-v2-goal-observable": policies.SawyerPlateSlideV2Policy,
    "plate-slide-back-v2-goal-observable": policies.SawyerPlateSlideBackV2Policy,
    "plate-slide-side-v2-goal-observable": policies.SawyerPlateSlideSideV2Policy,
    "push-v2-goal-observable": policies.SawyerPushV2Policy,
    "push-back-v2-goal-observable": policies.SawyerPushBackV2Policy,
    "push-wall-v2-goal-observable": policies.SawyerPushWallV2Policy,
    "reach-v2-goal-observable": policies.SawyerReachV2Policy,
    "reach-wall-v2-goal-observable": policies.SawyerReachWallV2Policy,
    "shelf-place-v2-goal-observable": policies.SawyerShelfPlaceV2Policy,
    "soccer-v2-goal-observable": policies.SawyerSoccerV2Policy,
    "stick-pull-v2-goal-observable": policies.SawyerStickPullV2Policy,
    "stick-push-v2-goal-observable": policies.SawyerStickPushV2Policy,
    "sweep-into-v2-goal-observable": policies.SawyerSweepIntoV2Policy,
    "sweep-v2-goal-observable": policies.SawyerSweepV2Policy,
    "window-close-v2-goal-observable": policies.SawyerWindowCloseV2Policy,
    "window-open-v2-goal-observable": policies.SawyerWindowOpenV2Policy,
}

THETA_LIMIT = 0.05


def collect_demo(env, demo_replay, num_demos, **kwargs):
    transitions = []
    print("collecting demos.")
    for _ in tqdm(range(num_demos)):
        transitions.extend(rollout_demo(env))
    # Restrict translation space by min_max
    for obs in transitions:
        demo_replay.add(obs)

    del transitions
    gc.collect()
    return


def rollout_demo(env):
    policy = POLICIES[env._task]()
    ts, success = 0.0, 0.0
    obs = env.reset()
    transitions = []
    while True:
        # Roll out an episode.
        a = policy.get_action(obs["state"])
        a = np.clip(a, -1.0, 1.0)

        next_obs = env.step({"action": a, "reset": False})
        success += float(next_obs["success"])
        success = min(success, 1.0)
        transitions.append(next_obs)
        ts += 1
        if next_obs["is_last"]:
            if success > 0:
                return transitions
            else:
                next_obs = env.reset()
                ts, success = 0.0, 0.0
                transitions = []
        obs = next_obs
