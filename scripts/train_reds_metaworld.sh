if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <task_name> <iteration> <device> <reward_training_steps> <dreamer_training_steps> <num_demos> <num_failure_demos>"
    exit 1
fi

TASK_TYPE=metaworld
TASK_NAME=${1}
ITERATION=${2}
DEVICES=${3}
TRAINING_STEPS=${4}
DREAMER_TRAINING_STEPS=${5}
NUM_DEMOS=${6}
NUM_FAILURE_DEMOS=${7}
BASE_PATH=${8}
SEED=0
REPLAY_PATH=${BASE_PATH}/pretrain_dreamerv3
REWARD_MODEL_PATH=${BASE_PATH}/reds_logdir

# if [ "$NUM_FAILURE_DEMOS" -lt 50 ]; then
#     NUM_DEMOS=50
# else
#     NUM_DEMOS=$NUM_FAILURE_DEMOS
# fi

XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=${DEVICES} python -m bpref_v2.reward_learning.train_reds \
    --comment ${TASK_NAME}-phase0 \
    --robot.data_dir ${BASE_PATH}/${TASK_TYPE}_data/${TASK_NAME} \
    --reds.embd_dim 512 \
    --reds.output_embd_dim 512 \
    --reds.n_layer 1 \
    --reds.n_head 4 \
    --reds.lr=5e-5 \
    --logging.output_dir ${REWARD_MODEL_PATH} \
    --batch_size=32 \
    --seed 0 \
    --model_type REDS \
    --early_stop True \
    --robot.task_name ${TASK_NAME} \
    --robot.num_demos 50 \
    --env "${TASK_TYPE}-${TASK_NAME}" \
    --robot.window_size=4 \
    --robot.skip_frame=1 \
    --reds.lambda_supcon=1.0 \
    --reds.lambda_epic=1.0 \
    --reds.epic_on_neg_batch=True \
    --reds.supcon_on_neg_batch=True \
    --use_failure=True \
    --reds.transfer_type clip_vit_b16 \
    --augmentations "crop|jitter" \
    --robot.num_workers=2 \
    --robot.output_type raw \
    --logging.online True \
    --train_steps=${TRAINING_STEPS}

for (( i = 1; ($i < $ITERATION); i++ )) 
do 
    XLA_PYTHON_CLIENT_PREALLOCATE=false LD_PRELOAD="" CUDA_VISIBLE_DEVICES=${DEVICES} python scripts/train_dreamer.py \
        --configs=${TASK_TYPE} reds_prior_rb \
        --logdir=${REPLAY_PATH}/${TASK_NAME}_ours_phase${i}_seed${SEED} \
        --task=${TASK_TYPE}_${TASK_NAME} \
        --reward_model_path=${REWARD_MODEL_PATH}/REDS/${TASK_TYPE}-${TASK_NAME}/${TASK_NAME}-phase${i}/s0/last_model.pkl \
        --seed=0 \
        --prior_rewards.disag=0.0 \
        --run.steps=${DREAMER_TRAINING_STEPS} \
        --wandb_name=reds-${TASK_TYPE}-pretrain-automatic-phase${i}

    python -m bpref_v2.utils.replay2traj \
        --data_path=${BASE_PATH}/${TASK_TYPE}_data/ \
        --task_name=${TASK_NAME} \
        --replay_path=${REPLAY_PATH} \
        --experiment_key=ours_phase${i} \
        --image_keys="image" \
        --chunk_size=1024 \
        --num_demos=${NUM_FAILURE_DEMOS}

    NUM_FAILURE_DEMOS=$((NUM_FAILURE_DEMOS + NUM_FAILURE_DEMOS))
    NUM_DEMOS=$((NUM_DEMOS + NUM_FAILURE_DEMOS))

    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=${DEVICES} python -m bpref_v2.reward_learning.train_reds \
        --comment ${TASK_NAME}-phase$((i + 1)) \
        --robot.benchmark ${TASK_TYPE} \
        --robot.data_dir ${BASE_PATH}/${TASK_TYPE}_data/${TASK_NAME} \
        --reds.embd_dim 512 \
        --reds.output_embd_dim 512 \
        --reds.n_layer 1 \
        --reds.n_head 4 \
        --reds.lr 5e-5 \
        --logging.output_dir ${REWARD_MODEL_PATH} \
        --batch_size=16 \
        --seed 0 \
        --model_type REDS \
        --early_stop True \
        --robot.task_name ${TASK_NAME} \
        --robot.num_demos ${NUM_DEMOS} \
        --env "${TASK_TYPE}-${TASK_NAME}" \
        --robot.window_size=4 \
        --robot.skip_frame=1 \
        --robot.image_keys='image' \
        --reds.lambda_supcon=1.0 \
        --reds.lambda_epic=1.0 \
        --reds.epic_on_neg_batch=True \
        --reds.supcon_on_neg_batch=True \
        --use_failure=True \
        --reds.transfer_type clip_vit_b16 \
        --augmentations "crop|jitter" \
        --robot.num_workers=2 \
        --robot.output_type raw \
        --logging.online True \
        --train_steps=${TRAINING_STEPS} \
        --num_failure_demos=${NUM_FAILURE_DEMOS}

    echo "ITERATION ${i} DONE, NUM_DEMOS ${NUM_DEMOS}"
done
