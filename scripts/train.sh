export CUDA_VISIBLE_DEVICES="0,1,2,3"
gpus=$(echo ${CUDA_VISIBLE_DEVICES:-""} | tr ',' '\n' | wc -l)


export WANDB_MODE=offline
single_card=False


exp_dir="./exps"
seed=31415926
lr=5e-5

alpha=1
beta=1

config_name=finelap_eat_config
data_config_name=data_eat
epochs=10

config="./config/finelap_eat_config.yaml"
data_config="./config/data_config/data_eat.yaml"

bsz=128
exp_name=FineLAP_EAT_ALL_gpus${gpus}_bsz${bsz}_seed${seed}_epoch${epochs}
MASTER_PORT=$((10000 + RANDOM % 50000))

# -m debugpy --listen 4567 --wait-for-client 
# debug
if [ $single_card == True ]; then
python "train.py" \
    --config "$config" \
    --data_config "$data_config" \
    --exp_dir "$exp_dir" \
    --exp_name "$exp_name" \
    --seed "$seed" \
    --lr ${lr} \
    --epochs "$epochs" \
    --alpha ${alpha} \
    --beta ${beta} \
    --batch_size ${bsz} \
    # --use_wandb
elif [ $single_card == False ]; then
torchrun --nproc_per_node="$gpus" --master_port="$MASTER_PORT" "train.py" \
    --config "$config" \
    --data_config "$data_config" \
    --exp_dir "$exp_dir" \
    --exp_name "$exp_name" \
    --seed "$seed" \
    --lr ${lr} \
    --epochs "$epochs" \
    --alpha ${alpha} \
    --beta ${beta} \
    --batch_size ${bsz} \
    # --use_wandb 
fi
