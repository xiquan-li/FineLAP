export CUDA_VISIBLE_DEVICES=0


config="./config/finelap_eat_config.yaml"
ckpt_path="./weights/finelap_fixed.pt"
audio_path="./resources/1.wav"

python infer.py \
    --config config/finelap_eat_config.yaml \
    --ckpt_path $ckpt_path \
    --audio_path $audio_path \
    --caption "A woman speaks, dishes clanking, food frying, and music plays" \
    --phrases Speech Dishes Frying Music Dog Cat Thunder Car \
    --output_dir output
