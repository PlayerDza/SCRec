

CUDA_VISIBLE_DEVICES=3 accelerate launch  --mixed_precision fp16 rating.py \
    --dataset amazon \
    --index 1 \
    --base_model facebook/opt-2.7B\
    --lambdat 0.2\
    --batch_size 64\
    --log_interval 1000\
    --prefix 1 \
    --lr 0.001\
    --tokenlen 32


CUDA_VISIBLE_DEVICES=3 accelerate launch  --mixed_precision fp16 rating_test.py \
    --dataset amazon \
    --index 1 \
    --base_model facebook/opt-2.7B\
    --lambdat 0.2\
    --prefix 2 \
    --batch_size 64\
    --tokenlen 32

