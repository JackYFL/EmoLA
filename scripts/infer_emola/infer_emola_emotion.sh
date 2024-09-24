export CUDA_VISIBLE_DEVICES=2
TASK="emotion"
DATASET="FABA"

MODEL_PATH="./saved_checkpoints/llava-v1.5-7b-lora-emola-emotion"
ANN_FILE="./data/FABAInstruct/eval/eval_emotion_anno.json"

python emollava/eval/eval_FABA.py \
    --model-path $MODEL_PATH \
    --task-type $TASK \
    --dataset $DATASET \
    --ann-file $ANN_FILE \
    --extra-name '' \
