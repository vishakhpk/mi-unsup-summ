export TRAIN_FILE=./path/to/finetune-lm-train
export TEST_FILE=./path/to/finetune-lm-eval

python run_language_modeling.py \
    --output_dir=output_large \
    --model_type=gpt2-large \
    --model_name_or_path=gpt2-large \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE
