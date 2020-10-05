export TRAIN_FILE=/tmp2/wikitext-103-raw/wiki.train.raw
export TEST_FILE=/tmp2/wikitext-103-raw/wiki.test.raw

python3 run_language_modeling.py \
    --output_dir=/tmp2/output/roberta \
    --model_type=roberta \
	--tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=$TRAIN_FILE \
	--per_gpu_train_batch_size=16 \
	--per_gpu_eval_batch_size=16 \
	--gradient_accumulation_steps=16 \
	--max_steps=30000 \
	--warmup_steps=5000 \
	--save_steps=2500 \
	--logging_steps=2500 \
	--block_size=128 \
	--learning_rate=1e-4 \
    --mlm

