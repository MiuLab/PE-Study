export TRAIN_FILE=/tmp2/wikitext-103-raw/wiki.train.raw
export TEST_FILE=/tmp2/wikitext-103-raw/wiki.test.raw
export OUTPUT=/tmp2/output/wikitext103

python3 run_language_modeling.py \
    --output_dir=$OUTPUT/gpt2.gpt2_pe.l128.fix \
    --model_type=gpt2 \
	--tokenizer_name=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
	--per_gpu_train_batch_size=16 \
	--per_gpu_eval_batch_size=16 \
	--gradient_accumulation_steps=2 \
	--num_train_epochs=3 \
	--save_steps=5000 \
	--logging_steps=5000 \
	--evaluate_during_training \
	--eval_all_checkpoints \
	--block_size=128 \
	--pe_type=gpt2 \
	--learning_rate=2.5e-4 \
	--warmup_steps=4000 \

python3 run_language_modeling.py \
    --output_dir=$OUTPUT/gpt2.gpt2_pe.skip.fix \
    --model_type=gpt2 \
	--tokenizer_name=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
	--per_gpu_train_batch_size=16 \
	--per_gpu_eval_batch_size=16 \
	--gradient_accumulation_steps=2 \
	--num_train_epochs=3 \
	--save_steps=5000 \
	--logging_steps=5000 \
	--evaluate_during_training \
	--eval_all_checkpoints \
	--block_size=128 \
	--pe_type=gpt2 \
	--learning_rate=2.5e-4 \
	--warmup_steps=4000 \
	--skip_pe \

python3 run_language_modeling.py \
    --output_dir=$OUTPUT/bert.bert_pe.l128.fix \
    --model_type=bert \
	--tokenizer_name=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
	--per_gpu_train_batch_size=16 \
	--per_gpu_eval_batch_size=16 \
	--gradient_accumulation_steps=2 \
	--num_train_epochs=3 \
	--save_steps=5000 \
	--logging_steps=5000 \
	--evaluate_during_training \
	--eval_all_checkpoints \
	--block_size=128 \
	--pe_type=bert \
	--learning_rate=2.5e-4 \
	--warmup_steps=4000 \
    --mlm


python3 run_language_modeling.py \
    --output_dir=$OUTPUT/bert.bert_pe.skip.fix \
    --model_type=bert \
	--tokenizer_name=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
	--per_gpu_train_batch_size=16 \
	--per_gpu_eval_batch_size=16 \
	--gradient_accumulation_steps=2 \
	--num_train_epochs=3 \
	--save_steps=5000 \
	--logging_steps=5000 \
	--evaluate_during_training \
	--eval_all_checkpoints \
	--block_size=128 \
	--pe_type=bert \
	--learning_rate=2.5e-4 \
	--warmup_steps=4000 \
    --mlm \
	--skip_pe

python3 run_language_modeling.py \
    --output_dir=$OUTPUT/bert.roberta_pe.l128.fix \
    --model_type=bert \
	--tokenizer_name=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
	--per_gpu_train_batch_size=16 \
	--per_gpu_eval_batch_size=16 \
	--gradient_accumulation_steps=2 \
	--num_train_epochs=3 \
	--save_steps=5000 \
	--logging_steps=5000 \
	--evaluate_during_training \
	--eval_all_checkpoints \
	--block_size=128 \
	--pe_type=roberta \
	--learning_rate=2.5e-4 \
	--warmup_steps=4000 \
    --mlm


python3 run_language_modeling.py \
    --output_dir=$OUTPUT/bert.roberta_pe.skip.fix \
    --model_type=bert \
	--tokenizer_name=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
	--per_gpu_train_batch_size=16 \
	--per_gpu_eval_batch_size=16 \
	--gradient_accumulation_steps=2 \
	--num_train_epochs=3 \
	--save_steps=5000 \
	--logging_steps=5000 \
	--evaluate_during_training \
	--eval_all_checkpoints \
	--block_size=128 \
	--pe_type=roberta \
	--learning_rate=2.5e-4 \
	--warmup_steps=4000 \
    --mlm \
	--skip_pe
