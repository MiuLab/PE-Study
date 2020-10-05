export CUDA_VISIBLE_DEVICES=0
for enc_pe in bert gpt2; do
	dec_pe=random
	dir=checkpoints/multi30k.$enc_pe.$dec_pe
	mkdir -p $dir
	python3 fairseq-train.py \
		data-bin/multi30k.tokenized.en-de \
		--arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.1 --weight-decay 0.0001 --fp16 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 2048 \
		--log-interval 1000 \
		--no-progress-bar \
		--decoder-learned-pos \
		--encoder-learned-pos \
		--max-epoch 40 \
		--encoder-pe $enc_pe \
		--decoder-pe $dec_pe \
		--save-dir $dir \
		--seed 7122 | tee $dir/logs
done
for dec_pe in bert gpt2; do
	enc_pe=random
	dir=checkpoints/multi30k.$enc_pe.$dec_pe
	mkdir -p $dir
	python3 fairseq-train.py \
		data-bin/multi30k.tokenized.en-de \
		--arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.1 --weight-decay 0.0001 --fp16 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 2048 \
		--log-interval 1000 \
		--no-progress-bar \
		--decoder-learned-pos \
		--encoder-learned-pos \
		--max-epoch 40 \
		--encoder-pe $enc_pe \
		--decoder-pe $dec_pe \
		--save-dir $dir \
		--seed 7122 | tee $dir/logs
done
for pe in random bert gpt2; do
	dir=checkpoints/multi30k.$pe.$pe
	mkdir -p $dir
	python3 fairseq-train.py \
		data-bin/multi30k.tokenized.en-de \
		--arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.1 --weight-decay 0.0001 --fp16 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 2048 \
		--log-interval 1000 \
		--no-progress-bar \
		--decoder-learned-pos \
		--encoder-learned-pos \
		--max-epoch 40 \
		--encoder-pe $pe \
		--decoder-pe $pe \
		--save-dir $dir \
		--seed 7122 | tee $dir/logs
done
enc_pe=bert
dec_pe=gpt2
dir=checkpoints/multi30k.$enc_pe.$dec_pe
mkdir -p $dir
python3 fairseq-train.py \
	data-bin/multi30k.tokenized.en-de \
	--arch transformer_iwslt_de_en --share-decoder-input-output-embed \
	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--dropout 0.1 --weight-decay 0.0001 --fp16 \
	--criterion cross_entropy \
	--max-tokens 2048 \
	--log-interval 1000 \
	--no-progress-bar \
	--decoder-learned-pos \
	--encoder-learned-pos \
	--max-epoch 40 \
	--encoder-pe $enc_pe \
	--decoder-pe $dec_pe \
	--save-dir $dir \
	--seed 7122 | tee $dir/logs
enc_pe=gpt2
dec_pe=bert
dir=checkpoints/multi30k.$enc_pe.$dec_pe
mkdir -p $dir
python3 fairseq-train.py \
	data-bin/multi30k.tokenized.en-de \
	--arch transformer_iwslt_de_en --share-decoder-input-output-embed \
	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--dropout 0.1 --weight-decay 0.0001 --fp16 \
	--criterion cross_entropy \
	--max-tokens 2048 \
	--log-interval 1000 \
	--no-progress-bar \
	--decoder-learned-pos \
	--encoder-learned-pos \
	--max-epoch 40 \
	--encoder-pe $enc_pe \
	--decoder-pe $dec_pe \
	--save-dir $dir \
	--seed 7122 | tee $dir/logs
enc_pe=sinusoid
dec_pe=sinusoid
dir=checkpoints/multi30k.$enc_pe.$dec_pe
mkdir -p $dir
python3 fairseq-train.py \
	data-bin/multi30k.tokenized.en-de \
	--arch transformer_iwslt_de_en --share-decoder-input-output-embed \
	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--dropout 0.1 --weight-decay 0.0001 --fp16 \
	--criterion cross_entropy \
	--max-tokens 2048 \
	--log-interval 1000 \
	--no-progress-bar \
	--decoder-learned-pos \
	--encoder-learned-pos \
	--max-epoch 40 \
	--encoder-pe $enc_pe \
	--decoder-pe $dec_pe \
	--save-dir $dir \
	--seed 7122 | tee $dir/logs
