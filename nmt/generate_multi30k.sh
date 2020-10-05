for p in `ls -d checkpoints/multi30k.*`; do
	echo $p
	fairseq-generate data-bin/multi30k.tokenized.en-de \
		--path $p/checkpoint_best.pt \
		--batch-size 64 --beam 5 --remove-bpe --no-progress-bar > $p/generate.txt
	tail -n 1 $p/generate.txt
done
