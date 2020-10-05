#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

TRAIN_URL='http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz'
VALID_URL='http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
TEST_URL='http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz'
TRAIN_GZ=training.tar.gz
VALID_GZ=validation.tar.gz
TEST_GZ=mmt_task1_test2016.tar.gz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=de
lang=en-de
prep=data/multi30k.tokenized.en-de
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$TRAIN_URL"
wget "$VALID_URL"
wget "$TEST_URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $TRAIN_GZ
tar zxvf $VALID_GZ
tar zxvf $TEST_GZ
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.$l
    tok=train.tok.$l

    cat $orig/$f | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tok $src $tgt $tmp/train.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tok.$l > $tmp/train.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    f=val.$l
    cat $orig/$f | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $tmp/$f
    f=test2016.$l
    cat $orig/$f | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $tmp/$f
    echo ""
    echo ""
done

TRAIN=$tmp/train.en-de
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L val.$L test2016.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
TEXT=data/multi30k.tokenized.en-de
fairseq-preprocess --source-lang en --target-lang de \
	--trainpref $TEXT/train --validpref $TEXT/val --testpref $TEXT/test2016 \
	--destdir data-bin/multi30k.tokenized.en-de \
	--workers 20
rm -rf $orig
