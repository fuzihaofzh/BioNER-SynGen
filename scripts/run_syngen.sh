#!/bin/bash


TASK=$1
MODE=$2
EXP="$TASK"__"$MODE"


PATH_TO_TRAIN_FILE=datasets/training_file_umls2020aa_en_uncased_no_dup_pairwise_pair_th50.txt

if [ "$DBG" = 1 ]; then
    export DBG=' -m ptvsd --host 0.0.0.0 --port 5678 --wait '
	EXP=dbg
	BSZ=3
else
    BSZ=100
fi

echo EXP: $EXP



export CUDA_VISIBLE_DEVICES=1
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES


export PYTHONPATH=src/sap:src

python $DBG src/train_syngen.py \
	--model_dir "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
	--train_dir $PATH_TO_TRAIN_FILE \
	--output_dir output/exps/$EXP \
	--use_cuda \
	--epoch 1 \
	--train_batch_size $BSZ \
	--learning_rate 2e-5 \
	--max_length 25 \
	--amp \
	--pairwise \
	--random_seed 33 \
	--loss ms_loss \
	--use_miner \
	--type_of_triplets "all" \
	--miner_margin 0.2 \
	--agg_mode "cls" \
	--user_mode $MODE \
	--task $TASK \
	--save_checkpoint_all




