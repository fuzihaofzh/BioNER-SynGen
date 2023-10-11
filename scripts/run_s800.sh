#!/bin/bash
TASK=s800
MODE="edge,pos=umls-scn,neg=umls-scn,ppreg=0.5"
#scripts/train_syngen.sh $TASK $MODE
export PYTHONPATH=src/sap:src
#python ./scripts/test_ner.py $TASK $MODE
python ./scripts/eval_ner.py $TASK $MODE 