#!/bin/bash
TASK=bc4chemd
MODE="edge,pos=umlsc,neg=sapc,ppreg=1.0,pndreg=0.1"
scripts/train_syngen.sh $TASK $MODE
export PYTHONPATH=src/sap:src
python ./scripts/test_ner.py $TASK $MODE
python ./scripts/eval_ner.py $TASK $MODE 