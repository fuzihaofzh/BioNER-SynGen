#!/bin/bash
TASK=bc5cdr-d
MODE="edge,pos=umlsdi,neg=sap,ppreg=1.0,pndreg=0.1"
scripts/train_syngen.sh $TASK $MODE
export PYTHONPATH=src/sap:src
python ./scripts/test_ner.py $TASK $MODE
python ./scripts/eval_ner.py $TASK $MODE 