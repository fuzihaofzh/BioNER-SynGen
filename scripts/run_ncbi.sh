#!/bin/bash
TASK=ncbi
MODE="edge,pos=umlsdi,neg=sap,ppreg,pndreg=0.1,ptm=pubmedbert"
scripts/train_syngen.sh $TASK $MODE
export PYTHONPATH=src/sap:src
python ./scripts/test_ner.py $TASK $MODE
python ./scripts/eval_ner.py $TASK $MODE 