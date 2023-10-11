#!/bin/bash
TASK=linnaeus
MODE="edge,pos=umls-scn,neg=umls-scn,ppreg,pndreg=0.2"
scripts/train_syngen.sh $TASK $MODE
export PYTHONPATH=src/sap:src
python ./scripts/test_ner.py $TASK $MODE
python ./scripts/eval_ner.py $TASK $MODE 