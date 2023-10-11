import fire
import json
import sys
import numpy as np
import re
import os
from collections import Counter

GEN_FILE = "generated_predictions.txt" 
#GEN_FILE = "retrieved_snomedid.txt"

def parse_tgt(tgts):
    res = []
    for tgt in tgts:
        res.append([[t.strip() for t in ps.split("[SEP]")] for ps in tgt.split("[SEP]")])
    return res


def main(dataset, mode, out_dir = "output/exps"): # nerel
    if type(mode) is not str:
        mode = ",".join(mode)
    exp = f"{dataset}__{mode}"
    print(f"Evaluating {exp}", file=sys.stderr)
    if not os.path.exists(f"{out_dir}/{exp}/{GEN_FILE}"):
        print(f"{out_dir}/{exp}/{GEN_FILE}")
        return
    gold =  [json.loads(line) for line in open(f"output/preprocessed/{dataset}/test.json").readlines()]
    mention_gold = [list([p.replace("[EOS]", "").strip() for p in s['tgt'].split('[SEP]')]) for s in gold]
    mention_gold = [[mm for mm in m if len(mm) > 0] for m in mention_gold]
    srcs = [s['src'] for s in gold]
    #(SEP), [SEPs], [SEAP], [SEPK], [SEp], [SEPS], [sept], (sep), [se p]
    #re.sub(r"[\(\[][Ss ][Ee ][Aa ]?[Pp ]?[Tt ]?[Ss ]?[Kk ]?[Ff ]?[\]\)]", "[SEP]", line)
    gen = [list([t.replace("[EOS]", "").strip() for t in line.split("[SEP]")]) for i, line in enumerate(open(f"{out_dir}/{exp}/{GEN_FILE}").readlines())]
    
    
    error_predict = []
    missing = []
    tot_pred, tot_gold = 0, 0
    for men, gol in zip(gen, mention_gold):
        for m in men:
            tot_pred += 1
            if m not in gol and len(m) > 0:# and m not in umls_gold:
                error_predict.append(m)
        for g in gol:
            tot_gold += 1
            if g not in men:
                missing.append(g)
    P = 1 - len(error_predict) / tot_pred
    R = 1 - len(missing) / tot_gold
    F1 = 2*P*R/(P + R)
    res = {
        "P" : round(np.mean(P), 3),
        "R" : round(np.mean(R), 3),
        "F1" : round(np.mean(F1), 3),
    }
    res = json.dumps(res, indent=2)
    print(res)
    open(f"{out_dir}/{exp}/eval_score.json", "w").write(res)
    ecounter = Counter(error_predict)
    print(f"error_predict: {len(error_predict)} / {tot_pred}, {ecounter.most_common(20)}")
    mcounter = Counter(missing)
    print(f"missing: {len(missing)} / {tot_gold}, {mcounter.most_common(20)}")
    json.dump({"Error":[f"{key} : {value}" for key,value in ecounter.items()], "Missing" : [f"{key} : {value}" for key,value in mcounter.items()]}, open(f"{out_dir}/{exp}/errors.json", "w"), indent=2)


if __name__ == "__main__":
    #main("ncbi", "dbg")
    fire.Fire(main)