import fire
import json
from tqdm.auto import tqdm
import os

from nltk.corpus import stopwords as nltkstopwords
import string
stopwords = nltkstopwords.words('english')
stopwords += [*(string.ascii_lowercase + string.ascii_uppercase)]
stopwords += [str(a) for a in range(10)] + ['-']
stopwords = set(stopwords)

dict_map = {
    "ncbi" : "umlsdi",
    "bc5cdr-d" : "umlsdi",
    "pubd" : "umlsdi",
    "bc5cdr-c" : "umlsc",
    "bc4chemd" : "umlsc",
    "pubc" : "umlsc",
    "s800" : "umls-flt-s800",
    "linnaeus" : "umls-flt-linnaeus",
}

import multiprocessing
from Levenshtein import ratio, setratio
ratio_thres = 0.95
def get_ratio(span_lst):
    return [ratio(spans[0], spans[1], score_cutoff=ratio_thres) for spans in span_lst]

def get_chunks(lst, n):
    l = len(lst) // (n - 1)
    sp = list(range(0, len(lst), l)) + [len(lst)]
    return [lst[sp[i]: sp[i + 1]] for i in range(len(sp) - 1)]


def main(task = "ncbi", mode = "wordmatch", part = "test"):
    user_mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None for e in (mode[0].split(',') if type(mode) is not str else mode.split(',')) }
    dict_name = dict_map[task]
    if "part" in user_mode:
        part = user_mode["part"]
    if "qumls" in user_mode:
        from quickumls import QuickUMLS
        threshold = 0.7
        if task in ["s800", "linnaeus"]:
            threshold = 0.7
        if user_mode["qumls"] != None:
            matcher = QuickUMLS(f"datasets/quickumls/{dict_name}", similarity_name = user_mode["qumls"], window = 11, threshold = threshold)
        else:
            matcher = QuickUMLS(f"datasets/quickumls/{dict_name}", window = 11, threshold = threshold)
        matcher.match("Candida tropicalis", best_match=False, ignore_syntax=False)
    if "strsim" in user_mode:
        from difflib import SequenceMatcher        
    if not os.path.exists(f"output/preprocessed/{task}/{part}.json"):
        print(f"File not Exists: output/preprocessed/{task}/{part}.json")
        return
    test = [json.loads(line) for line in open(f"output/preprocessed/{task}/{part}.json").readlines()]
    biodict = set([l.split("||")[-1].strip() for l in open(f"datasets/umls/{dict_name}.txt").readlines()])
    if "lower" in user_mode:
        biodict = set([s.lower() for s in biodict]) - stopwords
    if "pportion" in user_mode:
        lbiodict = list(biodict)
        biodict = set(lbiodict[:int(len(lbiodict) * float(user_mode["pportion"]))])
    lens = []
    all_spans = []
    for sample in tqdm(test):
        spans = []
        sample["src"] = " ".join(sample["src"].split())#.replace("-", "")
        if "qumls" in user_mode:
            res = list(set([(r['start'], r['end'], r['ngram']) for rr in matcher.match(sample["src"], best_match=False, ignore_syntax=False) for r in rr]))
            spans = [s[2] for s in res]
        else:
            for gram in range(1, 11):
                words = sample["src"].split()
                for i in range(len(words)):
                    if i + gram <= len(words):
                        w = " ".join(words[i : i + gram])
                        if w == "breast cancer":
                            a = 1
                        if "wordmatch" in user_mode:
                            if w in biodict or ("lower" in user_mode and w.lower() in biodict):
                                spans.append(w)
                        elif "strsim" in user_mode:
                            if max([SequenceMatcher(None, w, d).ratio() for d in biodict]) > 0.8:
                                spans.append(w)
                        elif "lev" in user_mode:
                            #pool = multiprocessing.Pool(10)
                            #chunks = get_chunks([(w, d) for d in biodict], 10)
                            #ratios = pool.map(get_ratio, chunks)
                            #if max(sum(ratios, [])) > ratio_thres:
                            if max([ratio(w, d, score_cutoff=0.95) for d in biodict]) > 0.95:
                                spans.append(w)
        spans = list(set(spans))
        all_spans.append(" [SEP] ".join(spans) + " [EOS]")
    os.system(f"mkdir -p output/sapbert/{task}__{mode}")
    if part == "train":
        open(f"output/sapbert/{task}__{mode}/generated_predictions_train.txt", "w").write("\n".join(all_spans))
    else:
        open(f"output/sapbert/{task}__{mode}/generated_predictions.txt", "w").write("\n".join(all_spans))

if __name__ == "__main__":
    fire.Fire(main)