from html import entities
from transformers import AutoTokenizer, AutoModel 
from transformers import AutoConfig, AutoModelForSequenceClassification
import fire
import os
import sys
sys.path.append("../") 
from sap.sap_eval import load_queries, load_dictionary
from sap.model_wrapper import (
    Model_Wrapper
)
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english') + [str(i) for i in range(10)]


def get_mention_id(mention, model_wrapper, dict_dense_embeds, eval_dictionary, topk = 1):
    mention_dense_embeds = model_wrapper.embed_dense(names=[mention], agg_mode='cls')

    # get score matrix
    dense_score_matrix = model_wrapper.get_score_matrix(
            query_embeds=mention_dense_embeds, 
            dict_embeds=dict_dense_embeds,
    )
    score_matrix = dense_score_matrix

    candidate_idxs = model_wrapper.retrieve_candidate_cuda(
            score_matrix = score_matrix, 
            topk = 1,
            batch_size=16,
            show_progress=False
    )
    np_candidates = [eval_dictionary[ind] for ind in candidate_idxs[0].tolist()]
    return np_candidates



def main(task, user_mode, outdir = "output/exps", bsz = 256, part = "test", avoidoverlap = True):
    model_path = os.path.join(outdir, f"{task}__{user_mode}")
    try:
        #tokenizer = AutoTokenizer.from_pretrained(model_path)  
        #model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
        user_mode_str = user_mode
        user_mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None for e in (user_mode[0].split(',') if type(user_mode) is not str else user_mode.split(',')) }
        model_wrapper = Model_Wrapper(user_mode).load_model(
            path=model_path,
            max_length=25,
            use_cuda=True,
        )
        tokenizer = model_wrapper.get_dense_tokenizer()
    except Exception as e:
        print(e)
        return 
    if "joint_dbg" in user_mode:
        eval_dictionary = load_dictionary(dictionary_path="tools/sapbert/evaluation/data/ncbi-disease/test_dictionary.txt")
        dict_names = [row[0] for row in eval_dictionary]
        dict_dense_embeds = model_wrapper.embed_dense(names=dict_names, show_progress=True, agg_mode="cls")
    if "pos" in user_mode:
        if user_mode['pos'] == "umls":
            umls_gold = set([l.split("||")[-1].strip() for l in open("datasets/sapbert/data/ncbi-disease/test_dictionary.txt").readlines()]) - set(STOPWORDS)
        elif user_mode['pos'] == "umlsdi":
            umls_gold = set([l.split("||")[-1].strip().lower() for l in open("datasets/umls/umlsdi.txt").readlines()]) - set(STOPWORDS)
        

    import json
    if not os.path.exists(f"output/preprocessed/{task}/{part}.json"):
        print(f"File not Exists: output/preprocessed/{task}/{part}.json")
        return
    test = [json.loads(line) for line in open(f"output/preprocessed/{task}/{part}.json").readlines()]
    lens = []
    all_spans = []
    for sample in test:
        spans = []
        sample["src"] = " ".join(sample["src"].split())#.replace("-", "")
        if "mutation is an unstable" in sample["src"]:
            a = 1
        for gram in range(1, 11):
            words = sample["src"].split()
            for i in range(len(words)):
                if i + gram <= len(words):
                    spans.append(" ".join(words[i : i + gram]))
        lens.append(len(spans))
        all_spans += spans

    from tqdm.auto import tqdm
    import numpy as np
    all_reps = []
    model_wrapper.scmodel.cuda()
    for i in tqdm(np.arange(0, len(all_spans), bsz)):
        toks = tokenizer.batch_encode_plus(all_spans[i:i+bsz], 
                                        padding="max_length", 
                                        max_length=25, 
                                        truncation=True,
                                        return_tensors="pt")
        toks_cuda = {}
        for k,v in toks.items():
            toks_cuda[k] = v.cuda()
        output = model_wrapper.scmodel(**toks_cuda)
        
        
        #output = model(**toks)
        #cls_rep = output.logits[:, 1]
        cls_rep = output.logits.softmax(1)[:, 1]
        
        all_reps.append(cls_rep.cpu().detach().numpy())
    all_scores = np.concatenate(all_reps, axis=0)


    st = 0
    results = []
    for idx in tqdm(range(len(lens))):
        remain_sent = test[idx]["src"]
        span_scores = all_scores[st : st + lens[idx]]
        spans = all_spans[st : st + lens[idx]]
        if "edge" in user_mode:
            match = [[s if s < 0.99 else 1.0, p] for s, p in zip(span_scores, spans) if s > 0.5 ]#or p.lower() in umls_gold
        else:
            match = [[s, p] for s, p in zip(span_scores, spans) if s > 0.5]
        match = sorted(match, key=lambda x: (-x[0], -len(x[1])))
        #match = sorted(match)[::-1]
        filter_match = []
        if avoidoverlap:
            for s, m in match:
                if m == "neisserial infection":
                    a = 1
                #for m in sorted(match, key=lambda x: len(x))[::-1]:
                if m in remain_sent:
                    if "joint_dbg" in user_mode:
                        mid = get_mention_id(m, model_wrapper, dict_dense_embeds, eval_dictionary)
                        filter_match.append(f"{m} [ID] {mid[0][1]} [DictName] {mid[0][0]}")
                    else:
                        filter_match.append(m)
                    remain_sent = remain_sent.replace(m, "")
        else:
            filter_match = [m for s, m in match]
        results.append(" [SEP] ".join(filter_match))
        st += lens[idx]
    open(os.path.join(outdir, f"{task}__{user_mode_str}/generated_predictions.txt"), "w").write("\n".join(results))

if __name__ == "__main__":
    fire.Fire(main)