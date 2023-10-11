from transformers import AutoTokenizer, AutoModel  
import numpy as np
from tqdm.auto import tqdm
import json
import os
import time
import fire
from scipy.spatial.distance import cdist
import faiss
import torch

from nltk.corpus import stopwords as nltkstopwords
import string
stopwords = nltkstopwords.words('english')
stopwords += [*(string.ascii_lowercase + string.ascii_uppercase)]
stopwords += [str(a) for a in range(10)]
stopwords = set(stopwords)

dict_map = {
    "ncbi" : "umlsdi",
    "bc5cdr-d" : "umlsdi",
    "bc5cdr-c" : "umlsc",
    "bc4chemd" : "umlsc",
    "s800" : "umls-flt-s800",
    "linnaeus" : "umls-flt-linnaeus",
}


#task = "ncbi"
#dict_name = "umlsdi"

def get_span_emb(all_spans, model, tokenizer, agg = "first"):
    all_reps = []
    bs = 128
    for i in tqdm(np.arange(0, len(all_spans), bs)):
        toks = tokenizer.batch_encode_plus(all_spans[i:i+bs], 
                                        padding="max_length", 
                                        max_length=25, 
                                        truncation=True,
                                        return_tensors="pt")
        toks_cuda = {}
        for k,v in toks.items():
            toks_cuda[k] = v.cuda(0)
        output = model(**toks_cuda)
        
        #output = model(**toks)
        if agg == "first":
            cls_rep = output[0][:,0,:]
        elif agg == "mean":
            cls_rep = output[0].mean(1)
        
        all_reps.append(cls_rep.cpu().detach().numpy())
    all_spans_emb = np.concatenate(all_reps, axis=0)
    return all_spans_emb

def search_cdist(query_cls_rep, all_dict_emb):
    dist = cdist(query_cls_rep, all_dict_emb)
    #nn_index = dist.argmin(1)
    min_dist = dist.min(1)
    return min_dist

def search_faiss(query_cls_rep, dict_index):
    return dict_index.search(query_cls_rep, 1)[0]

def main(task = "ncbi", mode = "vectormatch,ptm=biobert"):
    dict_name = dict_map[task]
    user_mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None for e in (mode[0].split(',') if type(mode) is not str else mode.split(',')) }
    outdir = f"{task}__{mode}"
    print(task, dict_name)
    test = [json.loads(line) for line in open(f"output/preprocessed/{task}/test.json").readlines()]

    def calc_emb():
        agg = "first"
        if "ptm" in user_mode:
            if user_mode["ptm"] == "sapbert":
                tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
                model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
                thres = 6
            elif user_mode["ptm"] == "biobert":
                tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")  
                model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2").cuda()
                thres = 4
            elif user_mode["ptm"] == "pubmedbert":
                tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")  
                model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext").cuda()
                thres = 3
            elif user_mode["ptm"] == "word2vec":
                tokenizer = AutoTokenizer.from_pretrained("nicoladecao/msmarco-word2vec256000-bert-base-uncased")  
                model = AutoModel.from_pretrained("nicoladecao/msmarco-word2vec256000-bert-base-uncased").cuda()
                thres = 3
                agg = "mean"
        
        lens = []
        all_spans = []
        span_max_len = 11
        if task == "bc4chemd":
            span_max_len = 11
        for sample in test:
            spans = []
            for gram in range(1, span_max_len):
                words = sample["src"].split()
                for i in range(len(words)):
                    if i + gram < len(words):
                        spans.append(" ".join(words[i : i + gram]))
            lens.append(len(spans))
            all_spans += spans
        biolist = list(set([l.split("||")[-1].strip() for l in open(f"datasets/umls/{dict_name}.txt").readlines()]))
        if "tfidf" in user_mode:
            from sklearn.feature_extraction.text import TfidfVectorizer
            """tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
            def tokenize_seq(txts):
                toks = tokenizer.batch_encode_plus(txts, 
                                        padding="max_length", 
                                        max_length=25, 
                                        truncation=True,
                                        return_tensors="pt")
                w2i = tokenizer.get_vocab()
                i2w = {w2i[w] : w for w in w2i}
                return [i2w[tt] for t in toks['input_ids'] for tt in t.tolist() if tt > 3]
            biolist = tokenize_seq(biolist)
            all_spans = tokenize_seq(all_spans)"""
            vectorizer = TfidfVectorizer(stop_words = None)
            if task == "bc4chemd": 
                vectorizer = TfidfVectorizer(stop_words = None, max_features = 20000)
            biolist = biolist
            X = vectorizer.fit_transform(biolist + all_spans)
            all_dict_emb = X[:len(biolist), :].todense().astype('float32')
            test_emb = X[len(biolist):, :].todense().astype('float32')
            thres = 0.7
        elif "vectormatch" in user_mode:
            test_emb = get_span_emb(all_spans, model, tokenizer, agg)
            all_dict_emb = get_span_emb(biolist, model, tokenizer, agg)
        
        if "tfidf" in user_mode:
            dict_index = faiss.IndexFlatIP(all_dict_emb.shape[1])
        else:
            dict_index = faiss.IndexFlatL2(all_dict_emb.shape[1])
        dict_index.add(all_dict_emb) 
        return lens, test_emb, all_spans, dict_index, thres
    lens, test_emb, all_spans, dict_index, thres = calc_emb()
    torch.cuda.empty_cache() 

    st = 0
    results = []
    for idx in tqdm(range(len(lens))):
        query_cls_rep = test_emb[st : st + lens[idx]]
        spans = all_spans[st : st + lens[idx]]
        #min_dist = search_cdist(query_cls_rep, all_dict_emb)
        min_dist = search_faiss(query_cls_rep, dict_index)
        if "tfidf" in user_mode:
            match0 = [[a, d] for d, a in zip(min_dist, all_spans[st : st + lens[idx]]) if d > thres]
            match = sorted(match0, key=lambda x: -x[1])
        else:
            match0 = [[a, d] for d, a in zip(min_dist, all_spans[st : st + lens[idx]]) if d < thres]
            match = sorted(match0, key=lambda x: x[1])
        #match = sorted(match)[::-1]
        filter_match = []
        avoidoverlap = True
        if avoidoverlap:
            remain_sent = test[idx]['src']
            for m, s in match:
                if m in remain_sent and m not in stopwords:
                    filter_match.append(m)
                    remain_sent = remain_sent.replace(m, "")
        else:
            filter_match = [m for s, m in match]
        results.append(" [SEP] ".join(filter_match))
        #results.append(" [SEP] ".join([a for n, a in zip(ner, all_spans[st : st + lens[idx]]) if n]))
        st = st + lens[idx]
    os.system(f"mkdir -p output/sapbert/{outdir}")
    open(f"output/sapbert/{outdir}/generated_predictions.txt", "w").write("\n".join(results))

if __name__ == "__main__":
    fire.Fire(main)