import re
import os
import glob
import numpy as np
import random
import random
import pandas as pd
import json
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import msgspec
LOGGER = logging.getLogger(__name__)
import multiprocessing
import time
import itertools
import sys


class QueryDataset_COMETA(Dataset):

    def __init__(self, data_dir, 
                load_full_sentence=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_composite={} filter_duplicate={}".format(
            data_dir, load_full_sentence, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            load_full_sentence=load_full_sentence,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, load_full_sentence, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        #concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        data_table = pd.read_csv(data_dir, sep='\t', encoding='utf8')

        for row in data_table.iterrows():
            mention = row[1]["Term"]
            sentence = row[1]["Example"]
            
            #print (mention)
            #print (sentence)

            cui = row[1]["General SNOMED ID"] # TODO: allow general/specific options
            if load_full_sentence: 
                data.append((mention, sentence, cui))
            else:
                data.append((mention, cui))

        if filter_duplicate:
            data = list(dict.fromkeys(data))
        print ("query size:",len(data))
        
        # return np.array data
        data = np.array(data)
        
        return data

class QueryDataset_COMETA_user(Dataset):

    def __init__(self, data_dir, 
                load_full_sentence=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_composite={} filter_duplicate={}".format(
            data_dir, load_full_sentence, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            load_full_sentence=load_full_sentence,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, load_full_sentence, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        #concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        data_table = pd.read_csv(data_dir, sep='\t', encoding='utf8')

        def parse_tgt(tgts):
            res = []
            for tgt in tgts:
                res.append([[t.strip() for t in ps.split("[SEP]")] for ps in tgt.split("[EOS]") if len(ps.split("[SEP]")) == 2])
            return res
        gen = parse_tgt(open("/gds/zhfu/workbench/explore/bio_nerel/output/exps/cometa_wiki_c2m_mention__bart/generated_predictions.txt").readlines())
        data_table["Term"] = [g[0][1] for g in gen]

        for row in data_table.iterrows():
            mention = row[1]["Term"]
            sentence = row[1]["Example"]
            
            #print (mention)
            #print (sentence)

            cui = row[1]["General SNOMED ID"] # TODO: allow general/specific options
            if load_full_sentence: 
                data.append((mention, sentence, cui))
            else:
                data.append((mention, cui))

        if filter_duplicate:
            data = list(dict.fromkeys(data))
        print ("query size:",len(data))
        
        # return np.array data
        data = np.array(data)
        
        return data

class QueryDataset_custom(Dataset):

    def __init__(self, data_dir, 
                load_full_sentence=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_duplicate={}".format(
            data_dir, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        with open(data_dir, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip("\n")
            if len(line.split("||")) == 2:
                _id, mention = line.split("||")
            elif len(line.split("||")) == 3: # in case using data with contexts
                _id, mention, context = line.split("||")
            else:
                raise NotImplementedError()
             
            data.append((mention, _id))

        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data, dtype=object)
        
        return data

class QueryDataset_pretraining(Dataset):

    def __init__(self, data_dir, 
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_duplicate={}".format(
            data_dir,filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        #concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        with open(data_dir, "r") as f:
            lines = f.readlines()

        for row in lines:
            row = row.rstrip("\n")
            snomed_id, mention = row.split("||")
            data.append((mention, snomed_id))

        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data)
        
        return data

class QueryDataset(Dataset):

    def __init__(self, data_dir, 
                filter_composite=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_composite={} filter_duplicate={}".format(
            data_dir, filter_composite, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_composite=filter_composite,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_composite, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        #concept_files = glob.glob(os.path.join(data_dir, "*.txt"))
        file_types = ("*.concept", "*.txt")
        concept_files = []
        for ft in file_types:
            concept_files.extend(glob.glob(os.path.join(data_dir, ft)))

        for concept_file in tqdm(concept_files):
            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()

            for concept in concepts:
                #print (concept)
                concept = concept.split("||")
                #if len(concept) !=5: continue
                mention = concept[3].strip().lower()
                cui = concept[4].strip()
                if cui.lower() =="cui-less": continue
                is_composite = (cui.replace("+","|").count("|") > 0)

                if filter_composite and is_composite:
                    continue
                else:
                    data.append((mention,cui))
        
        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data)
        
        return data


class DictionaryDataset():
    """
    A class used to load dictionary data
    """
    def __init__(self, dictionary_path):
        """
        Parameters
        ----------
        dictionary_path : str
            The path of the dictionary
        draft : bool
            use only small subset
        """
        LOGGER.info("DictionaryDataset! dictionary_path={}".format(
            dictionary_path 
        ))
        self.data = self.load_data(dictionary_path)
        
    def load_data(self, dictionary_path):
        name_cui_map = {}
        data = []
        with open(dictionary_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line == "": continue
                cui, name = line.split("||")
                name = name.lower()
                if cui.lower() == "cui-less": continue
                data.append((name,cui))
        
        #LOGGER.info("concerting loaded dictionary data to numpy array...")
        #data = np.array(data)
        return data

class MetricLearningDataset_pairwise(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer): #d_ratio, s_score_matrix, s_candidate_idxs):
        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        for line in lines:
            line = line.rstrip("\n")
            query_id, name1, name2 = line.split("||")
            self.query_ids.append(query_id)
            self.query_names.append((name1, name2))
        self.tokenizer = tokenizer
        self.query_id_2_index_id = {k: v for v, k in enumerate(list(set(self.query_ids)))}
    
    def __getitem__(self, query_idx):

        query_name1 = self.query_names[query_idx][0]
        query_name2 = self.query_names[query_idx][1]
        query_id = self.query_ids[query_idx]
        query_id = int(self.query_id_2_index_id[query_id])

        return query_name1, query_name2, query_id


    def __len__(self):
        return len(self.query_names)



class MetricLearningDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer): #d_ratio, s_score_matrix, s_candidate_idxs):
        LOGGER.info("Initializing metric learning data set! ...")
        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        cuis = []
        for line in lines:
            cui, _ = line.split("||")
            cuis.append(cui)

        self.cui2id = {k: v for v, k in enumerate(cuis)}
        for line in lines:
            line = line.rstrip("\n")
            cui, name = line.split("||")
            query_id = self.cui2id[cui]
            #if query_id.startswith("C"):
            #    query_id = query_id[1:]
            #query_id = int(query_id)
            self.query_ids.append(query_id)
            self.query_names.append(name)
        self.tokenizer = tokenizer
    
    def __getitem__(self, query_idx):

        query_name = self.query_names[query_idx]
        query_id = self.query_ids[query_idx]
        query_token = self.tokenizer.transform([query_name])[0]

        return torch.tensor(query_token), torch.tensor(query_id)

    def __len__(self):
        return len(self.query_names)


#################################################
class EdgeDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer): #d_ratio, s_score_matrix, s_candidate_idxs):
        with open(path, 'r') as f:
            lines = f.readlines()
        self.wikiwords = open("/gds/zhfu/workbench/explore/bio_nerel/datasets/wiki2/wiki.train.tokens").read().split()
        self.query_ids = []
        self.query_names = []
        for line in lines:
            line = line.rstrip("\n")
            query_id, name1, name2 = line.split("||")
            self.query_ids.append(query_id)
            self.query_names.append((name1, name2))
        self.tokenizer = tokenizer

    def __init__(self, path, tokenizer, user_mode, task): #dbg
        import json
        import nltk
        from nltk.corpus import stopwords
        import string
        self.stopwords = stopwords.words('english')
        self.stopwords += [*(string.ascii_lowercase + string.ascii_uppercase)]
        self.stopwords += [str(a) for a in range(10)]
        self.samples = [json.loads(line) for line in open(f"output/preprocessed/{task}/train.json").readlines()]
        self.src_words = [s["src"].split() for s in self.samples]
        self.tgt_words = [[t.strip() for t in s["tgt"].replace("[EOS]", "").strip().split("[SEP]") if t] for s in self.samples]
        self.all_term_words = sum(self.tgt_words, [])
        self.all_term_words_set = set(self.all_term_words)
        self.user_mode = user_mode
        if 'neg' in self.user_mode:
            if self.user_mode['neg'] == "wiki":
                self.wikiwords = open("datasets/wiki2/wiki.train.tokens").read().split()
            elif self.user_mode['neg'] == "pubmed":
                self.wikiwords = open("datasets/pubmed/pubmedt100ktkn.txt").read().split()
            elif self.user_mode['neg'] == "corpus":
                self.wikiwords = sum(self.src_words, [])
            elif self.user_mode['neg'] in ["sap", "sapc"]:
                self.sapnegs = [l.strip() for l in open(f"output/dev/{self.user_mode['neg']}_neg_samples7.txt").readlines()]
            elif self.user_mode['neg'].startswith("umls"):
                self.sapnegs = [l.strip() for l in open(f"output/dev/{self.user_mode['neg']}_neg_samples7.txt").readlines()]
            elif self.user_mode['neg'].startswith("wd-"):
                self.wikiwords = open(f"datasets/wikipedia/{self.user_mode['neg']}.txt").read().split()
        if 'pos' in self.user_mode:
            if self.user_mode['pos'] == "umls":
                self.samples = [l.split("||")[-1].strip() for l in open("datasets/sapbert/data/ncbi-disease/train_dictionary.txt").readlines()]
                self.all_term_words_set = set(self.samples)
            elif self.user_mode['pos'] in ["umlsdi", "umlssy", "umlsdidd", "umlsc"] or self.user_mode['pos'].startswith("umls-"):
                self.samples = [l.split("||")[-1].strip() for l in open(f"datasets/umls/{self.user_mode['pos']}.txt").readlines()]
                self.all_term_words_set = set(self.samples)
            elif self.user_mode['pos'] == "mesh":
                self.samples = [l.strip() for l in open("datasets/mesh/descterm.txt").readlines()]
                self.all_term_words_set = set(self.samples)
            elif self.user_mode['pos'] in ["ncbi", "ncbif"]:
                items = [json.loads(line) for line in open(f"output/preprocessed/ncbi/train.json").readlines()] + [json.loads(line) for line in open(f"output/preprocessed/ncbi/dev.json").readlines()]
                tgt_words = [[t.strip() for t in s["tgt"].replace("[EOS]", "").strip().split("[SEP]") if t] for s in items]
                self.samples = sum(tgt_words, [])
                self.all_term_words_set = set(self.samples)
                if self.user_mode['pos'] == "ncbif":
                    self.samples = list(self.all_term_words_set)
            elif self.user_mode['pos'].startswith("wd-"):
                self.samples = [l.split("||")[-1].strip() for l in open(f"datasets/wikidata/pos/{self.user_mode['pos']}.txt").readlines()]
        if 'pportion' in self.user_mode:
            self.samples = self.samples[:int(len(self.samples) * float(self.user_mode['pportion']))]
            self.all_term_words_set = set(self.samples)
        if 'ptrain' in self.user_mode:
            self.samples = self.samples[:int(len(self.samples) * float(self.user_mode['ptrain']))]
            self.all_term_words_set = set(self.samples)
        if 'addent' in self.user_mode:
            if self.user_mode['addent'] == 'wikidata':
                wikidata = [item['itemLabel'] for item in json.load(open("datasets/wikidata/query.json"))]
                self.samples += wikidata
        if 'enhance' in self.user_mode:
            if self.user_mode['enhance'] == "ncbi":
                items = [json.loads(line) for line in open(f"output/preprocessed/ncbi/train.json").readlines()] + [json.loads(line) for line in open(f"output/preprocessed/ncbi/dev.json").readlines()]
                tgt_words = [[t.strip() for t in s["tgt"].replace("[EOS]", "").strip().split("[SEP]") if t] for s in items]
                enhance_words = sum(tgt_words, [])
                self.samples += enhance_words
                self.all_term_words_set = set(self.samples)
        if 'dumpneg' in self.user_mode:
            self.dump_file = open("output/dev/dbg_neg.txt", "w")
        self.tokenizer = tokenizer
        self.task = task
        
    
    def __getitem__(self, query_idx):
        if 'pos' in self.user_mode:
            pos_term = [self.samples[query_idx]]
        elif len(self.tgt_words[query_idx]) > 0:
            #pos_term = random.sample(self.tgt_words[query_idx], 1)[0]
            pos_term = self.tgt_words[query_idx]
        else:
            pos_term = [random.sample(self.all_term_words, 1)[0]]
        
        negcnt = int(self.user_mode['negcnt']) if 'negcnt' in self.user_mode else 4
        res = []
        max_span_length = 8

        """
        swords = self.samples[query_idx]["src"].split()
        for i in range(len(swords)):
            for j in range(i + 1, min(len(swords), i + max_span_length)):
                neg_term = " ".join(swords[i : j])
                if neg_term not in self.all_term_words_set:
                    if random.random() < 0.05:
                        neg_term = random.sample("at was as As cancer cancers".split(), 1)[0]
                    res.append(neg_term)
        res = random.sample(res, min(10, len(res)))
        if "colorectal cancer" in res or "colorectal cancer" in pos_term:
            a = 1
        return pos_term, res, 1#"""


        for _ in range(negcnt):
            #"""
            while True:
                if 'neg' in self.user_mode:# and self.user_mode['neg'] in ["wiki", "pubmed"]:
                    if self.user_mode['neg'] in ["sap", "sapc"] or self.user_mode['neg'].startswith("umls"):
                        neg_term = random.choice(self.sapnegs)
                        res.append(neg_term)
                        if random.random() < 0.03:
                            res.append(random.sample(self.stopwords, 1)[0])
                        break
                    st0 = random.randint(0, len(self.wikiwords) - 60)
                    l1, l2 = random.randint(1, 20), random.randint(1, 20)
                    swords = self.wikiwords[st0 : st0 + l1] + random.choice(pos_term).split() + self.wikiwords[st0 + l1 : st0 + l1 + l2]
                    if 'nonegoverlap':
                        swords = self.wikiwords[st0 + l1 : st0 + l1 + l2 + 2]
                else:
                    swords = self.samples[query_idx]["src"].split()
            
                st = random.randint(0, len(swords) - 2)
                ed = random.randint(st + 1, min(len(swords), st + max_span_length))
                neg_term = " ".join(swords[st : ed])
                if neg_term not in self.all_term_words_set:
                    if random.random() < 0.03:
                        neg_term = random.sample(self.stopwords, 1)[0]
                    res.append(neg_term)
                    break
        if 'dumpneg' in self.user_mode:
            self.dump_file.write("\n".join(res))
        return pos_term, res, 1#"""

            
                

        """
            poswords = pos_term.split()
            swords = self.samples[query_idx]["src"].split()
            pi = pj = None
            for i in range(len(swords)):
                if swords[i : i + len(poswords)] == poswords:
                    pi, pj = i, i + len(poswords)
                    break
            while True:
                se = [random.randint(0, len(swords) - 1), random.randint(0, len(swords) - 1)]
                st, ed = min(se), max(se)
                if pj is None or not (ed < pi or pj < st):
                    neg_term = " ".join(swords[st : ed])
                    break

            if random.random() < 0.05:
                neg_term = random.sample(self.stopwords, 1)[0]
                neg_term = random.sample("at was as".split(), 1)[0]
            res.append(neg_term)
        
        return pos_term, res, 1#"""

        """pos_words = pos_name.split()
        negs = []
        while True:
            swords = random.sample(self.wikiwords, len(pos_words)) + pos_words + random.sample(self.wikiwords, len(pos_words))
            se = [random.randint(0, len(swords) - 1), random.randint(0, len(swords) - 1)]
            st, ed = min(se), max(se)
            if st < ed and swords[st : ed] != pos_words:
                negs.append(" ".join(swords[st : ed]))
            if len(negs) == 1:
                break
        return pos_name, negs[0], 1"""


    def __len__(self):
        return len(self.samples)



class GunerDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer, user_mode, task):
        self.positives = []
        self.negatives = []
        self.domains = []
        for line in open(f"output/preprocessed/guner/{task}/train.json").readlines():
            line = line.rstrip("\n")
            sample = json.loads(line)
            self.positives.append(sample["pos"])
            self.negatives.append(sample["neg"])
            self.domains.append(sample["domain"])
        if "dratio" in user_mode:
            dlen = int(len(self.positives) * float(user_mode["dratio"]))
            self.positives = self.positives[:dlen]
            self.negatives = self.negatives[:dlen]
            self.domains = self.domains[:dlen]
        if "neg" in user_mode:
            pos_set = set([s.lower() for s in self.positives])
            if user_mode["neg"] == "wiki2":
                negcorpus = open("datasets/wiki2/wiki.train.tokens").read().split()
            for i in range(len(self.negatives)):
                for j in range(len(self.negatives[i])):
                    while True:
                        l = len(self.negatives[i][j].split())
                        st0 = random.randint(0, len(negcorpus) - 60)
                        swords = " ".join(negcorpus[st0 : st0 + l])
                        if swords.lower() not in pos_set:
                            self.negatives[i][j] = swords
                            break
        if "cdnf" in user_mode:#cross domain negative filter
            pos_set = set([s.lower() for s in self.positives])
            for i in range(len(self.negatives)):
                for j in range(len(self.negatives[i])):
                    while self.negatives[i][j].lower() in pos_set:
                        self.negatives[i][j] = random.sample(self.negatives[i])

        self.tokenizer = tokenizer
        self.task = task
        self.user_mode = user_mode
    
    def __getitem__(self, query_idx):
        query_name1 = self.positives[query_idx]
        query_name2 = self.negatives[query_idx]
        domain_id = self.domains[query_idx]
        if "sel" in self.user_mode:
            domain_id = [domain_id] * (len(query_name2) + 1)
        return query_name1, query_name2, domain_id

    def __len__(self):
        return len(self.positives)

def make_domain_data(domain, user_mode, domain2id, max_span_length, negcnt, stopwords):
    #domain, user_mode, domain2id, max_span_length, negcnt, stopwords = params
    data = msgspec.json.decode(b"[%s]"%(b",".join(open(f"output/preprocessed/dinner/domain_data/{domain}.json", "rb").readlines())))
    positives = [d['term'] for d in data]
    negatives = []
    domains = []
    positives_set = set([p.lower() for p in positives])
    if "scaleup" in user_mode:
        # scaleup -> 1 : No change
        # scaleup -> 0 : All domain set to the same size
        mxcnt = 100000
        if len(positives) < mxcnt * 0.95:
            newcnt = int((len(positives) / mxcnt) ** float(user_mode["scaleup"]) * mxcnt)
            positives = positives + random.choices(positives, k = newcnt - len(positives))
    if "maxcnt" in user_mode:
        positives = positives[:int(user_mode["maxcnt"])]
    
    if "negalign" in user_mode:
        negcorpus_list = [" ".join(d['corpus']).split() for d in data]
    else:
        negcorpus = [w for d in data for w in " ".join(d['corpus']).split()]
    for pi, p in enumerate(positives):
        neg_terms = []
        domains.append(domain2id[domain])
        if "dyneg" in user_mode:
            continue
        if "negalign" in user_mode:
            negcorpus = negcorpus_list[pi]
            while len(negcorpus) < max_span_length:
                negcorpus = random.choice(negcorpus_list)
        neg_terms = sample_neg_span(negcnt, negcorpus, max_span_length, positives_set, stopwords)
        negatives.append(neg_terms)
    return positives, negatives, domains

def sample_neg_span(negcnt, negcorpus, max_span_length, positives_set, stopwords):
    neg_terms = []
    for _ in range(negcnt):
        while True:
            st = random.randint(0, len(negcorpus) - max_span_length)
            ed = random.randint(st + 1, min(len(negcorpus), st + max_span_length))
            neg_term = " ".join(negcorpus[st : ed])
            if random.random() < 0.03:
                    neg_term = random.sample(stopwords, 1)[0]
            if neg_term.lower() not in positives_set:
                neg_terms.append(neg_term)
                break
    return neg_terms


class DinnerDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer, user_mode, task):
        from nltk.corpus import stopwords
        import string
        self.stopwords = stopwords.words('english')
        self.stopwords += [*(string.ascii_lowercase + string.ascii_uppercase)]
        self.stopwords += [str(a) for a in range(10)]

        self.positives = []
        self.negatives = []
        self.domains = []
        negcnt = int(user_mode['negcnt']) if 'negcnt' in user_mode else 4
        max_span_length = 8
        if "msl" in user_mode:
            max_span_length = int(user_mode["msl"])
        domain2id = user_mode["domain2id"]
        domains = user_mode["config"]["train"]
        dmode = "stk"
        if sys.gettrace():
            dmode = "none"
        processed_data_list = []
        if dmode == "mp":
            with multiprocessing.Pool() as pool:
                print(pool._processes)
                processed_data_list = list(tqdm(pool.imap(make_domain_data, [[domain, user_mode, domain2id, max_span_length, negcnt, self.stopwords] for domain in domains])))
        elif dmode == "stk":
            import streamtask
            stk = streamtask.StreamTask()
            stk.add_data(domains)
            stk.add_module(make_domain_data, os.cpu_count() - 1, args = [user_mode, domain2id, max_span_length, negcnt, self.stopwords])
            stk.run()
            stk.join()
            print("finish join")
            processed_data_list = stk.get_results()
            print("finish stk.get_results")
        else:
            for domain in tqdm(domains):
                processed_data_list.append(make_domain_data(domain, user_mode, domain2id, max_span_length, negcnt, self.stopwords))
        self.positives = list(itertools.chain(*[d[0] for d in processed_data_list]))
        print("finish join positives")
        self.negatives = list(itertools.chain(*[d[1] for d in processed_data_list]))
        self.domains   = list(itertools.chain(*[d[2] for d in processed_data_list]))
        print("finish join domains")
        print(self.positives[:10])





        if "dyneg" in user_mode:
            if not hasattr(self, "negcorpus_dict"):
                self.negcorpus_dict = {}
                self.positives_set_dict = {}
            self.negcorpus_dict[domain2id[domain]] = negcorpus
            self.positives_set_dict[domain2id[domain]] = positives_set
        if "dratio" in user_mode:
            dlen = int(len(self.positives) * float(user_mode["dratio"]))
            self.positives = self.positives[:dlen]
            self.negatives = self.negatives[:dlen]
            self.domains = self.domains[:dlen]
        if "neg" in user_mode or "mixneg" in user_mode:
            pos_set = set([s.lower() for s in self.positives])
            if user_mode["neg"] == "wiki2":
                negcorpus = open("datasets/wiki2/wiki.train.tokens").read().split()
            for i in range(len(self.negatives)):
                for j in range(len(self.negatives[i])):
                    while True:
                        l = len(self.negatives[i][j].split())
                        st0 = random.randint(0, len(negcorpus) - 60)
                        swords = " ".join(negcorpus[st0 : st0 + l])
                        if swords.lower() not in pos_set:
                            if "mixneg" in user_mode:
                                if random.random() < float(user_mode["mixneg"]):
                                    self.negatives[i][j] = swords
                            else:
                                self.negatives[i][j] = swords
                            break

        self.tokenizer = tokenizer
        self.task = task
        self.user_mode = user_mode
        self.negcnt = negcnt
        self.max_span_length = max_span_length
        
    def __getitem__(self, query_idx):
        query_name1 = self.positives[query_idx]
        domain_id = self.domains[query_idx]
        if "dyneg" in self.user_mode:
            negcorpus = self.negcorpus_dict[domain_id]
            positives_set = self.positives_set_dict[domain_id]
            query_name2 = self.sample_neg_span(self.negcnt, negcorpus, self.max_span_length, positives_set)
        else:
            query_name2 = self.negatives[query_idx]
        if query_name1 in ["England", "Germany"]:
            a = 1
        if "unsel" not in self.user_mode:
            domain_id = [domain_id] * (len(query_name2) + 1)
        if "disreg" in self.user_mode:
            dt = max(min(random.randint(query_idx-10, query_idx+10), len(self.positives) - 1), 0)
            while domain_id[0] != self.domains[dt]:
                dt = max(min(random.randint(query_idx-10, query_idx+10), len(self.positives) - 1), 0)
            query_name1 = [query_name1, self.positives[dt]]
            query_name2 = query_name2 + self.negatives[dt]
            domain_id = domain_id * 2
        return query_name1, query_name2, domain_id

    def __len__(self):
        return len(self.positives)