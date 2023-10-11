import os
import pickle
import logging
import torch
import numpy as np
import time
from tqdm import tqdm
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from .metric_learning import *
from transformers import AutoConfig, AutoModelForSequenceClassification

from transformers import (
    AutoTokenizer, 
    AutoModel, 
)

LOGGER = logging.getLogger()


class Model_Wrapper(object):
    """
    Wrapper class for BERT encoder
    """

    def __init__(self, user_mode):
        self.tokenizer = None
        self.encoder = None
        self.user_mode = user_mode

    def get_dense_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    def save_model(self, path, context=False):
        # save bert vocab
        self.tokenizer.save_pretrained(path)

        # save bert model, bert config
        if "edge" in self.user_mode or "guner" in self.user_mode or "dinner" in self.user_mode:
            if isinstance(self.encoder, torch.nn.DataParallel):
                self.scmodel.base_model = self.encoder.module
                self.scmodel.classifier = self.classifier.module
            self.scmodel.save_pretrained(path) # maple
        else:
            self.encoder.save_pretrained(path)

    def save_train_state(self, path, data_loader, optimizer):
        torch.save([data_loader, optimizer.state_dict()], f'{path}/train_state.pth')

    def load_model(self, path, max_length=25, use_cuda=True, lowercase=True, num_labels = 2, local_files_only = False):
        self.load_bert(path, max_length, use_cuda, num_labels = num_labels, local_files_only = local_files_only)
        
        return self

    def load_bert(self, path, max_length, use_cuda, lowercase=True, num_labels = 2, local_files_only = False):
        self.tokenizer = AutoTokenizer.from_pretrained(path, 
                use_fast=True, do_lower_case=lowercase, local_files_only = local_files_only)
        if "ptm" in self.user_mode and self.user_mode["ptm"] == "gpt2":
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if "edge" in self.user_mode or "guner" in self.user_mode or "dinner" in self.user_mode:
            self.scmodel = AutoModelForSequenceClassification.from_pretrained(path, num_labels = num_labels, local_files_only = local_files_only)
            self.encoder = self.scmodel.base_model
            if "lenemb" in self.user_mode:
                class LenEmbEncoder(nn.Module):
                    def __init__(self, encoder):
                        super(LenEmbEncoder, self).__init__()
                        self.encoder = encoder
                        self.seq_len_embedding = nn.Embedding(1000, self.encoder.embeddings.word_embeddings.weight.shape[1])
                    def forward(self, **kwargs):
                        semb = self.seq_len_embedding((kwargs['input_ids'] > 0).sum(1))
                        res = self.encoder(**kwargs)
                        res.pooler_output = res.pooler_output + semb
                        return res
                lmodel = LenEmbEncoder(self.scmodel.base_model)
                self.scmodel.encoder = lmodel
                self.encoder = self.scmodel.encoder
            if "bigcls" in self.user_mode:
                fdim = self.scmodel.classifier.in_features
                self.scmodel.classifier = nn.Sequential(
                    nn.Linear(fdim, fdim),
                    nn.ReLU(),
                    nn.Linear(fdim, fdim),
                    nn.ReLU(),
                    nn.Linear(fdim, num_labels)
                )
            elif "sproj" in self.user_mode:
                class SProj(nn.Module):
                    def __init__(self, num_labels):
                        super(SProj, self).__init__()
                        self.sproj = nn.Linear(fdim, fdim * num_labels)
                        self.clf = nn.Linear(fdim, num_labels)
                        self.num_labels = num_labels
                        
                    def forward(self, x):
                        x1 = self.sproj(x)
                        x2 = x1.reshape(x1.shape[0], self.num_labels, -1).transpose(0, 1)
                        q = self.clf.weight.unsqueeze(-1)
                        v = x2.bmm(q).squeeze(-1).transpose(0, 1) + self.clf.bias.unsqueeze(0)
                        return v
                fdim = self.scmodel.classifier.in_features
                self.scmodel.classifier = SProj(num_labels)
             
            if "fixenc" in self.user_mode:
                params = dict(self.scmodel.named_parameters())
                for p in params:
                    if not p.startswith("classifier"):
                        params[p].requires_grad=False
            model_path = f"{path}/pytorch_model.bin"
            if os.path.isfile(model_path):
                self.scmodel.load_state_dict(torch.load(model_path))
        else:
            self.encoder = AutoModel.from_pretrained(path)
        if "adapter" in self.user_mode:
            if not self.encoder.has_adapters():
                self.encoder.add_adapter("adapter")
                self.encoder.train_adapter("adapter")
            else:
                self.encoder.set_active_adapters("adapter")
        if use_cuda:
            self.encoder = self.encoder.cuda()

        return self.encoder, self.tokenizer
    

    def get_score_matrix(self, query_embeds, dict_embeds, cosine=False, normalise=False):
        """
        Return score matrix

        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings

        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        if cosine:
            score_matrix = cosine_similarity(query_embeds, dict_embeds)
        else:
            score_matrix = np.matmul(query_embeds, dict_embeds.T)

        if normalise:
            score_matrix = (score_matrix - score_matrix.min() ) / (score_matrix.max() - score_matrix.min())
        
        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """
        
        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0,cols.shape[0])[:, np.newaxis],cols.shape[1],axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix,-topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix) 
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def retrieve_candidate_cuda(self, score_matrix, topk, batch_size=128, show_progress=False):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """

        res = None
        for i in tqdm(np.arange(0, score_matrix.shape[0], batch_size), disable=not show_progress):
            score_matrix_tmp = torch.tensor(score_matrix[i:i+batch_size]).cuda()
            matrix_sorted = torch.argsort(score_matrix_tmp, dim=1, descending=True)[:, :topk].cpu()
            if res is None: 
                res = matrix_sorted
            else:
                res = torch.cat([res, matrix_sorted], axis=0)

        return res.numpy()

    def embed_dense(self, names, show_progress=False, batch_size=2048, agg_mode="cls"):
        """
        Embedding data into dense representations

        Parameters
        ----------
        names : np.array
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval() # prevent dropout
        
        batch_size=batch_size
        dense_embeds = []

        #print ("converting names to list...")
        #names = names.tolist()

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(range(0, len(names), batch_size))
            else:
                iterations = range(0, len(names), batch_size)
                
            for start in iterations:
                end = min(start + batch_size, len(names))
                batch = names[start:end]
                batch_tokenized_names = self.tokenizer.batch_encode_plus(
                        batch, add_special_tokens=True, 
                        truncation=True, max_length=25, 
                        padding="max_length", return_tensors='pt')
                batch_tokenized_names_cuda = {}
                for k,v in batch_tokenized_names.items(): 
                    batch_tokenized_names_cuda[k] = v.cuda()
                
                last_hidden_state = self.encoder(**batch_tokenized_names_cuda)[0]
                if agg_mode == "cls":
                    batch_dense_embeds = last_hidden_state[:,0,:] # [CLS]
                elif agg_mode == "mean_all_tok":
                    batch_dense_embeds = last_hidden_state.mean(1) # pooling
                elif agg_mode == "mean":
                    batch_dense_embeds = (last_hidden_state * batch_tokenized_names_cuda['attention_mask'].unsqueeze(-1)).sum(1) / batch_tokenized_names_cuda['attention_mask'].sum(-1).unsqueeze(-1)
                else:
                    print ("no such agg_mode:", agg_mode)

                batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)
        
        return dense_embeds

