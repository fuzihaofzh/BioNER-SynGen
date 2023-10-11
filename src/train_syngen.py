#!/usr/bin/env python
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import autocast 
from torch.cuda.amp import GradScaler
from pytorch_metric_learning import samplers
import logging
import time
import pdb
import os
import json
import random
from tqdm import tqdm
import torch.nn.functional as F
import torchvision

import sys
sys.path.append("../") 

#import wandb
#wandb.init(project="sapbert")

from sap.data_loader import (
    DictionaryDataset,
    QueryDataset,
    QueryDataset_pretraining,
    MetricLearningDataset,
    MetricLearningDataset_pairwise,
    EdgeDataset,
    GunerDataset,
    DinnerDataset,
)
from sap.model_wrapper import (
    Model_Wrapper,
)
from sap.metric_learning import (
    Sap_Metric_Learning,
)

LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='sapbert train')

    # Required
    parser.add_argument('--model_dir', 
                        help='Directory for pretrained model')
    parser.add_argument('--train_dir', type=str, required=True,
                    help='training set directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=240, type=int)
    parser.add_argument('--num_workers',
                        help='train batch size',
                        default=0, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=3, type=int)
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000000)
    parser.add_argument('--amp', action="store_true", 
            help="automatic mixed precision training")
    parser.add_argument('--parallel', action="store_true") 
    #parser.add_argument('--cased', action="store_true") 
    parser.add_argument('--pairwise', action="store_true",
            help="if loading pairwise formatted datasets") 
    parser.add_argument('--random_seed',
                        help='epoch to train',
                        default=1996, type=int)
    parser.add_argument('--loss',
                        help="{ms_loss|cosine_loss|circle_loss|triplet_loss}}",
                        default="ms_loss")
    parser.add_argument('--use_miner', action="store_true") 
    parser.add_argument('--miner_margin', default=0.2, type=float) 
    parser.add_argument('--type_of_triplets', default="all", type=str) 
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_all_tok}") 
    parser.add_argument('--user_mode', default="", type=str, help="user_mode") 
    parser.add_argument('--task', default="", type=str, help="task name") 

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_dictionary(dictionary_path):
    """
    load dictionary
    
    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path
    )
    
    return dictionary.data
    
def load_queries(data_dir, filter_composite, filter_duplicate):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    
    return dataset.data

def load_queries_pretraining(data_dir, filter_duplicate):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset_pretraining(
        data_dir=data_dir,
        filter_duplicate=filter_duplicate
    )
    
    return dataset.data

def get_batch_xy(data, user_mode = {}):
    batch_x1, batch_x2, batch_y = data
    batch_x = {}
    for k in batch_x1:
        batch_x[k] = torch.cat([batch_x1[k], batch_x2[k]])
    batch_x_cuda = {}
    for k,v in batch_x.items():
        batch_x_cuda[k] = v.cuda()
    if "sel" in user_mode or "dinner" in user_mode:
        batch_y_cuda = batch_y.cuda()
    else:
        batch_y_cuda = torch.cat([batch_y.new_ones(batch_x1[k].shape[0]).cuda(), batch_y.new_zeros(batch_x2[k].shape[0]).cuda()])
    return batch_x_cuda, batch_y_cuda

def train(args, data_loader, model, scaler=None, model_wrapper=None, step_global=0):
    LOGGER.info(f"train {args.output_dir} !")
    
    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    total = len(data_loader)
    if type(data_loader) is list:
        from itertools import cycle
        total = len(data_loader[1])
        data_loader = zip(*data_loader) if len(data_loader[0]) > len(data_loader[1]) else zip(cycle(data_loader[0]), data_loader[1])
    pbar = tqdm(enumerate(data_loader), total=total)
    for i, data in pbar:
        model.optimizer.zero_grad()
        loss = 0
        if "edge" in model.user_mode or "guner" in model.user_mode or "dinner" in model.user_mode:
            batch_x_cuda, batch_y_cuda = get_batch_xy(data[1], model.user_mode)
            if "guner" in model.user_mode and "sel" not in model.user_mode:
                batch_y_cuda[:data[1][2].shape[0]] = data[1][2]
            """
            batch_x1, batch_x2, batch_y = data[1]
            batch_x = {}
            for k in batch_x1:
                batch_x[k] = torch.cat([batch_x1[k], batch_x2[k]])
            batch_x_cuda = {}
            for k,v in batch_x.items():
                batch_x_cuda[k] = v.cuda()
            batch_y_cuda = torch.cat([batch_y.new_ones(batch_x1[k].shape[0]).cuda(), batch_y.new_zeros(batch_x2[k].shape[0]).cuda()])"""

            #fea = model.scmodel(**batch_x_cuda)
            outputs = model.encoder(**batch_x_cuda)
            if outputs.pooler_output is None:
                outputs.pooler_output = outputs.last_hidden_state
            if 'nodropout' in model.user_mode or not hasattr(model.scmodel, "dropout"):
                fea = outputs.pooler_output
            else:
                fea = model.scmodel.dropout(outputs.pooler_output)
            if "bimsloss" in model.user_mode:
                loss = loss + model.loss(fea, batch_y_cuda)
            if 'pdreg' in model.user_mode: # positive distance regularization
                pr = float(model.user_mode['pdreg']) if model.user_mode['pdreg'] is not None else 1.0
                gamma = float(model.user_mode['gamma']) if model.user_mode['gamma'] is not None else 0.1
                batch_x0, batch_y0 = get_batch_xy(data[0])
                fea0 = model.encoder(**batch_x0)
                l = fea0.pooler_output.shape[0] // 2
                av, sv = fea0.pooler_output[:l], fea0.pooler_output[l:]
                #loss = loss + pr * ((av - sv)**2).mean()
                mloss = (((av - sv)**2).mean(1) - gamma).clamp(0)
                loss = loss + pr * mloss.sum() / max((mloss>0).sum(), 0.1)
            if 'pndreg' in model.user_mode: # positive negative distance regularization
                pr = float(model.user_mode['pndreg']) if model.user_mode['pndreg'] is not None else 1.0
                batch_x0, batch_y0 = get_batch_xy(data[0])
                fea0 = model.encoder(**batch_x0)
                l = fea0.pooler_output.shape[0] // 2
                av, pv, nv = fea0.pooler_output[:l], fea0.pooler_output[l:], fea[-l:]
                l = min(len(av), len(nv))
                loss = loss + pr * F.triplet_margin_loss(av[:l], pv[:l], nv[:l])
            if 'ppertube' in model.user_mode:
                s = float(model.user_mode['ppertube']) if model.user_mode['ppertube'] is not None else 0.1
                fea = fea * (1 + s * torch.rand_like(fea)* batch_y_cuda.unsqueeze(1))
            elif 'apertube' in model.user_mode:
                s = float(model.user_mode['apertube']) if model.user_mode['apertube'] is not None else 0.1
                fea = fea * (1 + s * torch.rand_like(fea))
            logits = model.scmodel.classifier(fea)
            if args.amp:
                with autocast():
                    if "focal" in model.user_mode:
                        loss = loss + torchvision.ops.sigmoid_focal_loss(logits.log_softmax(1)[:, 1], batch_y_cuda.type_as(logits), reduction = 'mean')
                    elif "dinner" in model.user_mode:
                        y = torch.cat([batch_y_cuda[:, 0], batch_y_cuda[:, 1:].flatten()])
                        lgts = logits[torch.arange(len(logits)), y]
                        label = lgts.new_zeros(lgts.shape)
                        label[:batch_y_cuda.shape[0]] = 1
                        gloss = F.binary_cross_entropy_with_logits(lgts, label)
                        loss = loss + gloss
                    elif "guner" in model.user_mode:
                        if "sel" not in model.user_mode:
                            label = logits.new_zeros(logits.shape)
                            label[torch.arange(len(label)), batch_y_cuda] = 1
                        if "lmargin" in model.user_mode:
                            gloss = F.binary_cross_entropy_with_logits(logits, label, reduce = False)
                            gloss = gloss.masked_select(gloss > float(model.user_mode["lmargin"])).mean()
                        elif "sel" in model.user_mode:
                            y = torch.cat([batch_y_cuda[:, 0], batch_y_cuda[:, 1:].flatten()])
                            lgts = logits[torch.arange(len(logits)), y]
                            label = lgts.new_zeros(lgts.shape)
                            label[:batch_y_cuda.shape[0]] = 1
                            gloss = F.binary_cross_entropy_with_logits(lgts, label)
                        else:
                           gloss = F.binary_cross_entropy_with_logits(logits, label)
                        loss = loss + gloss
                    else:
                        loss = loss + F.cross_entropy(logits, batch_y_cuda)
                    if 'ppreg' in model.user_mode:# positive pertubation regularization
                        s = float(model.user_mode['ppreg']) if model.user_mode['ppreg'] is not None else 0.1
                        fea1 = fea * (1 + s * torch.rand_like(fea))
                        logits1 = model.scmodel.classifier(fea1)
                        loss = loss + F.cross_entropy(logits1, logits.softmax(1).detach())

            else:
                loss = loss + model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda) 
        if not ("edge" in model.user_mode or "guner" in model.user_mode or "dinner" in model.user_mode) or "joint" in model.user_mode:
            if "edge" in model.user_mode:
                data = data[0]
            batch_x1, batch_x2, batch_y = data
            batch_x_cuda1, batch_x_cuda2 = {},{}
            for k,v in batch_x1.items():
                batch_x_cuda1[k] = v.cuda()
            for k,v in batch_x2.items():
                batch_x_cuda2[k] = v.cuda()

            batch_y_cuda = batch_y.cuda()
        
            if args.amp:
                with autocast():
                    if "joint" in model.user_mode:
                        s = float(model.user_mode['joint']) if model.user_mode['joint'] is not None else 0.1
                        loss = loss + s * model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda) 
                    else:
                        loss = loss + model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)  
            else:
                loss = loss + model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)  
        if args.amp:
            scaler.scale(loss).backward()
            if "gradclip" in model.user_mode:
                gradclip = float(model.user_mode["gradclip"]) if model.user_mode["gradclip"] is not None else 1000.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip)
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        pbar.set_postfix({'loss': loss.item()})
        train_loss += loss.item()
        #wandb.log({"Loss": loss.item()})
        train_steps += 1
        step_global += 1
        #if (i+1) % 10 == 0:
        #LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1,train_loss / (train_steps+1e-9)))
        #LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1, loss.item()))

        # save model every K iterations
        if step_global % args.checkpoint_step == 0:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_iter_{}".format(str(step_global)))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global
    
def main(args):
    init_logging()
    #init_seed(args.seed)
    user_mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None for e in (args.user_mode[0].split(',') if type(args.user_mode) is not str else args.user_mode.split(',')) }
    if 'epoch' in user_mode:
        args.epoch = int(user_mode['epoch'])
    if 'ptm' in user_mode:
        if user_mode['ptm'] == 'biobert':
            args.model_dir = "dmis-lab/biobert-base-cased-v1.2"
        elif user_mode['ptm'] == 'pubmedbert':
            args.model_dir = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        elif user_mode['ptm'] == 'bert':
            args.model_dir = "bert-base-uncased"
        elif user_mode['ptm'] == 'bert-large':
            args.model_dir = "bert-large-cased"
        elif user_mode['ptm'] == "roberta":
            args.model_dir = "roberta-base"
        elif user_mode['ptm'] == "deberta":
            args.model_dir = "microsoft/deberta-v3-base"
        elif user_mode['ptm'] == "scibert":
            args.model_dir = "allenai/scibert_scivocab_cased"
        else:
            args.model_dir = user_mode['ptm']
    if 'umlssy' in args.user_mode:
        args.checkpoint_step = 2000
    if 'lr' in user_mode:
        args.learning_rate = float(user_mode['lr'])
    if 'seed' in user_mode:
        args.random_seed = int(user_mode['seed'])
    if 'srand' in user_mode:
        args.random_seed += int(user_mode['srand'])
    if 'pportion' in user_mode:
        args.epoch = min(807, int(1. / float(user_mode['pportion'])))
        user_mode['neg'] = f"{user_mode['pos']}-sp={user_mode['pportion']}"
    print(args)

    torch.manual_seed(args.random_seed)
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        

    num_labels = 2
    if 'guner' in user_mode:
        id2domain = json.load(open(f"output/preprocessed/guner/{args.task}/domains.json"))
        num_labels = len(id2domain)
    if 'dinner' in user_mode:
        #all_domains = json.load(open(f"datasets/{args.task}/names2qid.json"))
        #id2domain = [r[1] for r in sorted([[all_domains[a][1], a] for a in all_domains])[::-1]]
        user_mode["config"] = json.load(open(f"output/preprocessed/dinner/config.json"))
        id2domain = list(user_mode["config"]["train"][user_mode["conf"]].keys())
        domain2id = {id2domain[i] : i for i in range(len(id2domain))}
        num_labels = len(id2domain)
        user_mode["id2domain"] = id2domain
        user_mode["domain2id"] = domain2id
    # load BERT tokenizer, dense_encoder
    model_wrapper = Model_Wrapper(user_mode)
    encoder, tokenizer = model_wrapper.load_bert(
        path=args.model_dir,
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        #lowercase=not args.cased
        num_labels = num_labels,#maple
    )
    
    # load SAP model
    model = Sap_Metric_Learning(
            encoder = encoder,
            learning_rate=args.learning_rate, 
            weight_decay=args.weight_decay,
            use_cuda=args.use_cuda,
            pairwise=args.pairwise,
            loss=args.loss,
            use_miner=args.use_miner,
            miner_margin=args.miner_margin,
            type_of_triplets=args.type_of_triplets,
            agg_mode=args.agg_mode,
            user_mode=user_mode,#maple
            scmodel=model_wrapper.scmodel if ('edge' in user_mode or 'guner' in user_mode or 'dinner' in user_mode) else None ,
    )

    if args.parallel:
        model.encoder = torch.nn.DataParallel(model.encoder)
        LOGGER.info("using nn.DataParallel")
    
    def collate_fn_batch_encoding(batch):
        query1, query2, query_id = zip(*batch)
        if type(query1[0]) is list:#edge
            query1 = [qq for q in query1 for qq in q]
        if type(query2[0]) is list:#edge
            query2 = [qq for q in query2 for qq in q]
        query_encodings1 = tokenizer.batch_encode_plus(
                list(query1), 
                max_length=args.max_length, 
                padding="max_length", 
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt")
        query_encodings2 = tokenizer.batch_encode_plus(
                list(query2), 
                max_length=args.max_length, 
                padding="max_length", 
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt")
        #query_encodings_cuda = {}
        #for k,v in query_encodings.items():
        #    query_encodings_cuda[k] = v.cuda()
        query_ids = torch.tensor(list(query_id))
        return  query_encodings1, query_encodings2, query_ids

    if args.pairwise:
        train_set = MetricLearningDataset_pairwise(
                path=args.train_dir,
                tokenizer = tokenizer
        )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.train_batch_size,# maple
            shuffle=True,
            num_workers=args.num_workers, # 16 maple
            collate_fn=collate_fn_batch_encoding
        )
        if "edge" in user_mode:
            edge_train_set = EdgeDataset(
                path=args.train_dir,
                tokenizer = tokenizer,
                user_mode = user_mode,
                task = args.task
            )
            add_loader = True
        elif "guner" in user_mode:
            edge_train_set = GunerDataset(
                path=args.train_dir,
                tokenizer = tokenizer,
                user_mode = user_mode,
                task = args.task
            )
            add_loader = True
        elif "dinner" in user_mode:
            edge_train_set = DinnerDataset(
                path=args.train_dir,
                tokenizer = tokenizer,
                user_mode = user_mode,
                task = args.task,
            )
            add_loader = True
        if add_loader:
            edge_train_loader = torch.utils.data.DataLoader(
                edge_train_set,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.num_workers, # 16 maple
                collate_fn=collate_fn_batch_encoding
            )
            if any([m in user_mode for m in ['joint', 'pdreg', 'pndreg']]):
                train_loader = [train_loader, edge_train_loader]
            else:
                train_loader = [edge_train_loader, edge_train_loader]
    else:
        train_set = MetricLearningDataset(
            path=args.train_dir,
            tokenizer = tokenizer
        )
        # using a sampler
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.train_batch_size,
            #shuffle=True,
            sampler=samplers.MPerClassSampler(train_set.query_ids,\
                2, length_before_new_iter=100000),
            num_workers=args.num_workers, #16 maple
            )
    # mixed precision training 
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    start = time.time()
    step_global = 0
    if "headtune" in model.user_mode:
        params = dict(model.named_parameters())
        for p in params:
            if p.startswith("encoder"):
                params[p].requires_grad = False
    for epoch in range(1,args.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))
        if 'epochseed' in model.user_mode:
            random.seed(0)

        # train
        train_loss, step_global = train(args, data_loader=train_loader, model=model, scaler=scaler, model_wrapper=model_wrapper, step_global=step_global)
        LOGGER.info('loss/train_per_epoch={}/{},{}'.format(train_loss,epoch, args.output_dir))
        
        # save model every epoch
        if args.save_checkpoint_all:
            """checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)"""
            model_wrapper.save_model(args.output_dir)#maple
        
        # save model last epoch
        if epoch == args.epoch:
            model_wrapper.save_model(args.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
