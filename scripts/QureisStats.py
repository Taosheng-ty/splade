import os,pickle
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
import gzip,pickle
from collections import defaultdict
import torch
from collections import Counter
import json
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")

from splade.datasets.datasets import PairsDatasetPreLoad, DistilPairsDatasetPreLoad, MsMarcoHardNegatives, \
    CollectionDatasetPreLoad

model_type_or_dir="distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
numToken=tokenizer.vocab_size

# ##for debug
tokenizationPath='output/toy1k/QuerytokenCop.pkl'
QueryStatsPath='output/toy1k/QueryStats.pkl'
inputDict={
    "dataset_path": 'data/toy_data1k/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
    "document_dir": 'data/toy_data1k/full_collection',
    "query_dir":'data/toy_data1k/train_queries/queries',
    "qrels_path":'data/msmarco/train_queries/qrels.json',
    }

# tokenizationPath='output/toy20/QuerytokenCop.pkl'
# QueryStatsPath='output/toy20/QueryStats.pkl'
# inputDict={
#     "dataset_path": 'data/toy_data1k/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
#     "document_dir": 'data/toy_data1k/full_collection',
#     "query_dir":'data/toy_data1k/train_queries/queries',
#     "qrels_path":'data/msmarco/train_queries/qrels.json',
# #     }
# inputDict={
#     "dataset_path": 'data/toy_data20/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
#     "document_dir": 'data/toy_data20/full_collection',
#     "query_dir":'data/toy_data20/train_queries/queries',
#     "qrels_path":'data/msmarco/train_queries/qrels.json',
    # }

# tokenizationPath='output/full/QuerytokenCop.pkl'
# QueryStatsPath='output/full/QueryStats.pkl'
# inputDict={
#     "dataset_path": 'data/msmarco/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
#     "document_dir": 'data/msmarco/full_collection',
#     "query_dir":'data/msmarco/train_queries/queries',
#     "qrels_path":'data/msmarco/train_queries/qrels.json'}

dataset=MsMarcoHardNegatives(**inputDict)
directory=os.path.dirname(tokenizationPath)
os.makedirs(directory,exist_ok=True)
if os.path.exists(tokenizationPath):
    with open(tokenizationPath, 'rb') as f:
        tokenCop = pickle.load(f)
else:
    ## load the corpus
    # inputDict={
    # "dataset_path": 'data/msmarco/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
    # "document_dir": 'data/msmarco/full_collection',
    # "query_dir":'data/msmarco/train_queries/queries',
    # "qrels_path":'data/msmarco/train_queries/qrels.json'}
    corpus=dataset.query_dataset.data_dict
    keys=list(dataset.query_dataset.data_dict.keys())
    listCorp=[]
    corpusLen=len(corpus)
    for key in tqdm(corpus,desc="reading data"):
        listCorp.append(corpus[str(key)])
    batch=100000
    numBatch=corpusLen//batch+1
    tokenCopList=[]
    for idx in tqdm(range(0,numBatch),desc="tokening"):
        if idx*batch>corpusLen:
            break
        tokenCopList+=list(tokenizer(listCorp[idx*batch:(idx+1)*batch], return_tensors="np")['input_ids'])
    tokenCop={keys[idx]:tokened  for  idx,tokened in enumerate(tokenCopList)}
    with open(tokenizationPath, 'wb') as f:
        pickle.dump(tokenCop, f)

InverseVocab={key[1]:key[0] for ind,key in enumerate(tokenizer.vocab.items())}
numToken=len(tokenizer.vocab.keys())

qrels=dataset.qrels
# log_tf=torch.zeros(numToken)
# log_tfToken={key:0 for key in tokenizer.vocab}
logPosdoc={}
log_tf=np.zeros(numToken)
from collections import Counter
for idx in tqdm(tokenCop,desc="processing data corpus"):
    # print(idx)
    tokenSent=tokenCop[idx]
    FreqTokenSent=Counter(tokenSent)
    for token in FreqTokenSent:
        freq=FreqTokenSent[token]
        log_tf[token]+=np.log(1+freq)
    if idx not in qrels:
        continue
    for doc in qrels[idx]:
        if qrels[idx][doc]>0:
            if doc not in logPosdoc:
                logPosdoc[doc]=defaultdict(float)
            for token in FreqTokenSent:
                freq=FreqTokenSent[token]
                logPosdoc[doc][token]+=np.log(1+freq)
    
with open(QueryStatsPath, 'wb') as f:
    pickle.dump({"corpus":log_tf,"pos_tf":logPosdoc}, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open('output/log-corpus-tf-token.pkl', 'rb') as f:
#     pickle.dump(log_tfToken, f)


# with open('output/log-corpus-tf-tokenid.pkl', 'rb') as f:
#     log_tf=pickle.load(f)
# indices=list(log_tf.keys())
# val=list(log_tf.values())
# numUniqToken=len(indices)
# # indices.append[0 for i in cortf]
# corpusEmb=torch.zeros(numUniqToken)
# corpusEmb[indices]=torch.tensor(val,dtype=torch.float32)
# with open('output/corpus-tf-log-tensor-dict.pkl', 'wb') as f:
#     pickle.dump(corpusEmb, f)


