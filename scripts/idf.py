import os,pickle
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
import gzip,pickle
from collections import defaultdict
import torch
from collections import Counter
import json
# scriptPath=os.path.dirname(os.path.abspath(__file__))
# os.chdir(scriptPath+"/..")
model_type_or_dir="distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
numToken=tokenizer.vocab_size


tokenizationPath='output/tokenCop.pkl'
if os.path.exists(tokenizationPath):
    with open(tokenizationPath, 'rb') as f:
        tokenCop = pickle.load(f)
else:
    ## load the corpus
    collection_picked_file="/raid/datasets/shared/MSMARCO/collection.pickle"
    if os.path.exists(collection_picked_file):
        with open(collection_picked_file, 'rb') as f:
            corpus = pickle.load(f)
    listCorp=[]
    corpusLen=len(corpus)
    for key in tqdm(range(corpusLen),desc="reading data"):
        listCorp.append(corpus[str(key)])
    batch=100000
    numBatch=corpusLen//batch+1
    tokenCop=[]
    for idx in tqdm(range(0,numBatch),desc="tokening"):
        if idx*batch>corpusLen:
            break
        tokenCop+=list(tokenizer(listCorp[idx*batch:(idx+1)*batch], return_tensors="np")['input_ids'])
    with open('output/tokenCop.pkl', 'wb') as f:
        pickle.dump(tokenCop, f)

# InverseVocab={key[1]:key[0] for ind,key in enumerate(tokenizer.vocab.items())}
# numToken=len(tokenizer.vocab.keys())

    
# log_idf={ind:0 for ind in range(numToken)}
# # log_idf=torch.zeros(numToken)
# log_idfToken={key:0 for key in tokenizer.vocab}
# from collections import Counter
# for tokenSent in tqdm(tokenCop[:5],desc="processing data corpus"):
#     FreqTokenSent=Counter(tokenSent)
#     for token in FreqTokenSent:
#         freq=FreqTokenSent[token]
#         log_idf[token]+=np.log(freq)
#         log_idfToken[InverseVocab[token]]+=np.log(freq)
# with open('output/corpus-tf-log-tensor-dict.pkl', 'wb') as f:
#     pickle.dump(log_idf, f)
# with open('output/log-corpus-tf-token.pkl', 'rb') as f:
#     pickle.dump(log_idfToken, f)


# with open('output/log-corpus-tf-tokenid.pkl', 'rb') as f:
#     log_idf=pickle.load(f)
# indices=list(log_idf.keys())
# val=list(log_idf.values())
# numUniqToken=len(indices)
# # indices.append[0 for i in cortf]
# corpusEmb=torch.zeros(numUniqToken)
# corpusEmb[indices]=torch.tensor(val,dtype=torch.float32)
# with open('output/corpus-tf-log-tensor-dict.pkl', 'wb') as f:
#     pickle.dump(corpusEmb, f)



TopK=10
dataset_path="data/msmarco/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
with gzip.open(dataset_path, "rb") as fIn:
    scores_dict = pickle.load(fIn)
TopK=10
log_idf={}


qrels_path="data/msmarco/train_queries/qrels.json"
with open(qrels_path) as reader:
    qrels = json.load(reader)
query_listTeacher = scores_dict.keys()
query_list=[]
for query in query_listTeacher:
    if str(query) in qrels.keys():
        query_list.append(query)

q_ids=list(scores_dict.keys())
for q_id in tqdm(q_ids,desc="processing data"):
    candidates=scores_dict[q_id]
    labelled_positive = list(qrels[str(q_id)].keys()) if str(q_id) in qrels else []
    p_idsTeacher = sorted(candidates,key=candidates.get,reverse=True)[:TopK]
    p_ids=[int(i)for i in labelled_positive]
    for pid in p_idsTeacher:
        if pid not in p_ids:
            p_ids.append(pid)
        if len(p_ids)>TopK:
            break
    log_idf[q_id]=defaultdict(int)
    for p_id in p_ids:
        tokenSent=tokenCop[p_id]
        FreqTokenSent=Counter(tokenSent)
        
        for token in FreqTokenSent:
            freq=FreqTokenSent[token]
            if np.log(freq)<=0:
                continue
            log_idf[q_id][token]+=np.log(freq)
    
        

# with open(f'output/log-doc-tf-topk-{TopK}.pkl', 'wb') as f:
#     pickle.dump(log_idf, f)


# with open(f'output/log-doc-tf-topk-{TopK}.pkl', 'rb') as f:
#     log_idf=pickle.load(f)
    
# indices=[[],[]]
# val=[]
# count_q=0
# convertedMap={}
# q_ids=list(log_idf.keys())
# for q_id in tqdm(q_ids,desc="processing data2"):
#     # localTf=log_idf[q_id]
#     for tokenid in log_idf[q_id]:
#         indices[0].append(count_q)
#         indices[1].append(tokenid)
#         val.append(log_idf[q_id][tokenid])
#     convertedMap[q_id]=count_q
#     count_q+=1
# s = torch.sparse_coo_tensor(indices, val, (len(q_ids), numToken))    

# with open(f'output/log-doc-tf-tensor-dict-topk-{TopK}.pkl', 'wb') as f:
#     pickle.dump([convertedMap,s], f)