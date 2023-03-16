import os,pickle
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
import gzip,pickle
from collections import defaultdict
from collections import Counter

scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
model_type_or_dir="distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)



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
# tokenCop=tokenizer(listCorp, return_tensors="np")['input_ids']
# InverseVocab={key[1]:key[0] for ind,key in enumerate(tokenizer.vocab.items())}
# numToken=len(tokenizer.vocab.keys())

dataset_path="data/msmarco/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
with gzip.open(dataset_path, "rb") as fIn:
    scores_dict = pickle.load(fIn)
TopK=10
log_idf={}
q_ids=list(scores_dict.keys())

for q_id in tqdm(q_ids,desc="processing data"):
    candidates=scores_dict[q_id]
    p_ids = sorted(candidates,key=candidates.get,reverse=True)[:TopK]
    log_idf[q_id]=defaultdict(int)
    for p_id in p_ids:
        tokenSent=tokenCop[p_id]
        FreqTokenSent=Counter(tokenSent)
        for token in FreqTokenSent:
            freq=FreqTokenSent[token]
            if np.log(freq)<=0:
                continue
            log_idf[q_id][token]+=np.log(freq)

with open(f'output/log-doc-tf-topk-{TopK}.pkl', 'wb') as f:
    pickle.dump(log_idf, f)
        