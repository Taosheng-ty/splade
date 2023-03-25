import os,pickle,sys
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
import gzip,pickle
from collections import defaultdict
import torch
from collections import Counter
import json
import csv          
def List2Tsv(inp: list,outputDir='output.tsv'):
    directory=os.path.dirname(outputDir)
    os.makedirs(directory,exist_ok=True)
    with open(outputDir, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerows(inp)
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
sys.path.insert(0,scriptPath+"/..")
from splade.datasets.datasets import MsMarcoHardNegativesWithPsuedo
    
# inputDict={
#     "dataset_path": 'data/msmarco/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
#     "document_dir": 'data/msmarco/full_collection',
#     "query_dir":'data/msmarco/train_queries/queries',
#     "qrels_path":'data/msmarco/train_queries/qrels.json',
#     "queryRepFile":'output/log-doc-tf-topk-10.pkl',
#     "corpusRepFile":'output/corpus-tf-log-tensor-dictNP.pkl',
#     "psuedo_topk":10}

inputDict={
    "dataset_path": 'data/toy_data1k/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
    "document_dir": 'data/toy_data1k/full_collection',
    "query_dir":'data/toy_data1k/train_queries/queries',
    "qrels_path":'data/msmarco/train_queries/qrels.json',
    "queryRepFile":'data/toy_data1k/psuedo/log-doc-tf-topk-10.pkl',
    "corpusRepFile":'output/corpus-tf-log-tensor-dictNP.pkl',
    "psuedo_topk":10,
    # "toy":True
    }
dataset=MsMarcoHardNegativesWithPsuedo(**inputDict)

qselect=dataset.query_list[0:20]
NewName="toy_data20"
Toyquerys=[[str(qid),dataset.query_dataset.data_dict[str(qid)]] for qid in qselect]
List2Tsv(Toyquerys,outputDir=f"data/{NewName}/train_queries/queries/raw.tsv")

ToyDocs=[]
for qid,query in Toyquerys:
    ToyDocs+=list(dataset.scores_dict[str(qid)].keys())
ToyDocs=list(set(ToyDocs))
ToyDocs=[[did,dataset.document_dataset.data_dict[str(did)]]for did in ToyDocs]

List2Tsv(ToyDocs,outputDir=f"data/{NewName}/full_collection/raw.tsv")

subscoreDict={qid:dataset.scores_dict[str(qid)] for qid,query in Toyquerys}
import pickle
with open(f"data/{NewName}/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl", 'wb') as fIn:
    pickle.dump(subscoreDict, fIn)
os.system(f"gzip data/{NewName}/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl")
subsquer_Rep={qid:dataset.quer_Rep[str(qid)] for qid,query in Toyquerys}
import pickle
os.makedirs(f"data/{NewName}/psuedo",exist_ok=True)
with open(f"data/{NewName}/psuedo/log-doc-tf-topk-10.pkl", 'wb') as fIn:
    pickle.dump(subsquer_Rep, fIn)

