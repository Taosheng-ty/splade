# %%
import sys
# sys.path.append("../")
import os
import numpy as np
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
sys.path.insert(0,".")
# %%
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
from splade.utils.utils import restore_model

# docType=sys.argv[1]
# assert docType in ["doc",'query'],"must be in [doc,query]"

# %%
model_type_or_dir = "experiments/PretrainedModel/splade_distil_CoCodenser_medium/"
# /home/taoyang/research/research_everyday/projects/DR/splade/splade/experiments/toyContrast/AQWPsuedotoy/checkpoint/model_ckpt
# /home/collab/u1368791/largefiles/TaoFiles/splade/experiments/cocondenser_ensemble_distil/checkpoint/model
model = Splade(model_type_or_dir, agg="max")

# %%
from splade.utils.utils import makedir, to_list
from tqdm.auto import tqdm
from collections import defaultdict
from splade.tasks.transformer_evaluator import SparseIndexing
from splade.datasets.datasets import CollectionDatasetPreLoad
from splade.datasets.dataloaders import CollectionDataLoader
from splade.datasets.datasets import PairsDatasetPreLoad, DistilPairsDatasetPreLoad, MsMarcoHardNegatives, \
    CollectionDatasetPreLoad
import pickle
model_type_or_dir="distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
numToken=tokenizer.vocab_size

class SparseEmbedding(SparseIndexing):
    def __init__(self,*param,**kwparam):
        super().__init__(*param,**kwparam)   
    def emb(self, collection_loader,storePath,id_dict=None):
        doc_ids = []
        embDoc={}
        if self.compute_stats:
            stats = defaultdict(float)
        count = 0
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader,desc="indexing")):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"id"}}
                if self.is_query:
                    batch_documents = self.model(q_kwargs=inputs)["q_rep"]
                else:
                    batch_documents = self.model(d_kwargs=inputs)["d_rep"]
                if self.compute_stats:
                    stats["L0_d"] += self.l0(batch_documents).item()
                row, col = torch.nonzero(batch_documents, as_tuple=True)

                data = batch_documents[row, col].cpu().numpy().astype(np.float16)
                row=to_list(row)
                col=to_list(col)
                # row = row + count
                # print(batch["id"])
                batch_ids = to_list(batch["id"])
                # print(batch_ids)
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                # print(row)
                for rowCur in row:
                    embDoc[str(batch_ids[rowCur])]=defaultdict(float)
                for ind,(rowCur,colCur) in enumerate(zip(row, col)):
                    embDoc[str(batch_ids[rowCur])][colCur]=data[ind]
        directory=os.path.dirname(storePath)
        os.makedirs(directory,exist_ok=True)
        with open(storePath, 'wb') as f:
            pickle.dump({"pos_tf":embDoc}, f,protocol=pickle.HIGHEST_PROTOCOL)
        return embDoc
# %%
model_training_config={}
model_training_config["tokenizer_type"]="experiments/PretrainedModel/splade_distil_CoCodenser_medium"
model_training_config["max_length"]=256
docTypes=['query',"doc"]
docTypes=["doc"]
dataPath="data/msmarco/"
# dataPath="data/toy_data20/"
# dataPath="data/toy_data1k/"
storePath=f"{dataPath}embStats/"
# %%
config={}
config["pretrained_no_yamlconfig"]=True

config["index_dir"]=None
folerNameDict={"query":"train_queries/queries","doc":"full_collection"}
NameDict={"query":"QueryStats","doc":"CorpusStats"}

inputDict={
    "dataset_path": f"{dataPath}/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz",
    "document_dir": f"{dataPath}/full_collection",
    "query_dir":f"{dataPath}/train_queries/queries",
    "qrels_path":'data/msmarco/train_queries/qrels.json',
    }
dataset=MsMarcoHardNegatives(**inputDict)
InverseVocab={key[1]:key[0] for ind,key in enumerate(tokenizer.vocab.items())}
numToken=len(tokenizer.vocab.keys())
for docType in docTypes:
    COLLECTION_PATH=f"{dataPath}{folerNameDict[docType]}"
    config["index_retrieve_batch_size"]=512 if docType=="query" else 96
    # %%
    d_collection = CollectionDatasetPreLoad(data_dir=COLLECTION_PATH, id_style="row_id")
    d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                    max_length=model_training_config["max_length"],
                                    batch_size=config["index_retrieve_batch_size"],
                                    shuffle=False, num_workers=10, prefetch_factor=4)


    # %%
    evaluator = SparseEmbedding(model=model, config=config, compute_stats=True)

    # %%
    storePath=f"{dataPath}embStats/emb_{NameDict[docType]}.pkl"
    embDoc=evaluator.emb(d_loader,storePath=storePath)
    log_tf=np.zeros(numToken)
    from collections import Counter
    for idx in tqdm(embDoc,desc="processing data corpus"):
        # print(idx)
        tokenSent=embDoc[idx]
        for token in tokenSent:
            value=tokenSent[token]
            log_tf[token]+=value
    logPosdoc={}
    logPsuedoPosdoc={}
    TopK=10
    FirstKey=list(dataset.scores_dict.keys())[0]
    convertFcn=str if isinstance(FirstKey,str) else int
    qrels=dataset.qrels
    
    if docType=="doc":
        for qid in tqdm(qrels,desc="processing query"):
            if convertFcn(qid) not in dataset.scores_dict:
                continue
            posDocids=list(qrels[qid].keys())
            # if len(posDocids)>0:
            logPosdoc[qid]=defaultdict(float)
            for doc in posDocids:
                if qrels[qid][doc]>0:
                    tokenSent=embDoc[doc]
                    for token in tokenSent:
                        value=tokenSent[token]
                        logPosdoc[qid][token]+=value
            DistillCandidates=dataset.scores_dict[convertFcn(qid)]
            DistillCandidates = sorted(DistillCandidates,key=DistillCandidates.get,reverse=True)[:TopK]
            while len(posDocids)<TopK:
                posDocids.append(str(DistillCandidates.pop(0)))
            logPsuedoPosdoc[qid]=defaultdict(float)
            for doc in posDocids:
                tokenSent=embDoc[doc]
                for token in tokenSent:
                    value=tokenSent[token]
                    logPsuedoPosdoc[qid][token]+=value
        storePath=f"{dataPath}embStats/{NameDict[docType]}.pkl"    
        with open(storePath, 'wb') as f:
            pickle.dump({"corpus":log_tf,"pos_tf":logPosdoc,"psuedo_tf":logPsuedoPosdoc}, f,protocol=pickle.HIGHEST_PROTOCOL)
    
    
    else:  
        for idx in tqdm(embDoc,desc="processing data corpus"):
            # print(idx)
            tokenSent=embDoc[idx]
            for token in tokenSent:
                value=tokenSent[token]
                log_tf[token]+=value
            if idx not in qrels:
                continue
            for doc in qrels[idx]:
                if qrels[idx][doc]>0:
                    if doc not in logPosdoc:
                        logPosdoc[doc]=defaultdict(float)
                    for token in tokenSent:
                        value=tokenSent[token]
                        logPosdoc[doc][token]+=value
        storePath=f"{dataPath}embStats/{NameDict[docType]}.pkl"    
        with open(storePath, 'wb') as f:
            pickle.dump({"corpus":log_tf,"pos_tf":logPosdoc}, f, protocol=pickle.HIGHEST_PROTOCOL)
