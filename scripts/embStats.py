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
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
from splade.utils.utils import restore_model

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
import pickle
class SparseEmbedding(SparseIndexing):
    def __init__(self,*param,**kwparam):
        super().__init__(*param,**kwparam)   
    def emb(self, collection_loader,storePath, id_dict=None):
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
            pickle.dump(embDoc, f,protocol=pickle.HIGHEST_PROTOCOL)
        return embDoc
# %%
model_training_config={}
model_training_config["tokenizer_type"]="experiments/PretrainedModel/splade_distil_CoCodenser_medium"
model_training_config["max_length"]=256

dataPath="data/msmarco/"
# dataPath="data/toy_data20/"
storePath=f"{dataPath}embStats/"
# %%
config={}
config["pretrained_no_yamlconfig"]=True
config["index_retrieve_batch_size"]=96
config["index_dir"]=None
COLLECTION_PATH=f"{dataPath}full_collection"

# %%
d_collection = CollectionDatasetPreLoad(data_dir=COLLECTION_PATH, id_style="row_id")
d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                max_length=model_training_config["max_length"],
                                batch_size=config["index_retrieve_batch_size"],
                                shuffle=False, num_workers=10, prefetch_factor=4)


# %%
evaluator = SparseEmbedding(model=model, config=config, compute_stats=True)

# %%
storePath=f"{dataPath}embStats/embDoc.pkl"
embDoc=evaluator.emb(d_loader,storePath=storePath)



