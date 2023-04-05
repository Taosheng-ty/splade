import gzip
import json
import os
import pickle
import random

from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch
import numpy as np

class PairsDatasetPreLoad(Dataset):
    """
    dataset to iterate over a collection of pairs, format per line: q \t d_pos \t d_neg
    we preload everything in memory at init
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.id_style = "row_id"

        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        print("Preloading dataset")
        self.data_dir = os.path.join(self.data_dir, "raw.tsv")
        with open(self.data_dir) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    query, pos, neg = line.split("\t")  # first column is id
                    self.data_dict[i] = (query.strip(), pos.strip(), neg.strip())
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        return self.data_dict[idx]


class DistilPairsDatasetPreLoad(Dataset):
    """
    dataset to iterate over a collection of pairs, format per line: q \t d_pos \t d_neg \t s_pos \t s_neg
    we preload everything in memory at init
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.id_style = "row_id"
        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        print("Preloading dataset")
        self.data_dir = os.path.join(self.data_dir, "raw.tsv")
        with open(self.data_dir) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    q, d_pos, d_neg, s_pos, s_neg = line.split("\t")
                    self.data_dict[i] = (
                        q.strip(), d_pos.strip(), d_neg.strip(), float(s_pos.strip()), float(s_neg.strip()))
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        return self.data_dict[idx]


class CollectionDatasetPreLoad(Dataset):
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    we preload everything in memory at init
    """

    def __init__(self, data_dir, id_style):
        self.data_dir = data_dir
        assert id_style in ("row_id", "content_id"), "provide valid id_style"
        # id_style indicates how we access the doc/q (row id or doc/q id)
        self.id_style = id_style
        self.data_dict = {}
        self.line_dict = {}
        print("Preloading dataset")
        with open(os.path.join(self.data_dir, "raw.tsv")) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    id_, *data = line.split("\t")  # first column is id
                    data = " ".join(" ".join(data).splitlines())
                    if self.id_style == "row_id":
                        self.data_dict[i] = data
                        self.line_dict[i] = id_.strip()
                    else:
                        self.data_dict[id_] = data.strip()
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        if self.id_style == "row_id":
            return self.line_dict[idx], self.data_dict[idx]
        else:
            return str(idx), self.data_dict[str(idx)]


class BeirDataset(Dataset):
    """
    dataset to iterate over a BEIR collection
    we preload everything in memory at init
    """

    def __init__(self, value_dictionary, information_type="document"):
        assert information_type in ["document", "query"]
        self.value_dictionary = value_dictionary
        self.information_type = information_type
        if self.information_type == "document":
            self.value_dictionary = dict()
            for key, value in value_dictionary.items():
                self.value_dictionary[key] = value["title"] + " " + value["text"]
        self.idx_to_key = {idx: key for idx, key in enumerate(self.value_dictionary)}

    def __len__(self):
        return len(self.value_dictionary)

    def __getitem__(self, idx):
        true_idx = self.idx_to_key[idx]
        return idx, self.value_dictionary[true_idx]


class MsMarcoHardNegatives(Dataset):
    """
    class used to work with the hard-negatives dataset from sentence transformers
    see: https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives
    """

    def __init__(self, dataset_path, document_dir, query_dir, qrels_path, *param,**kwparam):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        with gzip.open(dataset_path, "rb") as fIn:
            self.scores_dict = pickle.load(fIn)
        query_list = list(self.scores_dict.keys())
        with open(qrels_path) as reader:
            self.qrels = json.load(reader)
        self.query_list = list()
        for query in query_list:
            if str(query) in self.qrels.keys():
                self.query_list.append(query)
        print("QUERY SIZE = ", len(self.query_list))

    def __len__(self):
        return len(self.query_list)

    def __getitem__(self, idx,returnId=True):
        query = self.query_list[idx]
        q = self.query_dataset[str(query)][1]
        candidates_dict = self.scores_dict[query]
        candidates = list(candidates_dict.keys())
        positives = list(self.qrels[str(query)].keys())
        for positive in positives:
            candidates.remove(int(positive))
        positive = random.sample(positives, 1)[0]
        s_pos = candidates_dict[int(positive)]
        negative = random.sample(candidates, 1)[0]
        s_neg = candidates_dict[negative]
        d_pos = self.document_dataset[positive][1]
        d_neg = self.document_dataset[str(negative)][1]
        if returnId==True:
            return q.strip(), d_pos.strip(), d_neg.strip(), float(s_pos), float(s_neg), positive
        return q.strip(), d_pos.strip(), d_neg.strip(), float(s_pos), float(s_neg)



class MsMarcoHardNegativesWithPsuedo(MsMarcoHardNegatives):
    """
    class used to work with the hard-negatives dataset from sentence transformers
    see: https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives
    """
    def __init__(self,corpusStatsPath,queryStatsPath,psuedo_topk=10, *param,**kwparam):
        super().__init__(*param,**kwparam)
        with open(corpusStatsPath, 'rb') as f:
            CorpusStats=pickle.load(f)
        self.corpus_tf,self.doc_tf,self.psuedo_doc_tf=CorpusStats["corpus"],CorpusStats["pos_tf"],CorpusStats["psuedo_tf"]   
        with open (queryStatsPath, 'rb') as f:
            QueryStats=pickle.load(f)
        self.Qcorpus_tf,self.Qquery_tf=QueryStats["corpus"],QueryStats["pos_tf"]
        # self.cortf_Rep=self.cortf_Rep.type(torch.float32).cpu().numpy()
        numDocs=len(self.document_dataset)
        self.psuedo_topk=psuedo_topk
        self.numDocs=numDocs
        self.numVocab=len(self.corpus_tf)
        # self.ratio=np.sqrt(self.numDocs*self.psuedo_topk)  # avoid overflow 
        self.Qcorpus_tf=self.Qcorpus_tf.astype(np.float32)
        self.corpus_tf=self.corpus_tf.astype(np.float32)
    def __len__(self):
        return len(self.query_list)
    def convert_sparse2Dense(self,sparseRep):
        ind=list(sparseRep.keys())
        values=list(sparseRep.values())   
        DenseRep=np.zeros(self.numVocab).astype(np.float32)
        DenseRep[:]=0
        DenseRep[ind]=values   
        return DenseRep
        
    def __getitem__(self, idx):
        q,d_pos,d_neg,s_pos,s_neg,positive=super().__getitem__(idx,returnId=True)
        
        querySparse=self.Qquery_tf[positive]
        topicRep=self.convert_sparse2Dense(querySparse)
        
        query = self.query_list[idx]
        docSparse=self.doc_tf[str(query)]
        docRep=self.convert_sparse2Dense(docSparse)
        
        psuedoDocSparse=self.psuedo_doc_tf[str(query)]
        psuedoDocRep=self.convert_sparse2Dense(psuedoDocSparse)
        
        
        s_pos,s_neg=s_pos,s_neg
        
        return q,d_pos,d_neg,s_pos,s_neg,self.Qcorpus_tf,topicRep,self.corpus_tf,docRep,psuedoDocRep