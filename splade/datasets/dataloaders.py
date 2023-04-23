"""
custom dataloaders (for dynamic batching)
"""

import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from ..utils.utils import rename_keys
import numpy as np

class DataLoaderWrapper(DataLoader):
    def __init__(self, tokenizer_type, max_length, **kwargs):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        super().__init__(collate_fn=self.collate_fn, **kwargs, pin_memory=True)

    def collate_fn(self, batch):
        raise NotImplementedError("must implement this method")


class SiamesePairsDataLoader(DataLoaderWrapper):
    """Siamese encoding (query and document independent)
    train mode (pairs)
    """

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 3 (text) items (q, d_pos, d_neg)
        """
        q, d_pos, d_neg = zip(*batch)
        q = self.tokenizer(list(q),
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True)
        d_pos = self.tokenizer(list(d_pos),
                               add_special_tokens=True,
                               padding="longest",  # pad to max sequence length in batch
                               truncation="longest_first",  # truncates to self.max_length
                               max_length=self.max_length,
                               return_attention_mask=True)
        d_neg = self.tokenizer(list(d_neg),
                               add_special_tokens=True,
                               padding="longest",  # pad to max sequence length in batch
                               truncation="longest_first",  # truncates to self.max_length
                               max_length=self.max_length,
                               return_attention_mask=True)
        sample = {**rename_keys(q, "q"), **rename_keys(d_pos, "pos"), **rename_keys(d_neg, "neg")}
        return {k: torch.tensor(v) for k, v in sample.items()}


class DistilSiamesePairsDataLoader(DataLoaderWrapper):

    def collate_fn(self, batch):
        """
        """
        q, d_pos, d_neg, s_pos, s_neg = zip(*batch)
        q = self.tokenizer(list(q),
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True)
        d_pos = self.tokenizer(list(d_pos),
                               add_special_tokens=True,
                               padding="longest",  # pad to max sequence length in batch
                               truncation="longest_first",  # truncates to self.max_length
                               max_length=self.max_length,
                               return_attention_mask=True)
        # Do you want to handle this in a different class
        d_neg = self.tokenizer(list(d_neg),
                               add_special_tokens=True,
                               padding="longest",  # pad to max sequence length in batch
                               truncation="longest_first",  # truncates to self.max_length
                               max_length=self.max_length,
                               return_attention_mask=True)

        sample = {**rename_keys(q, "q"), **rename_keys(d_pos, "pos"), **rename_keys(d_neg, "neg"),
                  "teacher_p_score": s_pos, "teacher_n_score": s_neg}
        return {k: torch.tensor(v) for k, v in sample.items()}


class      DistilSiamesePairsDataLoaderWithPsuedo(DataLoaderWrapper):

    def collate_fn(self, batch):
        """
        """
        q, d_pos, d_neg, s_pos, s_neg ,topicRep, docRep,psuedoDocRep= zip(*batch)
        topicRep=np.stack(topicRep)
        docRep=np.stack(docRep)
        psuedoDocRep=np.stack(psuedoDocRep)
        # corpus=self.dataset.corpus
        # Qcorpus=self.dataset.Qcorpus
        # Qcorpus2ndSparse=self.dataset.Qcorpus2ndSparse
        # corpus2ndSparse=self.dataset.corpus2ndSparse
        q = self.tokenizer(list(q),
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True)
        d_pos = self.tokenizer(list(d_pos),
                               add_special_tokens=True,
                               padding="longest",  # pad to max sequence length in batch
                               truncation="longest_first",  # truncates to self.max_length
                               max_length=self.max_length,
                               return_attention_mask=True)
        # Do you want to handle this in a different class
        d_neg = self.tokenizer(list(d_neg),
                               add_special_tokens=True,
                               padding="longest",  # pad to max sequence length in batch
                               truncation="longest_first",  # truncates to self.max_length
                               max_length=self.max_length,
                               return_attention_mask=True)

        sample = {**rename_keys(q, "q"), **rename_keys(d_pos, "pos"), **rename_keys(d_neg, "neg"),
                  "teacher_p_score": s_pos, "teacher_n_score": s_neg,\
                      "topicRep":topicRep,"docRep":docRep,"psuedoDocRep":psuedoDocRep,\
                        # "corpus":corpus,"Qcorpus":Qcorpus
                        }
                        # "corpus2ndSparse":corpus2ndSparse,"Qcorpus2ndSparse":Qcorpus2ndSparse}
        output={}
        for k, v in sample.items():
            if torch.is_tensor(v):
                output[k]=v
            else:
                output[k]=torch.tensor(v) 
        return output

class CollectionDataLoader(DataLoaderWrapper):
    """
    """

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to self.max_length
                                           max_length=self.max_length,
                                           return_attention_mask=True)
        return {**{k: torch.tensor(v) for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long)}


class TextCollectionDataLoader(DataLoaderWrapper):
    """same but also return the input text
    """

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to max model length,
                                           max_length=self.max_length,
                                           return_attention_mask=True)
        return {**{k: torch.tensor(v) for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long),
                "text": d
                }
