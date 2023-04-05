import torch

"""general API for losses: the __call__ method receives out_d, a dict containing at least scores for positives 
and negatives  
"""


class PairwiseNLL:
    def __init__(self,*param,**kwparam):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def __call__(self, out_d):
        pos_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        scores = self.logsoftmax(torch.cat([pos_scores, neg_scores], dim=1))
        return torch.mean(-scores[:, 0])


class InBatchPairwiseNLL:
    """in batch negatives version
    """

    def __init__(self,*param,**kwparam):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def __call__(self, out_d):
        in_batch_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        # here in_batch_scores is a matrix of size bs * (bs / nb_gpus)
        nb_columns = in_batch_scores.shape[1]
        nb_gpus = int(in_batch_scores.shape[0] / nb_columns)
        temp = torch.cat([in_batch_scores, neg_scores], dim=1)  # concat neg score from BM25 sampling
        # shape (batch_size, batch_size/nb_gpus + 1)
        scores = self.logsoftmax(temp)
        return torch.mean(-scores[torch.arange(in_batch_scores.shape[0]),
                                  torch.arange(nb_columns).repeat(nb_gpus)])


class PairwiseBPR:
    """BPR loss from: http://webia.lip6.fr/~gallinar/gallinari/uploads/Teaching/WSDM2014-rendle.pdf
    """

    def __init__(self,*param,**kwparam):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def __call__(self, out_d):
        pos_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        return self.loss((pos_scores - neg_scores).squeeze(), torch.ones(pos_scores.shape[0]).to(self.device))


class DistilMarginMSE:
    """MSE margin distillation loss from: Improving Efficient Neural Ranking Models with Cross-Architecture
    Knowledge Distillation
    link: https://arxiv.org/abs/2010.02666
    """

    def __init__(self,*param,**kwparam):
        self.loss = torch.nn.MSELoss()

    def __call__(self, out_d):
        """out_d also contains scores from teacher
        """
        pos_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        teacher_pos_scores, teacher_neg_scores = out_d["teacher_pos_score"], out_d["teacher_neg_score"]
        margin = pos_scores - neg_scores
        teacher_margin = teacher_pos_scores - teacher_neg_scores
        return self.loss(margin.squeeze(), teacher_margin.squeeze())  # forces the margins to be similar


class HybridLoss(DistilMarginMSE):
    """MSE margin distillation loss from: Improving Efficient Neural Ranking Models with Cross-Architecture
    Knowledge Distillation
    link: https://arxiv.org/abs/2010.02666
    """

    def __init__(self,numDocs,numQueries,psuedo_topk,*param,**kwparam):
        self.numDocs=numDocs
        self.numQueries=numQueries
        self.psuedo_topk=psuedo_topk
        super().__init__(*param,**kwparam)
    def __call__(self, out_d):
        """out_d also contains scores from teacher
        """
        # assert "lambda_psuedo" in out_d and "lambda_hard" in out_d
        loss=0
        if "lambda_hard" in out_d  and out_d["lambda_hard"]>0:
            Marginal_loss=super().__call__(out_d)
            loss+=out_d["lambda_hard"]*Marginal_loss
        if "lambda_psuedo" in out_d and  out_d["lambda_psuedo"]>0:
            # Psuedo_loss=-(out_d['pos_q_rep']*(out_d["topic_Rep"])).sum(dim=1)
            q=out_d['pos_q_rep']
            topic=out_d["psuedo_topic_Rep"]
            corpus=out_d["cortf_Rep"]
            Psuedo_loss=-(torch.sum(q * topic, dim=-1)+self.psuedo_topk)/(torch.sum(q * corpus, dim=-1)+self.numDocs)
            Psuedo_loss=Psuedo_loss.sum()
            loss+=out_d["lambda_psuedo"]*Psuedo_loss
        if "lambda_Query" in out_d and out_d["lambda_Query"]>0:
            d=out_d['pos_d_rep']
            topic=out_d["Qtopic_Rep"]
            corpus=out_d["Qcortf_Rep"]
            QPsuedo_loss=-(torch.sum(d * topic, dim=-1)+1)/(torch.sum(d * corpus, dim=-1)+self.numQueries)
            QPsuedo_loss=QPsuedo_loss.sum()
            loss+=out_d["lambda_Query"]*QPsuedo_loss
            
        if "lambda_Doc" in out_d and out_d["lambda_Doc"]>0:
            #  "topic_Rep":topic_Rep,"cortf_Rep":cortf_Rep,"psuedo_topic_Rep":psuedo_topic_Rep
            q=out_d['pos_q_rep']
            topic=out_d["topic_Rep"]
            corpus=out_d["cortf_Rep"]
            QPsuedo_loss=-(torch.sum(q * topic, dim=-1)+1)/(torch.sum(q * corpus, dim=-1)+self.numDocs)
            QPsuedo_loss=QPsuedo_loss.sum()
            loss+=out_d["lambda_Doc"]*QPsuedo_loss
        return loss  # forces the margins to be similar



class DistilKLLoss:
    """Distillation loss from: Distilling Dense Representations for Ranking using Tightly-Coupled Teachers
    link: https://arxiv.org/abs/2010.11386
    """

    def __init__(self,*param,**kwparam):
        self.loss = torch.nn.KLDivLoss(reduction="none")

    def __call__(self, out_d):
        pos_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        teacher_pos_scores, teacher_neg_scores = out_d["teacher_pos_score"], out_d["teacher_neg_score"]
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        local_scores = torch.log_softmax(scores, dim=1)
        teacher_scores = torch.cat([teacher_pos_scores.unsqueeze(-1), teacher_neg_scores.unsqueeze(-1)], dim=1)
        teacher_scores = torch.softmax(teacher_scores, dim=1)
        return self.loss(local_scores, teacher_scores).sum(dim=1).mean(dim=0)
