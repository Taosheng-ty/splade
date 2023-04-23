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
        self.contrastfcn=self.contrast
        self.InBatchPairwiseNLL=InBatchPairwiseNLL()
        if "contrast" in kwparam:
            contrastfcns={"contrast":self.contrast,"contrastv1":self.contrastv1}
            self.contrastfcn=contrastfcns[kwparam["contrast"]]
        super().__init__(*param,**kwparam)
    def contrast(self,q_rep,topic,corpus,numDocs,numQ=1,*param,**kwparam):
        # loss=-torch.log((torch.sum(q_rep * topic, dim=-1)+numQ)/(torch.sum(q_rep * corpus, dim=-1)+numDocs))
        loss=-torch.log(torch.sum(q_rep * topic, dim=-1)/numQ+1)+torch.log((torch.sum(q_rep * corpus, dim=-1)/numDocs+1))
        return loss
    def contrastv1(self,q_rep,topic,corpus,numDocs,numQ=1,corpus2nd=None,*param,**kwparam):
        score=torch.sum(q_rep * topic, dim=-1,keepdim=True)
        corpusScore1stOrder=torch.sum(q_rep * corpus, dim=-1,keepdim=True)
        corpusScore2ndTemp= torch.sparse.mm(corpus2nd,q_rep.T)
        secondordSum=torch.sum(corpusScore2ndTemp.T*q_rep,-1,keepdim=True)
        
        loss=-torch.log(1+score+1/2*score**2)+torch.log(numDocs+corpusScore1stOrder+1/2*secondordSum)
        return loss
    def __call__(self, out_d):
        """out_d also contains scores from teacher
        """
        # assert "lambda_psuedo" in out_d and "lambda_hard" in out_d
        Loss={}
        MatchLoss=0
        # corpus=out_d["corpus"]+torch.sum(out_d['pos_d_rep'],dim=0,keepdim=True)+torch.sum(out_d['neg_d_rep'],dim=0,keepdim=True)
        # qcorpus=out_d["Qcorpus"]+torch.sum(out_d['pos_q_rep'],dim=0,keepdim=True)
        # corpus=out_d["corpus"]
        # qcorpus=out_d["Qcorpus"]
        if "lambda_hard" in out_d  and out_d["lambda_hard"]>0:
            HardLoss=super().__call__(out_d)
            Loss["HardLoss"]=out_d["lambda_hard"]*HardLoss
            MatchLoss+=Loss["HardLoss"]
        if "lambda_psuedo" in out_d and  out_d["lambda_psuedo"]>0:
            # Psuedo_loss=-(out_d['pos_q_rep']*(out_d["topic_Rep"])).sum(dim=1)
            q=out_d['pos_q_rep']
            # d=out_d['pos_d_rep']
            topic=out_d["psuedoDocRep"]
            corpus=out_d["corpus"]
            # Psuedo_loss=-torch.log((torch.sum(q * topic, dim=-1)+self.psuedo_topk)/(torch.sum(q * corpus, dim=-1)+self.numDocs))
            Psuedo_loss=self.contrastfcn(q,topic,corpus,self.numDocs,self.psuedo_topk)
            Psuedo_loss=Psuedo_loss.mean()
            Loss["PsuedoLoss"]=out_d["lambda_psuedo"]*Psuedo_loss
            MatchLoss+=Loss["PsuedoLoss"]
        if "lambda_Query" in out_d and out_d["lambda_Query"]>0:
            d=out_d['pos_d_rep']
            # topic=out_d["topicRep"]
            # q=out_d['pos_q_rep']
            topic=out_d["topicRep"] 
            qcorpusCur=out_d["model"].q_corpus
            q_corpus2ndSparse=out_d["model"].q_corpus2ndSparse
            QPsuedo_loss=self.contrastfcn(d,topic,qcorpusCur,self.numQueries,1,corpus2nd=q_corpus2ndSparse)
            QPsuedo_loss=QPsuedo_loss.mean()
            Loss["QPsuedoLoss"]=out_d["lambda_Query"]*QPsuedo_loss
            MatchLoss+=Loss["QPsuedoLoss"]            
        if "lambda_Doc" in out_d and out_d["lambda_Doc"]>0:
            #  "topic_Rep":topic_Rep,"corpus":corpus,"psuedoDocRep":psuedoDocRep
            q=out_d['pos_q_rep']
            # d=out_d['pos_d_rep']
            # topic=out_d["topic_Rep"]
            topic=out_d["docRep"]
            corpus=out_d["model"].d_corpus
            d_corpus2ndSparse=out_d["model"].d_corpus2ndSparse
            # ratio=d.max().detach()/topic.max()
            # topic=topic*ratio          
            # corpusCur=ratio+torch.sum(out_d['pos_d_rep'],dim=0,keepdim=True)+torch.sum(out_d['neg_d_rep'],dim=0,keepdim=True)
            # DPsuedo_loss=-torch.log((torch.sum(q * topic, dim=-1)+1)/(torch.sum(q * corpus, dim=-1)+self.numDocs))
            DPsuedo_loss=self.contrastfcn(q,topic,corpus,self.numDocs,1,corpus2nd=d_corpus2ndSparse)
            DPsuedo_loss=DPsuedo_loss.mean()
            Loss["DPsuedoLoss"]=out_d["lambda_Doc"]*DPsuedo_loss
            MatchLoss+=Loss["DPsuedoLoss"] 
        if "inBatch" in out_d and out_d["inBatch"]>0:
            inBatchLoss=self.InBatchPairwiseNLL(out_d)
            Loss["inBatch"]=out_d["inBatch"]*inBatchLoss
            MatchLoss+=Loss["inBatch"] 
        Loss["MatchLoss"]=MatchLoss
        return Loss  


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
