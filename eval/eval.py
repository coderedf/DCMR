#Base on the evaluation in AGAH
import torch
import torch.nn.functional as F

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


#calculation of MAP based on query_s from Bi-CMR
def MAP(query_code,
        database_code,
        query_S,
        device,
        topk=None
        ):
    num_q=query_S.shape[0]
    mAP=0.0

    for i in range(num_q):
        retrieval = query_S[i][num_q:]
        hamming_dist=0.5 * (database_code.shape[1] -query_code[i,:] @ database_code.t())
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]
        retrieval_cnt=retrieval.sum().int().item()
        if retrieval_cnt==0:
            continue
        score=torch.linspace(1,retrieval_cnt,retrieval_cnt).to(device)
        index=(torch.nonzero(retrieval==1).squeeze()+1.0).float()
        mAP+=(score / index).mean()
    mAP= mAP / num_q
    return mAP

#Ori-MAP
def OriMAP(query_code,
        database_code,
        query_labels,
        database_labels,
        device,
        topk=None,
        ):
    num_q=query_labels.shape[0]
    mAP=0.0

    for i in range(num_q):

        retrieval=(query_labels[i,:] @ database_labels.t() >0).float()
        retrieval[retrieval>1.0]=1.0   #只针对coco数据集合
        hamming_dist=0.5 * (database_code.shape[1] -query_code[i,:] @ database_code.t())
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]
        retrieval_cnt=retrieval.sum().int().item()
        if retrieval_cnt==0:
            continue

        score=torch.linspace(1,retrieval_cnt,retrieval_cnt).to(device)
        index=(torch.nonzero(retrieval==1).squeeze()+1.0).float()

        mAP+=(score / index).mean()
    mAP= mAP / num_q
    return mAP

#topK-precision curve
def calc_precisions_topn(qB, rB, query_S, topn):
    num_query = query_S.shape[0]

    precisions=0
    for iter in range(num_query):

        gnd = query_S[iter][:]
        gnd = torch.from_numpy(gnd)

        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]  #计算得到的
        right = torch.nonzero(gnd[: topn]).squeeze().numpy()
        right_num = right.size
        precisions+= (right_num/topn)
    precisions /= num_query
    return precisions

#function of PR curve
def pr_curve1(qB, rB, query_S):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = query_S[i][:]
        gnd = torch.from_numpy(gnd)

        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).type(torch.cuda.FloatTensor).cpu
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd.cpu().float() * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R
