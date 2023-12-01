import torch
import model.models as ITnet
import torch.nn.functional as F
import eval.eval as eva
from loguru import logger
import time
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # change

def InfoNCE(x, y, temperature=0.07):
    # Compute dot products between x and y and exponentiate
    dot_product = F.cosine_similarity(x.unsqueeze(1), y, dim=-1) / temperature
    exp_dot_product = torch.exp(dot_product)

    # Compute numerator and denominator
    numerator = torch.diag(exp_dot_product)
    denominator = torch.sum(exp_dot_product, dim=1)

    # Compute probability ratio and take logarithm
    prob_ratio = numerator / denominator
    log_prob_ratio = torch.log(prob_ratio)

    # Compute negative mean of log probability ratio
    loss = -torch.mean(log_prob_ratio)

    return loss

def InfoNCE_1(img_hash_c, txt_hash_c, img_hash_p, txt_hash_p, temperature=0.07):
    #use the specific hash codes within same instance as the negative point
    dot_product = F.cosine_similarity(img_hash_c.unsqueeze(1), txt_hash_c, dim=-1) / temperature
    exp_dot_product = torch.exp(dot_product)

    dot_product_i = F.cosine_similarity(img_hash_c.unsqueeze(1), img_hash_p, dim=-1) / temperature
    exp_dot_product_i = torch.exp(dot_product_i)
    dot_product_it = F.cosine_similarity(img_hash_c.unsqueeze(1), txt_hash_p, dim=-1) / temperature
    exp_dot_product_it = torch.exp(dot_product_it)
    dot_product_t = F.cosine_similarity(txt_hash_c.unsqueeze(1), txt_hash_p, dim=-1) / temperature
    exp_dot_product_t = torch.exp(dot_product_t)
    dot_product_ti = F.cosine_similarity(txt_hash_c.unsqueeze(1), img_hash_p, dim=-1) / temperature
    exp_dot_product_ti = torch.exp(dot_product_ti)

    # Compute numerator and denominator
    numerator = torch.diag(exp_dot_product)
    denominator = torch.diag(exp_dot_product) + torch.diag(exp_dot_product_i) +torch.diag(exp_dot_product_it)+\
                  torch.diag(exp_dot_product_t) +torch.diag(exp_dot_product_ti)

    # Compute probability ratio and take logarithm
    prob_ratio = numerator / denominator
    log_prob_ratio = torch.log(prob_ratio)

    # Compute negative mean of log probability ratio
    loss = -torch.mean(log_prob_ratio)

    return loss

def train(opt,train_loader,
              test_loader,
              dataset_loader,
              label_train_loader,
              ):

    #load model
    img_model,txt_model,label_model=ITnet.load_model(opt.mode_name,opt.code_length,opt.label_size,opt.input_size,opt.model)
    img_model.to(opt.device)
    txt_model.to(opt.device)
    label_model.to(opt.device)

    #parameters of model & optimizer of parameters
    img_params=list(img_model.parameters())
    txt_params=list(txt_model.parameters())
    label_params=list(label_model.parameters())

    img_optimizer = torch.optim.Adam(img_params,lr=opt.lr,betas =(0.5, 0.999))
    txt_optimizer=torch.optim.Adam(txt_params,lr=opt.lr,betas =(0.5, 0.999))
    label_optimizer = torch.optim.Adam(label_params, lr=opt.lr*10, betas =(0.5, 0.999))


    #Initialization
    loss=0
    avg_txt_img,average_map=0,0
    start = time.time()

    #Training epoch begin (Training code.)
    for epoch in range(opt.num_epochs):
        label_model.train()
        img_model.train()
        txt_model.train()
        #multi-label network initialization (according to SSAH)
        for i,(img, txt, labels, S, ind) in enumerate(train_loader):
            labels=labels.type(torch.FloatTensor).to(opt.device)

            #img & txt data but not training
            txt_trains = txt.to(opt.device)
            img_trains = img.to(opt.device)

            img_hash_c, img_hash_p, img_predict, img_cf, img_pf, img_f = img_model(img_trains)
            txt_hash_c, txt_hash_p, txt_predict, txt_cf, txt_pf, txt_f = txt_model(txt_trains)
            feature, label_hash_code, _ = label_model(labels)


            #self_supervised basic loss of labelNet according to SSAH & Bi-CMR
            cos_l = F.cosine_similarity(labels.unsqueeze(1), labels, dim=-1)
            cos_l_h = F.cosine_similarity(label_hash_code.unsqueeze(1), label_hash_code, dim=-1)
            loss_l = ((cos_l-cos_l_h)**2).mean()
            loss1 = ((label_hash_code @ label_hash_code.t()-opt.code_length*cos_l) **2).mean()

            img_sim = F.cosine_similarity(img_hash_c.unsqueeze(1), img_hash_c, dim=-1)
            txt_sim = F.cosine_similarity(txt_hash_c.unsqueeze(1), txt_hash_c, dim=-1)
            #normalization is a optional
            img_sim_norm = (img_sim - torch.min(img_sim)) / (torch.max(img_sim) - torch.min(img_sim))
            txt_sim_norm = (txt_sim - torch.min(txt_sim)) / (torch.max(txt_sim) - torch.min(txt_sim))
            label_sim_norm = (cos_l_h - torch.min(cos_l_h)) / (torch.max(cos_l_h) - torch.min(cos_l_h))
            eta = opt.eta
            # Adding the square maybe have a better effect
            a_img = img_sim_norm**2
            a_txt = txt_sim_norm**2
            a = eta * a_img + (1 - eta) * a_txt

            #first part of purify loss
            loss_purify1 = (a / (1 - torch.log(1.001 - label_sim_norm))).sum(1).mean()
            #second part of purify loss
            loss_purify2= (label_sim_norm / (1 - torch.log(1.001 - 0/5*(a)))).sum(1).mean()

            #triple loss
            loss_triple1 = ((img_hash_c @ label_hash_code.t() - opt.code_length * cos_l_h) ** 2).mean() \
                            + ((txt_hash_c @ label_hash_code.t() - opt.code_length * cos_l_h) ** 2).mean() \
                            + ((label_hash_code @ label_hash_code.t() - opt.code_length * cos_l_h) ** 2).mean()\
                            + ((img_hash_c @ txt_hash_c.t() - opt.code_length * cos_l_h) ** 2).mean()

            loss_triple2 = -torch.log(torch.diag(opt.code_length * cos_l_h - img_hash_p @ label_hash_code.t()) ** 2).mean() \
                              - torch.log(torch.diag(opt.code_length * cos_l_h - txt_hash_p @ label_hash_code.t() ) ** 2).mean() \
                              - torch.log(torch.diag(opt.code_length * cos_l_h - label_hash_code @ label_hash_code.t() ) ** 2).mean() \
                              - torch.log(torch.diag(opt.code_length * cos_l_h - img_hash_p @ txt_hash_p.t() ) ** 2).mean()


            loss = (1000*loss_l + loss1) + 0.001*(loss_purify1+loss_purify2) + loss_triple1 + loss_triple2

            # only label optimization
            label_optimizer.zero_grad()
            loss.backward()
            label_optimizer.step()

        # training of disentangled image and text networks
        for i,(img, txt, labels,S, ind) in enumerate(train_loader):
            txt_trains = txt.to(opt.device)
            img_trains = img.to(opt.device)
            labels=labels.to(opt.device).type(torch.cuda.FloatTensor)

            #todo: output of each model
            img_hash_c, img_hash_p, img_predict, img_cf, img_pf, img_f = img_model(img_trains)
            txt_hash_c, txt_hash_p, txt_predict, txt_cf, txt_pf, txt_f = txt_model(txt_trains)
            feature, label_hash_code, _ = label_model(labels)

            Sim = F.cosine_similarity(label_hash_code.unsqueeze(1), label_hash_code, dim=-1)
            # loss_triple
            loss_triple1 = ((img_hash_c @ label_hash_code.t() - opt.code_length * Sim) ** 2).mean() \
                            + ((txt_hash_c @ label_hash_code.t() - opt.code_length * Sim) ** 2).mean() \
                            + ((label_hash_code @ label_hash_code.t() - opt.code_length * Sim) ** 2).mean()\
                            + ((img_hash_c @ txt_hash_c.t()-opt.code_length * Sim) ** 2).mean()

            loss_triple2 = -torch.log(opt.code_length * Sim - torch.diag(img_hash_p @ label_hash_code.t()) ** 2).mean() \
                            - torch.log(opt.code_length * Sim - torch.diag(txt_hash_p @ label_hash_code.t() ) ** 2).mean() \
                            - torch.log(opt.code_length * Sim - torch.diag(label_hash_code @ label_hash_code.t() ) ** 2).mean()\
                            - torch.log(opt.code_length * Sim - torch.diag(img_hash_p @ txt_hash_p.t()) ** 2).mean()

            loss_triple = loss_triple1 + 0.1*loss_triple2

            #loss contrastive
            loss_contrastive_c = InfoNCE(img_hash_c, txt_hash_c)

            loss_contrastive_s = InfoNCE_1(img_hash_c, txt_hash_c, img_hash_p, txt_hash_p)

            #loss Ex
            loss_Ex = ((img_hash_p @ txt_hash_p.t())**2).mean()

            img_sim = F.cosine_similarity(img_hash_c.unsqueeze(1), img_hash_c, dim=-1)
            txt_sim = F.cosine_similarity(txt_hash_c.unsqueeze(1), txt_hash_c, dim=-1)
            img_txt_sim = F.cosine_similarity(img_hash_c.unsqueeze(1), txt_hash_c, dim=-1)
            label_sim = F.cosine_similarity(label_hash_code.unsqueeze(1), label_hash_code, dim=-1)

            img_label_sim = F.cosine_similarity(img_hash_c.unsqueeze(1), label_hash_code, dim=-1)
            txt_label_sim = F.cosine_similarity(txt_hash_c.unsqueeze(1), label_hash_code, dim=-1)

            # normalization is a optional
            img_sim_norm = (img_sim - torch.min(img_sim)) / (torch.max(img_sim) - torch.min(img_sim))
            txt_sim_norm = (txt_sim - torch.min(txt_sim)) / (torch.max(txt_sim) - torch.min(txt_sim))
            img_txt_sim_norm = (img_txt_sim - torch.min(img_txt_sim)) / (torch.max(img_txt_sim) - torch.min(img_txt_sim))
            label_sim_norm = (label_sim - torch.min(label_sim)) / (torch.max(label_sim) - torch.min(label_sim))


            img_label_sim_norm = (img_label_sim - torch.min(img_label_sim)) / (torch.max(img_label_sim) - torch.min(img_label_sim))
            txt_label_sim_norm = (txt_label_sim - torch.min(txt_label_sim)) / (torch.max(txt_label_sim) - torch.min(txt_label_sim))


            loss_refine = ((torch.exp(img_sim + txt_sim) - torch.exp(label_sim)) ** 2).sum(1).mean()
            #=============================
            # optional
            labels_pre = labels/labels.sum(1).unsqueeze(1).expand(labels.shape[0],opt.label_size)
            loss_class = F.kl_div(img_predict.log(),labels_pre,reduction='sum')+F.kl_div(txt_predict.log(),labels_pre,reduction='sum')

            loss= loss_triple + (loss_contrastive_c + 0.1*loss_contrastive_s) + 0.1*loss_Ex + 0.001*loss_refine

            #only modal optimization
            img_optimizer.zero_grad()
            txt_optimizer.zero_grad()
            loss.backward()
            img_optimizer.step()
            txt_optimizer.step()


        #learning_rate
        if epoch % opt.iter == 0:
            for params in img_optimizer.param_groups:
                params['lr'] = max(params['lr'] * 0.5, 1e-6)
            for params in txt_optimizer.param_groups:
                params['lr'] = max(params['lr'] * 0.5, 1e-6)
            for params in label_optimizer.param_groups:
                params['lr'] = max(params['lr'] * 0.5, 1e-6)

        #Evaluation code (test)
        ori_t2imap = test(True, dataset_loader, test_loader, img_model, txt_model, opt.device,
                                           opt.code_length, opt.topK, label_model)
        ori_i2tmap = test(False, dataset_loader, test_loader, img_model, txt_model, opt.device,
                                           opt.code_length, opt.topK, label_model)
        # '[loss_m2m: {:.4f}'loss_m2m.item(),
        logger.info('[itr: {}][time:{:.4f}]'
                    '[I2T_ori_map: {:.4f}]'
                    '[T2I_ori_map: {:.4f}]'.format(epoch + 1,
                                                   time.time() - start,
                                                   ori_i2tmap, ori_t2imap))
        start = time.time()
        loss = 0

        # save model path
        if (ori_i2tmap + ori_t2imap) / 2 > average_map:
            average_map = (ori_i2tmap + ori_t2imap) / 2
            print("ori_avg_map:", average_map)
            # save the model
            txt_file_name = 'txt_net_{}_length_{}.t'.format(opt.dataname,opt.code_length)
            img_file_name = 'img_net_{}_length_{}.t'.format(opt.dataname,opt.code_length)
            label_file_name = 'label_net_{}_length_{}.t'.format(opt.dataname,opt.code_length)
            torch.save(txt_model, os.path.join('result_modal', txt_file_name))
            torch.save(img_model, os.path.join('result_modal', img_file_name))
            torch.save(label_model, os.path.join('result_modal', label_file_name))




# test function (Evaluation code.)
def test(t2i,data_loader,test_loader,img_model,txt_model,device,code_length,topK,label_model):
    img_model.eval()
    txt_model.eval()
    label_model.eval()

    # Original evaluation needs labels
    query_labels = torch.FloatTensor(test_loader.dataset.labels).to(device)
    database_labels = torch.FloatTensor(data_loader.dataset.labels).to(device)

    if t2i: #text query images
        query_S = torch.FloatTensor(test_loader.dataset.S_data).to(device)
        query_code = code_(False,txt_model, test_loader, code_length, device).to(device)
        #database_code=code_(True,img_model,data_loader, code_length, device).to(device)

    else: #image query texts
        query_S = torch.FloatTensor(test_loader.dataset.S_data).to(device)
        query_code = code_(True,img_model, test_loader, code_length, device).to(device)
        #database_code = code_(False,txt_model, data_loader, code_length, device).to(device)

    database_code = label_code_(label_model,data_loader, code_length, device).to(device)

    #calculate the result of MAP  based on query_s from Bi-CMR
    meanAP = eva.MAP(query_code, database_code, query_S, device,topK)

    #calculate the result of original MAP
    OrimAP = eva.OriMAP(query_code, database_code, query_labels, database_labels, device, topK)
    return meanAP

#generate binary hash codes of modal network
def code_(img,model,dataloader,code_length,device):
    with torch.no_grad():
        num=len(dataloader.dataset)
        code=torch.zeros([num,code_length])
        if img:
            for i,(trains,_,_,_,index) in enumerate(dataloader):
                trains=trains.to(device)
                outputs,_,_,_,_,_=model(trains)
                code[index,:]=outputs.sign().cpu()
        else:
            for i,(_,trains,_,_,index) in enumerate(dataloader):

                trains=trains.to(device)
                outputs,_,_,_,_,_=model(trains)
                code[index,:]=outputs.sign().cpu()
    return code


def label_code_(model,dataloader,code_length,device):
    with torch.no_grad():
        num = len(dataloader.dataset)
        code = torch.zeros([num, code_length])
        for i, (_, _,trains,_, index) in enumerate(dataloader):
            trains = trains.type(torch.FloatTensor).to(device)
            _,outputs, _ = model(trains)
            code[index, :] = outputs.sign().cpu()
    return code


