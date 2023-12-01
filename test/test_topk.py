from torch.utils.data import Dataset, DataLoader
import os
import os.path
import numpy as np
import torch
import scipy.misc
from torchvision import transforms
from PIL import Image
import sys
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
rootPath = os.path.split(BASE_DIR)[0]
sys.path.append(rootPath)
from data import data_deal as dd
import configparser
import eval.eval as eva
import sys


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load(root,img,txt):
    retrieval_dataset = data(root, img, txt)
    data_loader = DataLoader(dataset=retrieval_dataset,
                              batch_size=128,
                              shuffle=False,
                              num_workers=4)

    return data_loader


def CNNF_loader(path):
    img=scipy.misc.imread(path, mode='RGB')
    img = scipy.misc.imresize(img, (224, 224))
    img = img.transpose((2, 1, 0))
    img = torch.from_numpy(img.astype('float32'))
    return img

def VGG_loader(path):
    img=Image.open(path).convert('RGB')
    img=transform(img)
    return img

class data(Dataset):
    def __init__(self,root,img_data,txt_data):
        self.dataloader = CNNF_loader
        self.root=root
        self.imgs_data = []
        self.txts_data=[]
        #self.labels=[]

        for img in img_data:
            self.imgs_data.append(os.path.join(root,img))
        self.txts_data=txt_data
        #self.labels = label_data

    def __getitem__(self, item):

        img,txt=self.imgs_data[item],self.txts_data[item]
        img = self.dataloader(img)
        txt = torch.from_numpy(txt).type(torch.FloatTensor)
        return img,txt,item

    def __len__(self):
        return len(self.imgs_data)



def code_(img_model,txt_model,dataloader,code_length):
    with torch.no_grad():
        num=len(dataloader.dataset)
        img_code=torch.zeros([num,code_length])
        txt_code=torch.zeros([num,code_length])

        for i,(img,txt,index) in enumerate(dataloader):
            # print(i)
            img_trains=img.cuda()
            txt_trains = txt.cuda()
            img_outputs,_,_,_=img_model(img_trains)
            txt_outputs,_,_,_=txt_model(txt_trains)
            img_code[index,:]=img_outputs.sign().cpu()
            txt_code[index,:] =txt_outputs.sign().cpu()

    return img_code,txt_code



if __name__=='__main__':
    cf = configparser.ConfigParser()
    # IF MIRFlickr25K:
    cf.read("config1.ini")
    # IF NUS-WIDE10.5K:
    # cf.read("config.ini")

    root=cf.get("Code_produce","root")

    if cf.get("Code_produce", "dataname") == 'mirflickr':
        img, txt, labels, S = dd.mirflickr_data()
    elif cf.get("Code_produce", "dataname") == 'nus_wide':
        img, txt, labels, S = dd.nus_wide_data()
    loader=load(root,img,txt)

    txt_model=torch.load(cf.get("Code_produce","txt_model")).cuda()
    img_model=torch.load(cf.get("Code_produce","img_model")).cuda()

    bits=int(cf.get("Code_produce","bits"))
    img_code,txt_code=code_(img_model,txt_model,loader,bits)

    img_code = img_code.numpy()
    txt_code = txt_code.numpy()

    img_code[img_code==-1]=0
    txt_code[txt_code==-1]=0

    query_num=int(cf.get("Code_produce","query_num"))
    random_index = np.random.permutation(range(img_code.shape[0]))[0:query_num]
    query_img_code=img_code[random_index,:]
    query_txt_code=txt_code[random_index,:]
    query_label=labels[random_index,:]
    query_S = S[random_index, :]

    img_code=torch.from_numpy(img_code)#.cuda()
    txt_code = torch.from_numpy(txt_code)#.cuda()
    query_img_code = torch.from_numpy(query_img_code)#.cuda()
    query_txt_code = torch.from_numpy(query_txt_code)#.cuda()
    labels = torch.from_numpy(labels)#.cuda()
    query_label = torch.from_numpy(query_label)#.cuda()


    topk=[10,100,200,300,400,500,600,700,800,900,1000]

    for k in topk:
        i_t_pre=eva.calc_precisions_topn(query_img_code,txt_code,query_S,k)
        t_i_pre=eva.calc_precisions_topn(query_txt_code, img_code, query_S, k)
        print(k)
        print(i_t_pre)
        print(t_i_pre)




