from torch.utils.data import Dataset, DataLoader
import os
import os.path
import numpy as np
import torch
import scipy.misc
from torchvision import transforms


#image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# create data_loader(test_loader, train_loader, retrieval_loader)
def load_train(config,img_data,txt_data,label_data,S_data, seed=None):
    np.random.seed(seed)
    random_index = np.random.permutation(range(img_data.shape[0]))

    img_query=img_data[random_index[0:config.query]]
    img_train=img_data[random_index[config.query:config.query+config.train]]
    img_retrieval=img_data[random_index[config.query:]]

    txt_query = txt_data[random_index[0:config.query]]
    txt_train = txt_data[random_index[config.query:config.query + config.train]]
    txt_retrieval = txt_data[random_index[config.query:]]

    S_data = S_data[random_index[0:]]
    S_data = np.transpose(S_data)

    S_query = S_data[random_index[0:config.query]]
    S_train = S_data[random_index[config.query:config.query + config.train]]
    S_retrieval = S_data[random_index[config.query:]]

    label_query = label_data[random_index[0:config.query]]
    label_train = label_data[random_index[config.query:config.query + config.train]]
    label_retrieval = label_data[random_index[config.query:]]


    train_dataset=data(config.root,img_train,txt_train,label_train,S_train)
    query_dataset=data(config.root,img_query,txt_query,label_query,S_query)
    retrieval_dataset=data(config.root,img_retrieval,txt_retrieval,label_retrieval,S_retrieval)

    train_loader=DataLoader(dataset=train_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.num_works)

    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_works)

    retrieval_loader = DataLoader(dataset=retrieval_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_works)



    label_train_dataset = label(label_train)

    label_train_loader=DataLoader(dataset=label_train_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.num_works)

    return train_loader,query_loader,retrieval_loader,label_train_loader

#CNNF_loader
def CNNF_loader(path):
    img=scipy.misc.imread(path, mode='RGB')
    img = scipy.misc.imresize(img, (224, 224))
    img = img.transpose((2, 1, 0))
    img = torch.from_numpy(img.astype('float32'))
    return img

# def VGG_loader(path):
#     img=Image.open(path).convert('RGB')
#     img=transform(img)
#     return img

# return data set item
# class data(Dataset):
#     def __init__(self,root,img_data,txt_data,label_data,S_data):
#         self.dataloader = CNNF_loader
#         self.root=root
#         self.imgs_data = []
#         self.txts_data=[]
#         self.S_data = []
#         self.labels=[]
#
#
#         for img in img_data:
#             self.imgs_data.append(os.path.join(root,img))
#         self.txts_data=txt_data
#         self.S_data = S_data
#         self.labels=label_data
#
#
#     def __getitem__(self, item):
#         img,txt,target, S =self.imgs_data[item],self.txts_data[item], self.labels[item] ,self.S_data[item]
#         img = self.dataloader(img)
#         txt = torch.from_numpy(txt).type(torch.FloatTensor)
#         return img,txt,target,S,item
#
#     def __len__(self):
#         return len(self.imgs_data)

class label(Dataset):
    def __init__(self,label_data):
        self.labels=label_data
    def __getitem__(self, item):
        target=self.labels[item]
        return target,item
    def __len__(self):
        return len(self.labels)

def load_train_nus(config,img_data,txt_data,label_data, seed=None):
    np.random.seed(seed)
    random_index = np.random.permutation(range(img_data.shape[0]))

    img_query=img_data[random_index[0:config.query]]
    img_train=img_data[random_index[config.query:config.query+config.train]]
    img_retrieval=img_data[random_index[config.query:]]

    txt_query = txt_data[random_index[0:config.query]]
    txt_train = txt_data[random_index[config.query:config.query + config.train]]
    txt_retrieval = txt_data[random_index[config.query:]]

    label_query = label_data[random_index[0:config.query]]
    label_train = label_data[random_index[config.query:config.query + config.train]]
    label_retrieval = label_data[random_index[config.query:]]


    train_dataset=data(config.root,img_train,txt_train,label_train)
    query_dataset=data(config.root,img_query,txt_query,label_query)
    retrieval_dataset=data(config.root,img_retrieval,txt_retrieval,label_retrieval)

    train_loader=DataLoader(dataset=train_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.num_works)

    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_works)

    retrieval_loader = DataLoader(dataset=retrieval_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_works)



    label_train_dataset = label(label_train)

    label_train_loader=DataLoader(dataset=label_train_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.num_works)

    return train_loader,query_loader,retrieval_loader,label_train_loader

class data(Dataset):
    def __init__(self,root,img_data,txt_data,label_data):
        self.dataloader = CNNF_loader
        self.root=root
        self.imgs_data = []
        self.txts_data=[]
        self.labels=[]


        for img in img_data:
            self.imgs_data.append(os.path.join(root,img))
        self.txts_data=txt_data
        self.labels=label_data


    def __getitem__(self, item):
        img,txt,target=self.imgs_data[item],self.txts_data[item], self.labels[item]
        img = self.dataloader(img)
        txt = torch.from_numpy(txt).type(torch.FloatTensor)
        return img,txt,target,item

    def __len__(self):
        return len(self.imgs_data)