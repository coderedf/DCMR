import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import scipy.io as scio

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13,
            "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn,
            "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn}

#load model
def load_model(model_name,code_length,label_size,input_size,model_path):
    img_model=None

    if model_name=='CNNF':
        pre = scio.loadmat(model_path)
        img_model = ImgModule(code_length, label_size, pre)
    if model_name in vgg_dict:
        img_model=IMGNet(model_name,label_size,code_length)
    txt_model=TXTNet(input_size,code_length,label_size)
    label_model=LabelNet(label_size,code_length,label_size)
    return img_model,txt_model,label_model

#image Module
class ImgModule(nn.Module):
    def __init__(self, code_length,label_size,pretrain_model=None):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        self.features = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            nn.ReLU(inplace=True),
            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            nn.ReLU(inplace=True),
        )
        #conv common feature
        self.con=nn.Sequential(
            nn.Conv2d(4096,8192,1),
            nn.ReLU(True)
        )
        #private feature
        self.private = nn.Sequential(
            nn.Conv2d(4096, 8192, 1),
            nn.ReLU(True)
        )
        #private hashing
        self.hash_private = nn.Sequential(
            nn.Linear(in_features=8192, out_features=code_length),
            nn.Tanh()
        )
        # fc8
        self.classifier = nn.Linear(in_features=8192, out_features=label_size)

        self.hash=nn.Sequential(
            nn.Linear(in_features=8192,out_features=code_length),
            nn.Tanh()
        )
        self.layer_norm=nn.LayerNorm(8192)
        self.relu_=nn.ReLU()
        self.mean = torch.zeros(3, 224, 224)
        if pretrain_model:
            self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))


    def forward(self, x):
        if x.is_cuda:
            img = x - self.mean.cuda()
        else:
            img = x - self.mean
        x = self.features(img)
        # common feature
        original_f = self.con(x)
        class_f = original_f.view(original_f.size(0), 8192)
        # classifier
        feature_pre = self.classifier(class_f)
        img_predict = F.softmax(feature_pre, dim=1)

        # private feature
        private_feature = self.private(x)
        private_f = private_feature.view(private_feature.size(0), 8192)

        # common hash code
        hash_code = self.hash(class_f)
        # private hash code
        hash_code_p = self.hash_private(private_f)
        return hash_code, hash_code_p, img_predict, class_f, private_f, original_f


#text module
class TXTNet(nn.Module):
    def __init__(self, input_size,code_length,label_size):
        super(TXTNet, self).__init__()
        self.txt_module = nn.Sequential(
            nn.Linear(input_size,1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.ReLU(True)
        )
        self.classifier=nn.Linear(4096,label_size)

        #common feature
        self.common = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True)
        )
        #private feature
        self.private = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True)
        )
        # common hashing
        self.hash_common = nn.Sequential(
            nn.Linear(in_features=4096, out_features=code_length),
            nn.Tanh()
        )
        #private hashing
        self.hash_private = nn.Sequential(
            nn.Linear(in_features=4096, out_features=code_length),
            nn.Tanh()
        )
        self.hash = nn.Sequential(
            nn.Linear(in_features=4096, out_features=code_length),
            nn.Tanh()
        )
        self.layer_norm = nn.LayerNorm(4096)


    def forward(self, x):
        features=self.txt_module(x)
        features_pre = self.classifier(features)
        txt_predict = F.softmax(features_pre, dim=1)

        #feature:
        common_f = self.common(features)
        private_f = self.private(features)

        #hashing
        hash_code = self.hash(features)
        hash_code_p = self.hash_private(private_f)
        return hash_code, hash_code_p, txt_predict, common_f, private_f, features

class LabelNet(nn.Module):
    def __init__(self,input_size,code_length,label_size):
        super(LabelNet,self).__init__()
        self.label_module=nn.Sequential(
            nn.Linear(input_size,1024),
            nn.ReLU(True),
            nn.Linear(1024,4096),
            nn.ReLU(True)
        )
        self.classifier = nn.Linear(4096, label_size)

        self.hash = nn.Sequential(
            nn.Linear(in_features=4096, out_features=code_length),
            nn.Tanh()
        )
    def forward(self, x):

        features=self.label_module(x)

        hash_code=self.hash(features)
        label_pre = self.classifier(features)
        label_predict = F.softmax(label_pre, dim=1)

        return label_pre,hash_code,label_predict

#other VGG image Module
class IMGNet(nn.Module):
    def __init__(self, name,label_size,code_length,dim=4096):
        super(IMGNet,self).__init__()
        model_vgg = vgg_dict[name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_vgg.classifier[i])

        self.feature=nn.Sequential(
            nn.Linear(model_vgg.classifier[6].in_features,dim),
            nn.ReLU(True)
        )
        self.classifier = nn.Linear(in_features=4096, out_features=label_size)

        self.hash = nn.Sequential(
            nn.Linear(in_features=4096, out_features=code_length),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        features=self.feature(x)
        features_pre = self.classifier(features)
        img_predict = F.softmax(features_pre, dim=1)
        hash_code = self.hash(features)

        return hash_code,img_predict

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import scipy.io as scio

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13,
            "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn,
            "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn}

#load model
def load_model(model_name,code_length,label_size,input_size,model_path):
    img_model=None

    if model_name=='CNNF':
        pre = scio.loadmat(model_path)
        img_model = ImgModule(code_length, label_size, pre)
    if model_name in vgg_dict:
        img_model=IMGNet(model_name,label_size,code_length)
    txt_model=TXTNet(input_size,code_length,label_size)
    label_model=LabelNet(label_size,code_length,label_size)
    return img_model,txt_model,label_model

#image Module
class ImgModule(nn.Module):
    def __init__(self, code_length,label_size,pretrain_model=None):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        self.features = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            nn.ReLU(inplace=True),
            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            nn.ReLU(inplace=True),
        )
        #conv common feature
        self.con=nn.Sequential(
            nn.Conv2d(4096,8192,1),
            nn.ReLU(True)
        )
        #private feature
        self.private = nn.Sequential(
            nn.Conv2d(4096, 8192, 1),
            nn.ReLU(True)
        )
        #private hashing
        self.hash_private = nn.Sequential(
            nn.Linear(in_features=8192, out_features=code_length),
            nn.Tanh()
        )
        # fc8
        self.classifier = nn.Linear(in_features=8192, out_features=label_size)

        self.hash=nn.Sequential(
            nn.Linear(in_features=8192,out_features=code_length),
            nn.Tanh()
        )
        self.layer_norm=nn.LayerNorm(8192)
        self.relu_=nn.ReLU()
        self.mean = torch.zeros(3, 224, 224)
        if pretrain_model:
            self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))


    def forward(self, x):
        if x.is_cuda:
            img = x - self.mean.cuda()
        else:
            img = x - self.mean
        x = self.features(img)
        # common feature
        original_f = self.con(x)
        class_f = original_f.view(original_f.size(0), 8192)
        # classifier
        feature_pre = self.classifier(class_f)
        img_predict = F.softmax(feature_pre, dim=1)

        # private feature
        private_feature = self.private(x)
        private_f = private_feature.view(private_feature.size(0), 8192)

        # common hash code
        hash_code = self.hash(class_f)
        # private hash code
        hash_code_p = self.hash_private(private_f)

        return hash_code, hash_code_p, img_predict, class_f, private_f, original_f


#text module
class TXTNet(nn.Module):
    def __init__(self, input_size,code_length,label_size):
        super(TXTNet, self).__init__()
        self.txt_module = nn.Sequential(
            nn.Linear(input_size,1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.ReLU(True)
        )
        self.classifier=nn.Linear(4096,label_size)

        #common feature
        self.common = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True)
        )
        #private feature
        self.private = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True)
        )
        # common hashing
        self.hash_common = nn.Sequential(
            nn.Linear(in_features=4096, out_features=code_length),
            nn.Tanh()
        )
        #private hashing
        self.hash_private = nn.Sequential(
            nn.Linear(in_features=4096, out_features=code_length),
            nn.Tanh()
        )
        self.hash = nn.Sequential(
            nn.Linear(in_features=4096, out_features=code_length),
            nn.Tanh()
        )
        self.layer_norm = nn.LayerNorm(4096)


    def forward(self, x):
        features=self.txt_module(x)
        features_pre = self.classifier(features)
        txt_predict = F.softmax(features_pre, dim=1)

        #feature:
        common_f = self.common(features)
        private_f = self.private(features)

        #hashing
        hash_code = self.hash(features)
        hash_code_p = self.hash_private(private_f)
        return hash_code, hash_code_p, txt_predict, common_f, private_f, features

class LabelNet(nn.Module):
    def __init__(self,input_size,code_length,label_size):
        super(LabelNet,self).__init__()
        self.label_module=nn.Sequential(
            nn.Linear(input_size,1024),
            nn.ReLU(True),
            nn.Linear(1024,4096),
            nn.ReLU(True)
        )
        self.classifier = nn.Linear(4096, label_size)

        self.hash = nn.Sequential(
            nn.Linear(in_features=4096, out_features=code_length),
            nn.Tanh()
        )
    def forward(self, x):

        features=self.label_module(x)

        hash_code=self.hash(features)
        label_pre = self.classifier(features)
        label_predict = F.softmax(label_pre, dim=1)

        return label_pre,hash_code,label_predict

#other VGG image Module
class IMGNet(nn.Module):
    def __init__(self, name,label_size,code_length,dim=4096):
        super(IMGNet,self).__init__()
        model_vgg = vgg_dict[name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_vgg.classifier[i])

        self.feature=nn.Sequential(
            nn.Linear(model_vgg.classifier[6].in_features,dim),
            nn.ReLU(True)
        )
        self.classifier = nn.Linear(in_features=4096, out_features=label_size)

        self.hash = nn.Sequential(
            nn.Linear(in_features=4096, out_features=code_length),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        features=self.feature(x)
        features_pre = self.classifier(features)
        img_predict = F.softmax(features_pre, dim=1)
        hash_code = self.hash(features)

        return hash_code,img_predict

