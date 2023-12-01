import argparse
import torch
import model.train_test as tt
import data.data_loader1 as loader
from data import data_deal as dd


def parser_():
    parser = argparse.ArgumentParser(description='Text Classification')
    #MIRFlickr25K config setting:
    # parser.add_argument('--root', default='../mirflickr', type=str)#The address of images of MIRFlickr25K'
    # parser.add_argument('--dataname', default='mirflickr', type=str)
    # parser.add_argument('--topK', default=18015, type=int)
    # parser.add_argument('--label_size', default=24, type=int)
    # parser.add_argument('--input_size', default=1386, type=int)
    # parser.add_argument('--query', default=2000, type=int)
    # parser.add_argument('--train', default=10000, type=int)
    # parser.add_argument('--lam', default=0.4, type=float)
    #********************************************************
    #NUS-WIDE10.5K config setting
    parser.add_argument('--root', default='../nus_wide', type=str)#The address of sampled images of nus-wide10.5K from oringinal nus-wide'
    parser.add_argument('--dataname', default='nus_wide', type=str)
    parser.add_argument('--topK', default=191634, type=int)
    parser.add_argument('--label_size', default=21, type=int)
    parser.add_argument('--input_size', default=1000, type=int)
    parser.add_argument('--query', default=2100, type=int)
    parser.add_argument('--train', default=10500, type=int)
    parser.add_argument('--lam', default=0.7, type=float)
    #********************************************************
    #common config setting ((Pre-)trained model(s).)
    parser.add_argument('--model', default='../imagenet-vgg-f.mat')
    parser.add_argument('--mode_name', default='CNNF', type=str)
    parser.add_argument('--gpu', default=0, type=int,help='Use gpu(default: 0. -1: use cpu)')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--threshold', default=0, type=int)
    parser.add_argument('--code_length', default=64, type=int)
    parser.add_argument('--num_epochs', default=120, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--iter', default=20, type=int)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--llr', default=1e-3, type=float)
    parser.add_argument('--num_works', default=8, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.01, type=float)
    parser.add_argument('--eta', default=0.9, type=float)


    args = parser.parse_args()
    return args

if __name__=='__main__':

    config=parser_()

    if config.gpu == -1:
        config.device = torch.device("cpu")
    else:
        config.device = torch.device("cuda:%d" % config.gpu)

    if config.dataname=='mirflickr':
        img,tags,labels, S=dd.mirflickr_data()
        train_loader, test_loader, dataset_loader, label_train_loader = loader.load_train_nus(config, img, tags, labels,1)
    elif config.dataname == 'nus_wide':
        # img, tags, labels, S= dd.nus_wide_data()
        img, tags, labels = dd.nus_wide_data()
        train_loader, test_loader, dataset_loader, label_train_loader = loader.load_train_nus(config, img, tags, labels, 1)
    # train_loader, test_loader, dataset_loader, label_train_loader = loader.load_train(config, img, tags, labels, S, 1)

    tt.train(config,train_loader,
                    test_loader,
                    dataset_loader,
                    label_train_loader
                    )
