import os
import numpy as np
from scipy import io as sio
import sys

##MIRFlick25K deal
def mirflickr_data():
    root= os.path.abspath(os.path.dirname(__file__))
    img_url=os.path.join(root,"mirflickr25K/img_database.txt")
    tags_url=os.path.join(root,"mirflickr25K/txt_database.npy")
    label_url=os.path.join(root,"mirflickr25K/mir_label.txt")
    S_url = os.path.join(root, "mirflickr25K/S_mir.npy")

    img_data=[]
    labels=[]
    #loading image
    with open(img_url, 'r') as img_file_read:
        while True:
            line = img_file_read.readline()
            if not line:
                break
            pos = line.split()[0]
            img_data.append(pos)
    img_data = np.array(img_data)
    #loading tags(text)
    tags_data=np.load(tags_url)
    #loading S(ground truth relevance of instances)
    S_data = np.load(S_url)
    #loading original multi-label
    with open(label_url, 'r') as label_file_read:
        while True:
            line = label_file_read.readline()
            if not line:
                break
            label = line.split()
            labels.append(label)
    labels = np.array(labels, dtype=np.float)

    return img_data,tags_data,labels, S_data

#loading nus_wide10.5K
# def nus_wide_data():
#     root = os.path.abspath(os.path.dirname(__file__))
#     img_url = os.path.join(root, "nus_wide_105k/img_10500.txt")
#     tags_url = os.path.join(root, "nus_wide_105k/txt_10500_database.npy")
#     label_url = os.path.join(root, "nus_wide_105k/label_10500.txt")
#     S_url = os.path.join(root, "nus_wide_105k/S_nus_10.5k.npy")
#
#     img_data = []
#     labels = []
#     # loading image
#     with open(img_url, 'r') as img_file_read:
#         while True:
#             line = img_file_read.readline()
#             if not line:
#                 break
#             pos = line.split()[0]
#             img_data.append(pos)
#
#     img_data = np.array(img_data)
#     # loading tags(text)
#     tags_data = np.load(tags_url)
#     # loading S(ground truth relevance of instances)
#     S_data = np.load(S_url)
#     # loading original multi-label
#     with open(label_url, 'r') as label_file_read:
#         while True:
#             line = label_file_read.readline()
#             if not line:
#                 break
#             label = line.split()
#             labels.append(label)
#     labels = np.array(labels, dtype=np.float)
#
#     return img_data, tags_data, labels ,S_data

#loading nus_wide
def nus_wide_data():

    root = os.path.abspath(os.path.dirname(__file__))
    img_url = os.path.join(root, "nus_wide/img_database.txt")
    tags_url = os.path.join(root, "nus_wide/txt_database.npy")
    label_url = os.path.join(root, "nus_wide/nus_wide_label.txt")
    img_data = []
    labels = []

    with open(img_url, 'r') as img_file_read:
        while True:
            line = img_file_read.readline()
            if not line:
                break
            pos = line.split()[0]
            img_data.append(pos)
    img_data = np.array(img_data)
    tags_data = np.load(tags_url)

    with open(label_url, 'r') as label_file_read:
        while True:
            line = label_file_read.readline()
            if not line:
                break
            label = line.split()
            labels.append(label)
    labels = np.array(labels, dtype=np.float)

    return img_data, tags_data, labels