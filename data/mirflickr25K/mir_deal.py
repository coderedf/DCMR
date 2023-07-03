#bag-of-word of text model on mirflickr25k
import numpy as np
import pickle

# deal common tags
tag_idx, idx_tag = {}, {}
cnt = 0

TXT_P=""
with open("common_tags.txt", 'r',encoding="utf-8") as f:
    for line in f:
        line = line.split()
        tag_idx[line[0]] = cnt
        idx_tag[cnt] = line[0]
        cnt += 1
DIM_TXT = len(tag_idx.keys())
print(DIM_TXT)


# BoW
all_txt = np.zeros((20015, DIM_TXT),dtype=np.uint8)
count=0
with open("txt_database.txt",encoding="utf-8") as f:
    for line in f:
        lin=line.split()
        for s in lin:
            if s in tag_idx:
                all_txt[count][tag_idx[s]] = 1
        count+=1
    f.close()
with open("word_dict",'wb') as f:
    pickle.dump(tag_idx,f)
np.save("txt_database", all_txt)