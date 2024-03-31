import sys
import os
from tqdm import tqdm

'''
file1 = open("val_label.txt","a")#append mode
for dirname, _, filenames in os.walk('./gtFine/val'):
    for filename in filenames:
        #print(os.path.join(dirname,filename))
        print(filename)
        file1.write(filename + "\n")
file1.close()
'''
file0 = open("train_aug_unlabeled_1-2.txt","r")
file1 = open("train_image_unlabeled_1-2.txt","a")#append mode
file2 = open("train_label_unlabeled_1-2.txt","a")#append mode
for line in file0:
    #print(os.path.join(dirname,filename))
    filename_img_path = line.split()[0]
    #print(filename_img_path)
    filename_img = filename_img_path.split('/')[-1]
    prefix = filename_img.split('.')[0]
    print(prefix)
    cityname = prefix.split('_')[0]
    print(cityname)
    filename_img = cityname + '/' + prefix + '_leftImg8bit.png'
    filename_lbl = cityname + '/' + prefix + '_gtFine_labelIds.png'
    file1.write(filename_img + "\n")
    file2.write(filename_lbl + "\n")
file1.close()
file2.close()