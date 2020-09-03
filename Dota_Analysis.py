#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:47:30 2020

@author: gramener
"""


import os
from os import listdir
from PIL import Image
import shutil
import cv2

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier
def draw_bounding_box(img, x, y, x_plus_w, y_plus_h):
    # print(x,y,x_plus_w,y_plus_h)
    label = "plane"
    color = (255,0,0)

    cv2.rectangle(img,(x, y),(x_plus_w, y_plus_h),color,2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
sav_label ="/media/gramener/HDD/plane_counting/DOTA Dataset/train-20200521T065807Z-001/train/Image_labels_Yolo/labels"
sav_img = "/media/gramener/HDD/plane_counting/DOTA Dataset/train-20200521T065807Z-001/train/Image_labels_Yolo/images"
img_direc = "/media/gramener/HDD/plane_counting/DOTA Dataset/train-20200521T065807Z-001/train/images/part1/images"
direc = "/media/gramener/HDD/plane_counting/DOTA Dataset/train-20200521T065807Z-001/train/labelTxt-v1.5/DOTA-v1.5_train"
list_dir = listdir(direc)
j=0
for i in list_dir:
    f = open(direc+'/'+i,'r')
    image_file_name = i.split("txt")[0]+"png"
    # img = Image.open(img_direc+'/'+image_file_name)
    img=cv2.imread(img_direc+'/'+image_file_name)
    height,width,channel = img.shape
    print(i,image_file_name)
    annotation_list = []
    for line in f.readlines():
        line_list = line.split(" ")
        if(len(line_list)>1):
            if(line_list[len(line_list)-2]=="plane"):
                # print(i,image_file_name)
                line_list = line_list[0:len(line_list)-2]
                line_list = [int(float(k)) for k in line_list]
                x = ((line_list[0]+line_list[4])/2)/width
                y = ((line_list[1]+line_list[5])/2)/height
                h = abs((line_list[5]-line_list[1])/height)
                w = abs((line_list[4]-line_list[0])/width)
                # x = line_list[2]
                # y = line_list[3]
                # h = line_list[6]
                # w = line_list[7]
                _str_output = str(0)+' '+str(truncate(x,6))+' '+str(truncate(y,6))+' '+str(truncate(w,6))+' '+str(truncate(h,6))+'\n'
                annotation_list.append(_str_output)
                # draw_bounding_box(img, int(abs(x-(w/2))*width), int(abs(y-(h/2))*height), int((x+(w/2))*width),int((y+(h/2))*height))
                # draw_bounding_box(img, x, y, h, w)
                # cv2.imwrite(sav_img+'/'+"object-detection1.jpg", img)
    if(len(annotation_list)>0):
        shutil.copy(img_direc+'/'+image_file_name,sav_img)
        w_file = open(sav_label+'/'+i,"w")
        w_file.writelines(annotation_list) 
        