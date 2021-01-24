
import sys
import argparse
import os


from PIL import Image


source_paths = [
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/0",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/1",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/2",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/3",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/4",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/5",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/6",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/7",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/8",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/9",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/a",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/b",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/c",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/d",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/e",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/f",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/g",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/h",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/i",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/j",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/k",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/l",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/m",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/n",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/o",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/p",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/q",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/r",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/s",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/t",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/u",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/v",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/w",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/x",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/y",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/z",


    ]
target_paths = [
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/0",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/1",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/2",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/3",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/4",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/5",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/6",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/7",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/8",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/9",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/a",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/b",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/c",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/d",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/e",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/f",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/g",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/h",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/i",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/j",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/k",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/l",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/m",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/n",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/o",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/p",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/q",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/r",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/s",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/t",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/u",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/v",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/w",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/x",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/y",
    "/workspace/home/aden/work/project/o-viton/temp/org/K-Fashion AI_data_202011/train/z",
    ]
IMG_W = 256
IMG_H = 512

for idx, (source_path, target_path) in enumerate(zip(source_paths, target_paths)):

    filenames = os.listdir(source_path)
    imagefiles = []
    cnt = 0
    for filename in filenames:
        
        fullfilepath = os.path.join(source_path, filename)

        if os.path.isdir(fullfilepath) == False:
            base=os.path.basename(fullfilepath)
            name = os.path.splitext(base)[0]
            ext = os.path.splitext(base)[1]
            if ext.lower() == ".png" or ext.lower() == ".jpg" :
                image = Image.open(fullfilepath)		
                resize_image = image.resize((IMG_W, IMG_H),Image.LANCZOS)
                try:
                    os.makedirs(target_path)
                except:
                    pass
                resize_image.save("{}/{}.png".format(target_path,name), "PNG")
                print("{} -> {}".format(fullfilepath, os.path.join(target_path,name+".png") ))
                cnt = cnt+1
