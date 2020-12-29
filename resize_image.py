
import sys
import argparse
import os


from PIL import Image


source_paths = ["/home/dataset_/test_img","/home/dataset_/test_label"]
target_paths = ["/home/dataset/test_img","/home/dataset/test_label"]
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
