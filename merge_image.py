import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
# Read In And Grayscale The Image

def process(img):
    image = cv2.imread(img,1)
    img_name = os.path.basename(img)
    img_name_final = str(a) + '.jpg'
    path = 'input_images/' + img_name_final
    cv2.imwrite(path,image)

a=0
b=20
c = True
folders = glob.glob('clips/0313-2/{0}'.format(b))
while(c):
    b=b+5
    for folder in folders:
        for f in glob.glob(folder+'/*.jpg'):
            a=a+1
            folders = glob.glob('clips/0313-2/{0}'.format(b))
            print(folders)
            process(f)
            if(b==53160):
                c = False
            

            





