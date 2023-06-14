import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import newaxis
from skimage.transform import resize
import pickle

a = []
b = []

for (i,image_file) in enumerate(glob.iglob('input_images/*.jpg')):
        img = cv2.imread(image_file)
        a.append(resize(img, (80, 160, 3)))
        if(i%100==0):
            print(i)

f = open('input_final_1.p','wb')
pickle.dump(a, f, protocol=2)
f.flush()



for (i,image_file) in enumerate(glob.iglob('output/*.jpg')):
        img = cv2.imread(image_file)
        temp = resize(img, (80, 160, 3))
        temp = temp[:,:,1]
        temp = temp[:,:,newaxis]
        b.append(temp)
        if(i%100==0):
            print(i)


g = open('output_final_1.p','wb')
pickle.dump(b, g, protocol=2)
g.flush()
b=[]

