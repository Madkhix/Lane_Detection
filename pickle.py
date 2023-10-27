import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import newaxis
from skimage.transform import resize
import pickle

# Two lists named 'a' and 'b' are created, the images of these lists will be used to store
a = []
b = []

# Process all files with *.jpg extension in 'input_images'
for (i,image_file) in enumerate(glob.iglob('input_images/*.jpg')):
        # Read the image
        img = cv2.imread(image_file)
         # Görüntüyü yeniden boyutlandırın (80x160 piksel) ve 'a' listesine ekleyin
        a.append(resize(img, (80, 160, 3)))
        # Her 100 görüntüde bir ilerlemenin kontrolü
        if(i%100==0):
            print(i)
                
# Save list 'a' in a file named 'input_final_1.p' (serialize with Pickle)
f = open('input_final_1.p','wb')
pickle.dump(a, f, protocol=2)
f.flush()

# Process all *.jpg files in 'output' folder
for (i,image_file) in enumerate(glob.iglob('output/*.jpg')):
        # Read the image
        img = cv2.imread(image_file)
        # Resize the image (80x160 pixels) and make it a single channel
        temp = resize(img, (80, 160, 3))
        temp = temp[:,:,1] # Select green channel
        temp = temp[:,:,newaxis] # Add a newaxis to the `temp` array along the third dimension.
        # 'b' listesine ekleyin
        b.append(temp)
        # Her 100 görüntüde bir ilerlemenin kontrolü
        if(i%100==0):
            print(i)

# 'output_final_1.p' adlı bir dosyaya 'b' listesini kaydedin (Pickle ile seri hale getirin)
g = open('output_final_1.p','wb')
pickle.dump(b, g, protocol=2)
g.flush()
# 'b' listesini temizleyin (bunu yapmamızın nedeni bellek kullanımını azaltmaktır)
b=[]

