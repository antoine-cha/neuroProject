# Loading images

from conf import *
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("Loading images from: " + IMGSET_PATH)


def subArray(arr, x, y):
  size = 20
  subArr = np.empty([size, size])
  for i in range(size):
    for j in range(size):
      subArr[i,j] = arr[i+x,j+y]
   
  return subArr


def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
os.chdir(IMGSET_PATH)
for file in glob.glob("*.tif"):
  print("File "+file)
  img = plt.imread(IMGSET_PATH+file)
  gray = rgb2gray(img)
  print(subArray(gray,1,1))
  
    





