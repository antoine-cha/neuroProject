# Loading images

from conf import *
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint

print("Loading images from: " + IMGSET_PATH)

nbrPatch = 100
sizePatch = 20

def subArray(arr, x, y):
  subArr = np.empty([sizePatch, sizePatch])
  for i in range(sizePatch):
    for j in range(sizePatch):
      subArr[i,j] = arr[i+x,j+y]
   
  return subArr

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


patches = []
owd = os.getcwd()
os.chdir(IMGSET_PATH)
for file in glob.glob("*.tif"):
  print("File "+file)
  img = plt.imread(IMGSET_PATH+file)
  gray = rgb2gray(img)
 
  for i in range(nbrPatch):
    x = randint(0,gray.shape[0]-sizePatch)
    y = randint(0,gray.shape[1]-sizePatch)
    patches.append(subArray(gray,x,y))
  
    

# Save the patches in a file
# into patches.npy
os.chdir(owd)
np.save(PATCHES_PATH, patches)



