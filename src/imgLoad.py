# Loading images

from conf import *
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint, shuffle

print("Loading images from: " + IMGSET_PATH)

def subArray(arr, x, y, sizePatch):
  subArr = np.empty([sizePatch, sizePatch])
  for i in range(sizePatch):
    for j in range(sizePatch):
      subArr[i,j] = arr[i+x,j+y]
   
  return subArr

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

if __name__ == "__main__":
  nbrPatch = 100 # Number of patch per image
  sizePatch = 20 # Size in pixel of a patch

  patches = []
  #Read image files
  for file in glob.glob(IMGSET_PATH+"*.tif"):
    print("File "+file)
    img = plt.imread(IMGSET_PATH+file)
    gray = rgb2gray(img)
   
    #Extract random patches
    for i in range(nbrPatch):
      x = randint(0,gray.shape[0]-sizePatch)
      y = randint(0,gray.shape[1]-sizePatch)
      patches.append(subArray(gray,x,y,sizePatch))
  
  #Shuffle patches:
  np.random.shuffle(patches)
  
  # Save the patches in a file
  # into patches.npy
  np.save(PATCHES_PATH, patches)



