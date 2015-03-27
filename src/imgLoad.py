# Loading images

from conf import *
import glob
import os


print("Loading images from: " + IMGSET_PATH)

os.chdir(IMGSET_PATH)
for file in glob.glob("*.tif"):
  with open(IMGSET_PATH+file, 'r') as content_file:
    print("File "+file)





