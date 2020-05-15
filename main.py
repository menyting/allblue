import cv2
import os
import glob
from convertIm import convertImg
from PIL import Image

#read Image and write Image
img_dir = "test"
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)

i = 0
for f1 in files:
    image = convertImg(cv2.imread(f1))
    image.save("result\\result" + str(i) + ".jpg", "JPEG", )
    i += 1
