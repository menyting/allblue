from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def isBlack(pix):
    return (pix[0] == 0 and pix[1] == 0 and pix[2] == 0)

def oneColor(img, gray, mask):

    img = Image.fromarray(img).convert('RGB')
    gray = Image.fromarray(gray).convert('RGB')
    mask = Image.fromarray(mask).convert('RGB')
    #
    # plt.imshow(mask)
    # plt.show()

    pix_img = img.load()
    pix_gray = gray.load()
    pix_mask = mask.load()

    w = img.size[0]
    h = img.size[1]

    image = Image.new('RGB',(w,h))
    pix_image = image.load()

    black = [0,0,0]
    for i in range(w):
        for j in range(h):
            if isBlack(pix_mask[i,j]):
                pix_image[i,j] = pix_gray[i,j]
            else:
                pix_image[i,j] = pix_img[i,j]
    # plt.imshow(image)
    # plt.show()
    return image

def convertImg(img):
    low_blue = np.array([94,80,2])
    high_blue = np.array([126,255,255])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, low_blue, high_blue)
    img_mask = cv2.bitwise_and(hsv,hsv, mask=blue_mask)

    image = oneColor(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
        cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(img_mask,cv2.COLOR_HSV2RGB))

    return image

# plt.imshow(image)
# plt.show()
