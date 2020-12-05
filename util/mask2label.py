import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import nibabel as nib
import os
import cv2
import math


def water(img_path):
    src = cv2.imread(img_path)
    img = src.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)


    kernel2 = np.ones((7, 7), np.uint8)
    sure_bg = cv2.dilate(opening, kernel2, iterations=3)


    dist_transform = cv2.distanceTransform(sure_bg, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)


    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)


    ret, markers1 = cv2.connectedComponents(sure_fg)


    markers = markers1 + 1


    markers[unknown == 255] = 0

    markers3 = cv2.watershed(img, markers)

    img[markers3 == -1] = [0, 0, 0]

    img[markers3 == 1] = [0, 0, 0]

    img[markers3 == 2] = [255, 255, 255]
    img[markers3 == 3] = [255, 255, 255]
    img[markers3 == 4] = [255, 255, 255]
    return img

def segmentation(img_path):
    src= cv2.imread(img_path)
    img = src.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, gray.max(), 255,  cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
   
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
   
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
  
    return opening


source_path = r"a2_b path"

path = source_path + "/masks"

output_path = source_path + '/labels'


path_list = os.listdir(path)
path_list.sort()
len1 = 0
count = 0
for filename in path_list:
    count += 1
    cont_area = []
    len1 += 1
    image_path = os.path.join(path, filename)
    src = cv2.imread(image_path)
    result = water(image_path)

    index = filename.rfind('.')
    filename = filename[:index]
    filename = filename[:-5] + "_segmentation"

    cv2.imwrite(output_path +'/'+ filename+".png", result)
    print(round(count * 100 / len(path_list), 2), "%")
 