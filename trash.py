import os
import json
import skimage.io
import numpy as np
from skimage.transform import resize

datadir = "D:\My_Code\database\images\export"
ann_link = os.path.join(datadir, "_annotations.coco.json")
dataset = json.load(open(ann_link))

file_name = []
for image in dataset['images']:
    file_name.append(image['file_name'])

annotations = []

for i in range(len(file_name)):
    annotations.append([])

for image in dataset['annotations']:
    image_id = image['image_id']
    annotations[image_id].append(image['bbox'])

index = 1463

image_name = file_name[index]
image_link = os.path.join(datadir, image_name)
print(image_link)
image = skimage.io.imread(image_link)

new_size = 512

image = resize(image,(new_size,new_size))

print(image.shape)

bboxes = annotations[index]

for bbox in bboxes:
    x,y,w,h = bbox

    y1,x1,y2,x2 = y,x,y+h,x+w

    x1 = x1 * 1.0*new_size / 416
    x2 = x2 * 1.0*new_size / 416
    y1 = y1 * 1.0*new_size / 416
    y2 = y2 * 1.0*new_size / 416


    y1 = int(y1)
    x1 = int(x1)
    y2 = int(y2)
    x2 = int(x2)


    print(y1,x1,y2,x2)

    for y in range(y1,y2+1):
        image[y][x1][0] = 1.0
        image[y][x1][1] = 0
        image[y][x1][2] = 0

    for y in range(y1,y2+1):
        image[y][x2][0] = 1.0
        image[y][x2][1] = 0
        image[y][x2][2] = 0

    for x in range(x1,x2+1):
        image[y1][x][0] = 1.0
        image[y1][x][1] = 0
        image[y1][x][2] = 0
    for x in range(x1,x2+1):
        image[y2][x][0] = 1.0
        image[y2][x][1] = 0
        image[y2][x][2] = 0

from skimage.io import imsave
imsave('foo.png',image)
