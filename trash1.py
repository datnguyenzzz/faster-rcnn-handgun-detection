import os
import skimage.io
import json
import numpy as np

datadir = "D:\My_Code\database\images\export"
ann_link = os.path.join(datadir, "_annotations.coco.json")
dataset = json.load(open(ann_link))

file_name = []
for image in dataset['images']:
    file_name.append(image['file_name'])

f1 = []
f2 = []
f3 = []

for name in file_name:
    image_link = os.path.join(datadir, name)
    #print(image_link)
    image = skimage.io.imread(image_link)

    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    if image.shape[2] == 4:
        image = np.delete(image, -1, 2)

    f1.append(np.mean(image[:,:,0]))
    f2.append(np.mean(image[:,:,1]))
    f3.append(np.mean(image[:,:,2]))

print(np.mean(f1),np.mean(f2),np.mean(f3))
