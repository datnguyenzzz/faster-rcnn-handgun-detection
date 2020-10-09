from PIL import Image
import os
import json

ROOT_DIR = os.path.abspath("../database")
ANNOTATIONS = os.path.join(ROOT_DIR, "annotations")
IMAGES = os.path.join(ROOT_DIR,"images\\gun")

ann_link = ANNOTATIONS + "\\" + 'val' + ".json"

jsonFile = open(ann_link, "r") # Open the JSON file for reading
annotations = json.load(jsonFile) # Read the JSON into the buffer
jsonFile.close() # Close the JSON file

annotations = list(annotations.values())

annotations = [a for a in annotations if a['regions']]


for i,a in enumerate(annotations):
    dataset_dir = os.path.join(IMAGES, 'train')
    a['filename'] = str(i+1) + '.jpg'

print(annotations)

jsonFile = open("val.json", "w+")
jsonFile.write(json.dumps(annotations))
jsonFile.close()
