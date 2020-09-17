from PIL import Image
import os

ROOT_DIR = os.path.abspath("../database")
IMAGES = os.path.join(ROOT_DIR,"images\\val")

for i in range(1,21):

    link = os.path.join(IMAGES, str(i) + ".jpg")
    print(link)
    img = Image.open(link)

    img = img.resize((768,768), Image.ANTIALIAS)
    img.save(link)
