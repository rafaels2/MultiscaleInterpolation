from PIL import Image, ImageOps
import numpy as np

FILENAME = "pexels-h-r-5802771.jpg"
_IMAGE = ImageOps.grayscale(Image.open(FILENAME).rotate(90))
IMAGE = np.array(_IMAGE) / 255


def original_function(x, y):
    if x > 1 or x < -1:
        import ipdb

        ipdb.set_trace()
    try:
        x = int(((x + 0.95) / 2) * IMAGE.shape[0])
        y = int(((y + 0.95) / 2) * IMAGE.shape[1])

        return IMAGE[x, y]
    except:
        import ipdb

        ipdb.set_trace()
