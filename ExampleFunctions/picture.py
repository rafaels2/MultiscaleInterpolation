from PIL import Image
import numpy as np

FILENAME = "HeadScan.jpg"
_IMAGE = Image.open(FILENAME)
IMAGE = np.array(_IMAGE)


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
