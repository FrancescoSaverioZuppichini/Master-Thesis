# from utilities.postprocessing.utils import KrockPatchExtractStrategy
from utilities.postprocessing.utils import KrockPatchExtractStrategyNumpy
import numpy as np

def hm_patch_generator(hm, step, alpha, max_advancement, res):
    patch_extract = KrockPatchExtractStrategyNumpy(max_advancement)

    h, w = hm.shape
    print(h,w)
    x = 50
    while x + 50< w:
        y = 50
        while y + 50 < h:
            yield patch_extract(hm, x,y, alpha, res)[0]
            y += step
            print(y)
        x += step
        # print(x)


def hm_patch_list(hm, step, alpha, max_advancement, res):
    patch_extract = KrockPatchExtractStrategyNumpy(max_advancement)

    h, w = hm.shape
    images = []

    offset =  max_advancement // res + 14

    x = offset
    while x + offset< w:
        temp = []
        y = offset
        while y + offset < h:
            patch= patch_extract(hm, x,y, alpha, res)[0]
            y += step
            temp.append(patch)
        x += step
        images.append(temp)

    return np.array(images)