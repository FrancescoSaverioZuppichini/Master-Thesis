from utilities.postprocessing.utils import KrockPatchExtractStrategy

def hm_patch_generator(hm, step, alpha, max_advancement, res):
    patch_extract = KrockPatchExtractStrategy(max_advancement)

    h, w = hm.shape
    x = 0
    while x < w:
        y = 0
        while y < h:
            yield patch_extract(hm, x,y, alpha, res)[0]
            y += step
        x += step