class Config:
    WORKERS = 16
    BASE_DIR = '/home/francesco/Desktop/carino/vaevictis/data/train/'
    # BASE_DIR = '/home/francesco/Desktop/carino/vaevictis/data/test/'
    BAG_FOLDER = BASE_DIR + 'bags/'
    CSV_FOLDER = BASE_DIR + 'csvs/'
    MAPS_FOLDER = BASE_DIR + '/maps/'
    PATCH_SIZE = 100
    ADVANCEMENT_TH = 0.09
    TIME_WINDOW = 100
    HEIGHT_SCALE_FACTOR = 1
    SKIP_EVERY = 12
    DATASET_FOLDER = '/home/francesco/Desktop/data/test/' + 'dataset/{}-{}-{}-{}-no-trick'.format(100,
                                                                                         PATCH_SIZE,
                                                                                         ADVANCEMENT_TH,
                                                                                         SKIP_EVERY) # 100 = n sims
    CENTER_H_PATCH = True