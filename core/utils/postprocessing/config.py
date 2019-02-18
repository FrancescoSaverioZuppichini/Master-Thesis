class Config:
    WORKERS = 16
    # BASE_DIR = '/home/francesco/Desktop/carino/vaevictis/data/train_no_tail/val/'
    BASE_DIR = '/home/francesco/Desktop/carino/vaevictis/data/flat_spawns/val/'
    # BASE_DIR = '/home/francesco/Desktop/carino/vaevictis/data/test/'
    BAG_FOLDER = BASE_DIR + 'bags/'
    CSV_FOLDER = BASE_DIR + 'csvs/'
    MAPS_FOLDER = '/home/francesco/Documents/Master-Thesis/core/maps/val/'

    PATCH_SIZE = 100
    ADVANCEMENT_TH = 0.12
    TIME_WINDOW = 125
    HEIGHT_SCALE_FACTOR = 1
    SKIP_EVERY = 12
    DATASET_FOLDER = '/home/francesco/Desktop/data/val/' + 'dataset/{}-{}-{}-{}-no_tail-spawn'.format(100,
                                                                                         PATCH_SIZE,
                                                                                         ADVANCEMENT_TH,
                                                                                         SKIP_EVERY) # 100 = n sims
    CENTER_H_PATCH = True