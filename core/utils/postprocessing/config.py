class Config:
    WORKERS = 16
    BASE_DIR = '/home/francesco/Desktop/data/train/'
    # BASE_DIR = '/home/francesco/Desktop/carino/vaevictis/data/train/'
    BAG_FOLDER = BASE_DIR + 'bags/'
    CSV_FOLDER = BASE_DIR + 'csvs/'
    DATASET_FOLDER = BASE_DIR + '/dataset/new-medium'
    MAPS_FOLDER = BASE_DIR + '/maps/'
    PATCH_SIZE = 80
    ADVANCEMENT_TH = 0.09
    TIME_WINDOW = 100
    HEIGHT_SCALE_FACTOR = 1
    SKIP_EVERY = 25
    CENTER_H_PATCH = True