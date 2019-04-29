from estimators.utils import get_learner, get_transform
from Config import Config
from fastai.vision import ClassificationInterpretation
from fastai.metrics import accuracy, dice
from estimators.callbacks import ROC_AUC, Timer
from utilities.visualisation import DataFrameVisualization

import matplotlib.pyplot as plt
# test_root = '/media/francesco/saetta/krock-dataset/querry-up-hill/csvs_patches/'
test_root = '/media/francesco/saetta/krock-dataset/test/csvs_patches/'

# test_root = '/media/francesco/saetta/no-shift-88-750/train/df/slope_rocks2'

model_name = Config.BEST_MODEL_NAME
model_dir = Config.BEST_MODEL_DIR

# vis = DataFrameVisualization.from_root(test_root)
# vis(tr=0.45)

learner, dl = get_learner(model_name, model_dir, callbacks=[], root=test_root, tr=0.2,
                         patches_dir='/media/francesco/saetta/krock-dataset/test/patches/0.66/',
                         transform=get_transform(scale=1, should_aug=False),
                         time_window=50 * 2,
                         patch_size=0.66)
# learner.model.eval()
print(len(dl))
loss, acc, roc_auc = learner.validate(metrics=[accuracy, ROC_AUC()])

interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix()
plt.show()
print('[INFO] roc auc = {:.4f} acc = {}'.format(roc_auc, acc))
# print('acc = {}'.format(acc))