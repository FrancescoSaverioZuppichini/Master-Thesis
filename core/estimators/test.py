from estimators.utils import get_learner
from estimators.data.transformations import get_transform
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
model_dir = '/media/francesco/Carino/vaevictis/data/microresnet#4-gate=3x3-n=1-se=True--0.001-0.66-1557774998.5508704/'

# vis = DataFrameVisualization.from_root(test_root)
# vis(tr=0.45)

learner, dl = get_learner(model_name, model_dir, callbacks=[], root=test_root, tr=0.2,
                         patches_dir='/media/francesco/saetta/krock-dataset/test/patches/0.66/',
                         transform=get_transform(scale=1, should_aug=False, debug=False),
                         time_window=50 * 2,
                         patch_size=0.66)
# learner.model.eval()
print(len(dl))
learner.model.eval()
loss, acc, roc_auc = learner.validate(metrics=[accuracy, ROC_AUC()])

interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix()
plt.show()
print('[INFO] roc auc = {:.4f} acc = {}'.format(roc_auc, acc))
# print('acc = {}'.format(acc))