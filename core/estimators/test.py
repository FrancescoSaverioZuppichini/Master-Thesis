from estimators.utils import get_learner
from estimators.data.transformations import get_transform
from Config import Config
from fastai.vision import ClassificationInterpretation
from fastai.metrics import accuracy, dice, accuracy_thresh
from estimators.callbacks import ROC_AUC, Timer
from utilities.visualisation import DataFrameVisualization
from torchsummary import summary
from fastai.layers import CrossEntropyFlat, MSELossFlat
import matplotlib.pyplot as plt
from fastai.callback import Callback
from functools import partial

time_window = 100
test_root = '/media/francesco/saetta/krock-dataset/new-test-random/{}/csvs/'.format(time_window)
patches_dir = '/media/francesco/saetta/krock-dataset/new-test-random/{}/patches/'.format(time_window)
#


# model_name = 'microresnet#3-gate=7x7-n=1-se=True'
# model_dir = '/media/francesco/Carino/vaevictis/data/1558825182.753924'
criterion = MSELossFlat()

# test_root = '/media/francesco/saetta/krock-dataset/new-new-val/{}/csvs/'.format(time_window)
# patches_dir = '/media/francesco/saetta/krock-dataset/new-new-val/{}/patches/'.format(time_window)

model_name = 'microresnet#3-gate=7x7-n=1-se=True-regression'
model_dir = '/media/francesco/Carino/vaevictis/data/1560857566.16732'

# vis = DataFrameVisualization.from_root(test_root)
# vis(tr=0.45)

learner, dl = get_learner(model_name, model_dir, callbacks=[], load_metric='loss',
                          criterion=criterion,
                          root=test_root,
                          tr=0.2,
                          patches_dir=patches_dir,
                          transform=get_transform(),
                          time_window=50 * 2,
                          # more_than=0,
                          patch_size=0.7)

# learner.model.eval()
print(len(dl))
learner.model.eval()
summary(learner.model.cuda(), (1, 78, 78))

# loss, acc, roc_auc = learner.validate(metrics=[accuracy, ROC_AUC()])
# loss = learner.validate(metrics=[accuracy, ROC_AUC()])
loss= learner.validate(metrics=[partial(accuracy_thresh, thresh=0.2)])
print(loss)
# interp = ClassificationInterpretation.from_learner(learner)
# interp.plot_confusion_matrix()
# plt.show()
# print('[INFO] roc auc = {:.5f} acc = {}'.format(roc_auc, acc))

# print('acc = {}'.format(acc))
