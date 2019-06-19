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
import numpy as np
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
plt.style.use(['science'])
class Scatter(Callback):
    """
    Compute roc auc by storing all the predictions/targets and run it once
    when the epoch is over
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outs = np.array([])
        self.targs = np.array([])

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        last_output = last_output.cpu().numpy()
        targs = last_target.cpu().numpy()

        self.outs = np.append(self.outs, last_output)
        self.targs = np.append(self.targs, targs)

    def on_epoch_end(self, last_metrics, last_target, last_output, **kwargs):

        self.outs = self.outs.reshape((-1))
        self.targs = self.targs.reshape((-1))

        fig = plt.figure(figsize=(10,8))
        plt.scatter(self.targs, self.outs, s=1)
        ax = plt.gca()
        ax.set_xlabel('Predictions', fontsize=18)
        ax.set_ylabel('Targets', fontsize=18)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        plt.show()
        fig.savefig('/home/francesco/Desktop/regression_scatter.png', dpi=400, transparent=True)

        return {'last_metrics': last_metrics}

learner, dl = get_learner(model_name, model_dir, callbacks=[Scatter()], load_metric='loss',
                          criterion=criterion,
                          root=test_root,
                          tr=None,
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
loss = learner.validate(metrics=[])
print(loss)
# interp = ClassificationInterpretation.from_learner(learner)
# interp.plot_confusion_matrix()
# plt.show()
# print('[INFO] roc auc = {:.5f} acc = {}'.format(roc_auc, acc))

# print('acc = {}'.format(acc))


# fig.savefig(out_dir + '/pca-test-patches-{}-{}.png'.format(n_clusters, highlight_class), dpi=400)