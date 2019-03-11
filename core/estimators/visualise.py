import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.nn.functional import softmax
from fastai.train import Learner, DataBunch, DatasetType
from fastai.callback import Callback
from torch.utils.data import DataLoader
from datasets.TraversabilityDataset import TraversabilityDataset
from datasets.TraversabilityDataset import get_dataloaders, get_transform, TraversabilityDataset
from models.resnet import *
from models.omar_cnn import *
from models import zoo



model = zoo['microresnet#4-gate=3x3-n=2-se=True']

test_dl = DataLoader(TraversabilityDataset.from_root(root='/home/francesco/Desktop/data/750/test/df/',
                                                     more_than=-0.5,
                                                     transform=get_transform(None, scale=10),
                                                     tr=0.45,
                                                     n=10), shuffle=False, batch_size=128, num_workers=16)


class Visualise(Callback):
    """
    Store each input, target and prediction into a Dataframe in order to
    perform custom queries on the dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = None

    def on_batch_end(self, last_input, last_output, last_target, train, **kwargs):
        if not train:
            to_np = lambda x: x.cpu().numpy()
            last_target = to_np(last_target)
            last_output_act = softmax(last_output, dim=1)
            prediction = to_np(torch.argmax(last_output_act, dim=1))
            last_output_act = to_np(last_output_act)

            output_0, output_1 = last_output_act[:,0].tolist(), last_output_act[:,1].tolist()

            df = pd.DataFrame(data={'input': to_np(last_input).tolist(),
                                    'output_0': output_0,
                                    'output_1': output_1,
                                    'prediction': prediction.tolist(),
                                    'target': last_target.tolist() })

            if self.df is None:
                self.df = df
            else:
                self.df = pd.concat([self.df, df])


    def plot(self, sample):
        for img in sample['input']:
            img = np.array(img).squeeze()
            sns.heatmap(img,
                        vmin=0,
                        # annot=True,
                        # linewidths=.5,
                        fmt='0.2f')
            plt.show()


vis = Visualise()

learner = Learner(data=DataBunch(test_dl, test_dl), model=model,
                  callbacks=[vis],
                  model_dir='/home/francesco/Desktop/carino/vaevictis/data/microresnet#4-gate=3x3-n=2-se=True-750-0.001-92-1552230465.2925863/')

learner.load('roc_auc')

# loss, acc, roc = learner.validate(learner.data.test_dl, metrics=[accuracy, ROC_AUC()])
import pandas as pd

preds, targs = learner.get_preds(ds_type=DatasetType.Valid)

preds = softmax(preds, dim=1)
preds = torch.argmax(preds, dim=1)
print(preds)
# def show_preds(leaner):

best  = vis.df.sort_values(['output_1'], ascending=False).head(2)
worst  = vis.df.sort_values(['output_0'], ascending=False).head(2)

print(best['output_1'], worst['output_0'])

vis.plot(best)
#
vis.plot(worst)

