from models import *
from utils import get_learner, get_probs_and_labels_from_preds
from callbacks import StoreBestWorstAndSample, ROC_AUC
from os import path
from datasets.TraversabilityDataset import TraversabilityDataset, get_transform
import torch
import pandas as pd
import tqdm
model_dir = '/home/francesco/Desktop/carino/vaevictis/data/microresnet#4-gate=3x3-n=2-se=False-no-shift-88-750-0.001-None-1554052968.796346'
print(model_dir)

cb = StoreBestWorstAndSample(308)

root = '/media/francesco/saetta/correct-88-750/train/'
df = '/media/francesco/saetta/correct-88-750/test/df/querry-big-10/1550307709.2522066-complete.csv-patch.csv'




# learner, dataset = get_learner('microresnet#4-gate=3x3-n=2-se=False', model_dir, callbacks=[], dataset=dataset)

concat = TraversabilityDataset.from_root(root, tr=0.45, transform=get_transform(scale=10))

# def model_visualize(*args, **kwargs):
#     dataset = TraversabilityDataset(*args, **kwargs)
#     learner, dataset = get_learner('microresnet#4-gate=3x3-n=2-se=False', model_dir, callbacks=[], dataset=dataset)
#
#     preds = learner.get_preds(learner.data.test_dl)
#
#     probs, labels = get_probs_and_labels_from_preds(preds)
#
#     dataset.df['prediction'] = labels
#
#     print(dataset.false_pos['advancement'].mean())
#     print(dataset.false_neg['advancement'].mean())
#
#     return dataset.df

false_pos_df = None
false_neg_df = None

bar = tqdm.tqdm(concat.datasets)

total_df = None

for dataset in bar:
    bar.set_description('[INFO] Reading {}'.format(dataset.df_path))
    learner, _ = get_learner('microresnet#4-gate=3x3-n=2-se=False', model_dir, callbacks=[], dataset=dataset)

    preds = learner.get_preds(learner.data.test_dl)

    probs, labels = get_probs_and_labels_from_preds(preds)

    dataset.df['prediction'] = labels.cpu().tolist()

    false_pos = dataset.false_pos['advancement']
    false_neg = dataset.false_neg['advancement']

    false_pos_df = pd.concat([false_pos_df,  dataset.false_pos])
    false_neg_df = pd.concat([false_neg_df, dataset.false_neg])

    total_df = pd.concat([total_df, dataset.df])


print('***********************')
print(false_pos_df['advancement'].mean())
print(false_neg_df['advancement'].mean())

    # return dataset.df
# model_visualize(df, root=root, tr=0.45, transform=get_transform(scale=10))
# loss, roc = learner.validate(learner.data.test_dl, metrics=[ROC_AUC()])
#
# # root = '/media/francesco/saetta/test/'
# # cb.plot(cb.best.head(10))
# cb.plot(cb.false_pos.head(2))
# cb.plot(cb.false_neg.head(2))

from torch.utils.data import DataLoader







