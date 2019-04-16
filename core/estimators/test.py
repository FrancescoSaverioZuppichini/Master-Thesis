from estimators.utils import get_learner, get_transform
from Config import Config
from fastai.metrics import accuracy, dice
from estimators.callbacks import ROC_AUC, Timer

# test_root = '/media/francesco/saetta/no-shift-88-750/test/'
test_root = '/media/francesco/saetta/no-shift-88-750/train/df/slope_rocks2'

model_name = Config.BEST_MODEL_NAME
model_dir = Config.BEST_MODEL_DIR

learner, _ = get_learner(model_name, model_dir, callbacks=[], root=test_root, tr=0.45, transform=get_transform(scale=10))

loss, acc, roc_auc = learner.validate(metrics=[accuracy, ROC_AUC()])

print('[INFO] roc auc = {}'.format(roc_auc))