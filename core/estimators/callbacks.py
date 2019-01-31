from comet_ml import Experiment

from fastai.train import LearnerCallback

class LRscheduler(LearnerCallback):
    def on_epoch_end(self, ):
        print('ddsa')