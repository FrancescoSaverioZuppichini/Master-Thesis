from comet_ml import Experiment

from fastai.train import LearnerCallback
import tqdm

class CometCallback(LearnerCallback):
    def __init__(self, api_key, params,  learner,*args, **kwargs):
        super().__init__(learner, *args, **kwargs)
        self.exp = Experiment(api_key)
        self.pr = params

    def on_epoch_end(self, n_epochs, last_loss, last_metrics, *args, **kwargs):
        if last_metrics is not None:
            print(last_metrics)
            self.exp.log_current_epoch(n_epochs)
            self.exp.log_metric("accuracy", last_metrics[0].item())

        return False