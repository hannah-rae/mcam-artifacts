
import random

import numpy as np

import trainer.base
import dataset.mcam1

class McamTrainer(trainer.base.Trainer):

    dataset_class = dataset.mcam1.McamDataSet

    def __init__(self, dataset, len_history=10, xent_threshold=1, init_compression=1):
        self.xent_threshold = xent_threshold
        self.init_compression = init_compression
        self.max_compression = 95
        super(self.__class__, self).__init__(dataset, len_history=len_history)

    def params_from_stats_history(self, stats_history):
        if False:
        #if len(stats_history) < self.len_history:
            compression = self.init_compression
        else:
            mean_xent = np.mean(stats_history)
            if mean_xent < self.xent_threshold and self.max_compression < 95:
                self.max_compression += 1
            compression = random.randint(self.init_compression, self.max_compression)
        return {'compression': compression}