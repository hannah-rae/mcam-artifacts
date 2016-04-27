
import collections
import time

class Trainer(object):

    dataset_class = NotImplemented

    def __init__(self, dataset, len_history=1):
        if isinstance(dataset, self.dataset_class):
            self.dataset = dataset
        else:
            raise TypeError('Trainer.__init__: wrong dataset class')

        self.len_history = len_history
        self.stats_history = collections.deque([], self.len_history)
        self.last_saved = time.time()

    def params_from_stats_history(self, stats_history):
        raise NotImplementedError

    def train_step(self, learner):
        params = self.params_from_stats_history(list(self.stats_history))
        print 'compression', params['compression']
        print 'global step', learner.get_global_step()
        inputs, labels = self.dataset.next(**params)
        stats = learner.train(inputs, labels)
        print 'stats', stats
        self.stats_history.append(stats)

    def train(self, learner):
        try:
            while True:
                self.train_step(learner)
                cur_time = time.time()
                if cur_time - self.last_saved > 600:
                    learner.saver.save(learner.sess, 'saved_sessions/saved_%d_%d' % (cur_time, learner.get_global_step()))
                    self.last_saved = cur_time
        except StopIteration:
            pass
        finally:
            cur_time = time.time()
            learner.saver.save(learner.sess, 'saved_sessions/saved_%d_%d' % (cur_time, learner.get_global_step()))






