
import collections

import tensorflow as tf

import trainer.mcam1
import dataset.mcam1
import dataset.mcam_image.local
import learner.mcam1

def test_params_from_stats_history():
    len_history = 10
    xent_threshold = 1
    ds = dataset.mcam1.McamDataSet(dataset.mcam_image.local.McamImage, b=0.5, w=0.5, window_size=28, stride=100, margin=10)
    tr = trainer.mcam1.McamTrainer(ds, len_history=len_history, xent_threshold=xent_threshold, init_compression=1)

    tr.params_from_stats_history([xent_threshold + 1] * (len_history - 1))
    p1 = tr.max_compression

    tr.params_from_stats_history([xent_threshold - 1] * (len_history - 1))
    p2 = tr.max_compression

    tr.params_from_stats_history([xent_threshold + 1] * (len_history + 1))
    p3 = tr.max_compression

    tr.params_from_stats_history([xent_threshold - 1] * (len_history + 1))
    p4 = tr.max_compression

    return (p1, p2, p3, p4) == (1, 1, 1, 2)




def main():
    print test_params_from_stats_history()


if __name__ == '__main__':
    main()