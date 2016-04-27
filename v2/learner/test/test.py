
import tensorflow as tf
import dataset.mcam1
import dataset.mcam_image.local
import learner.mcam1


def test_train_test_call():
    ds = dataset.mcam1.McamDataSet(dataset.mcam_image.local.McamImage, b=0.5, w=0.5, window_size=28, stride=100, margin=10)
    inputs, labels = ds.next(compression=5)
    nn = learner.mcam1.McamLearner()
    train_stats = nn.train(inputs, labels)
    test_stats = nn.test(inputs, labels)
    outputs = nn(inputs)
    return train_stats > 0 and test_stats > 0 and outputs.shape == (len(inputs), 2)


def main():
    print test_train_test_call()


if __name__ == '__main__':
    main()