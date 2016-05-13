import trainer.mcam1 as trainer
import learner.mcam1 as learner
import dataset.mcam1 as dataset
import dataset.mcam_image.sese_nfs as image
#import dataset.mcam_image.local as image


def main():
    ds = dataset.McamDataSet(image.McamImage, b=7, w=-700, window_size=100, stride=100*4)
    tr = trainer.McamTrainer(ds, len_history=10, xent_threshold=1, init_compression=55)
    nn = learner.McamLearner()
    tr.train(nn)


if __name__ == '__main__':
    main()





