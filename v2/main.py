import trainer.mcam1 as trainer
import learner.mcam1 as learner
import dataset.mcam1 as dataset
#import dataset.mcam_image.sese_nfs as image
import dataset.mcam_image.local as image


def main():
    ds = dataset.McamDataSet(image.McamImage, b=42.89159257, w=-4.12832718, window_size=160, stride=200)
    tr = trainer.McamTrainer(ds, len_history=10, xent_threshold=1, init_compression=70)
    nn = learner.McamLearner(window_size=160)
    tr.train(nn)


if __name__ == '__main__':
    main()





