import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import mnist


class NNTrainer(Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        super(NNTrainer, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def getData():
    # ミニバッチサイズ
    minibatch_size = 128

    # 訓練、テストデータ抽出
    trainData, testData = mnist.get_mnist()
    # ミニバッチに分ける
    train_iter = iterators.SerialIterator(trainData, minibatch_size)
    test_iter = iterators.SerialIterator(
        testData, minibatch_size, False, False)

    return train_iter, test_iter


def trainer(train_iter, test_iter):
    # 学習する回数を決める。
    max_epoch = 10

    # Classifierでmodelをラップすることで、modelに損失の計算プロセスを追加します。
    # 引数に損失関数を指定しない場合は、softmax_cross_entropyを使います。
    model = NNTrainer()
    model = L.Classifier(model)
    chainer.serializers.save_hdf5('my_model.npz', model, compression=4)

    # モデル（隠れ層が50層、出力が3層）
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # IteratorとOptimizerを使うupdaterを作る
    updater = training.updaters.StandardUpdater(train_iter, optimizer)

    # Trainerを用意する。updaterを渡すことで使える。epochを指定する。outはデータを保存する場所。
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='./result')

    # ログファイルを保存する機能を追加
    trainer.extend(extensions.LogReport())

    # 定期的に状態をシリアライズ（保存）する機能
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='model_epoch-{.updater.epoch}'))

    # # テストデータを使ってモデルの評価を行う機能
    trainer.extend(extensions.Evaluator(test_iter, model))

    # ターミナル上に出力
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))

    # 損失関数の値をグラフにする機能
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))

    # 正答率をグラフにする機能
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()

    # パラメータセーブ＆モデル化
    chainer.serializers.save_npz('NNTrainer.npz', model, compression=True)


def main():
    # データセット取得
    train, test = getData()
    # 学習
    trainer(train, test)


main()
