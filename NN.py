# numpy
import numpy as np

# chainer
from chainer import gradient_check, Variable, Chain
from chainer import datasets, iterators, optimizers, serializers
import chainer.functions as F
import chainer.links as L

# sklearnのimport
from sklearn import datasets, model_selection

# 可視化ライブラリのimport
import matplotlib.pyplot as plt


# 三層ニューラルネットワーク
class NN(Chain):

    def __init__(self, n_hid=100, n_out=3):

        super().__init__()

        with self.init_scope():
            self.l1 = L.Linear(None, n_hid)
            self.l2 = L.Linear(n_hid, n_out)

    def __call__(self, x):
        hid = F.relu(self.l1(x))
        return self.l2(hid)


def getData():

    # アヤメの品種データを読み込む
    iris = datasets.load_iris()
    return model_selection.train_test_split(iris.data.astype(np.float32), iris.target)


def train(train_data, train_label):

    # モデル（隠れ層が50層、出力が3層）
    model = NN()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # dataをVariableに変換します。
    train_data_variable = Variable(train_data.astype(np.float32))
    train_label_variable = Variable(train_label.astype(np.int32))

    loss_log = []
    for itr in range(10000):
        # パラメータの勾配を初期化
        model.cleargrads()

        # 学習
        prod_label = model(train_data_variable)
        loss = F.softmax_cross_entropy(prod_label, train_label_variable)
        loss.backward()

        # パラメータを更新
        optimizer.update()
        loss_log.append(loss.data)

    # lossのグラフ表示
    plt.plot(loss_log)
    plt.show()

    return model


def test(model, test_data, test_label):
    # テストデータをVaribleに変換
    test_data_variable = Variable(test_data.astype(np.float32))

    # テスト
    y = model(test_data_variable)
    y = F.softmax(y)

    # 最大値の要素を持つインデックスを返す(１は読み取る配列の添字を指定)
    # pred_labelは正解と思われる添字を集めた配列
    pred_label = np.argmax(y.data, 1)

    # accuracy（正解数/テストデータ数）
    acc = np.sum(pred_label == test_label) / (len(test_label))
    print('Accuracy : {0}%'.format(round(acc * 100, 2)))


def main():
    # データセット取得
    train_data, test_data, train_label, test_label = getData()
    # 訓練
    model = train(train_data, train_label)
    # テスト
    test(model, test_data, test_label)


main()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# porpose   : アヤメの品種３つのうちどれかを当てる
# optimizer : Adam

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
