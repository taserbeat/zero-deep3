import dezero.functions as F
import dezero.layers as L
from dezero import Layer
from dezero import utils

from typing import List, Tuple, Union


# =============================================================================
# Model / Sequential / MLP
# =============================================================================
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    """
    全結合層を複数まとめて扱うクラス
    """

    def __init__(self, fc_output_sizes: Union[List, Tuple], activation=F.sigmoid):
        """複数の全結合層を構築する

        Args:
            fc_output_sizes): 各全結合層の出力サイズをリスト、またはタプルで指定。
                例えば、(10, 1)の場合は2つのレイヤーを作成し、1つ目の出力サイズが10、2つ目の出力サイズが1となる。

                (10, 10, 1)の場合はさらにレイヤーが1つ増えることになる。

            activation: すべての層で使用する活性化関数の種類。 デフォルトはシグモイド関数となる。
        """

        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
