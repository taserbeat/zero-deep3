import numpy as np


class Variable:
    """
    変数を表すクラス
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        return


class Function:
    """
    関数を表す基底クラス
    """

    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)  # 具体的な計算はforward()をオーバーライドして行う
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    """
    入力された値を2乗するクラス
    """

    def forward(self, x):
        return x ** 2


x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
