import numpy as np


class Variable:
    """
    変数を表すクラス
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        return


data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)
