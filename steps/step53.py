import os
if '__file__' in globals():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

# 学習済みモデルのファイルパス
artifact_file_path = "artifacts/my_mlp.npz"

max_epoch = 3
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

# パラメータの読み込み
if os.path.exists(artifact_file_path):
    model.load_weights(artifact_file_path)
    print("Success to load artifact!")
    pass

for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()  # type: ignore
        optimizer.update()
        sum_loss += float(loss.data) * len(t)  # type: ignore

    print('epoch: {}, loss: {:.4f}'.format(
        epoch + 1, sum_loss / len(train_set)))

# パラメータの保存
model.save_weights(artifact_file_path)
print("Success to save artifact!")
