if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'  # type: ignore
y.backward(create_graph=True)  # type: ignore

# iters = 0のときは1階微分、iters = 1のときは2階微分、...
iters = 0

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)  # type: ignore

gx = x.grad
gx.name = 'gx' + str(iters + 1)  # type: ignore
plot_dot_graph(gx, verbose=False, to_file='tanh.png')
