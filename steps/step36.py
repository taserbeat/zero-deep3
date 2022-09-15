if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable


x = Variable(np.array(2.0))
y: Variable = x * x  # type: ignore
y.backward(create_graph=True)

gx = x.grad
x.cleargrad()

z: Variable = gx ** 3 + y  # type: ignore
z.backward()

print(x.grad)
