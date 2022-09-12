# Add import path for the dezero directory.
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa
from dezero import Variable  # noqa


x = Variable(np.array(1.0))
y = (x + 3) ** 2  # type: ignore
y.backward()

print(y)
print(x.grad)
