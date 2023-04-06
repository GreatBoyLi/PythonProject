import numpy as np

a = np.zeros((3, 4))
b = np.ones((3, 4))
c = np.stack([a, b])
print(c)
