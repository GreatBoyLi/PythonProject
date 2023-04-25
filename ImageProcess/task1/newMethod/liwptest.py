import numpy as np

a = np.ones((2,2), dtype=np.uint8)
a[0][0] = -1
print(a)