from matplotlib import pyplot as plt
import numpy as np


a = np.array([1, 2, 3])
b = [4, 5, 6, 7]

c = zip(a, b)

for x, y in c:
    print(x)
    print(y)