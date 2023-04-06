import numpy as np
from numpy import fft
from skimage import io


def dft(img):
    H, W = img.shape

    G = np.zeros((H, W), dtype=complex)
    x = np.tile(np.arange(W), (H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)

    for v in range(H):
        for u in range(W):
            # print(f'v == {v}')
            # print(f'u == {u}')
            G[v, u] = np.sum(img * np.exp(-2j * np.pi * (x * u / W + y * v / H))) / np.sqrt(H * W)

    return G




carrier = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
print(11111)
# a = dft(carrier)
print(22222)
b = fft.fft2(carrier)
a = np.abs(b)
c = np.angle(b)
print(33333)

t = 100 + 100j
m = np.abs(t)
n = np.angle(t)
p = m * np.cos(n) + m * np.sin(n) * 1j
q = m * np.cos(n) + m * np.sin(31 / 255 * 2 * np.pi) * 1j
o = n / (2 * np.pi) * 255
