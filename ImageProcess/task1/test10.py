from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import numpy.fft as fft

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
chinese = FontProperties(fname='../font/SourceHanSansSC-Normal.otf', size=13)

Fs = 1000  # 采样频率
T = 1 / Fs  # 采样周期
L = 1000  # 信号长度
t = [i * T for i in range(L)]
t = np.array(t)

S = 0.2 + 0.7 * np.cos(2 * np.pi * 50 * t + 20 / 180 * np.pi) + 0.2 * np.cos(2 * np.pi * 100 + 70 / 180 * np.pi)

complex_array = fft.fft(S)


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
axe = axes.ravel()
axe[0].plot(1000 * t[1: 51], S[1: 51], label='S', color='green')
axe[0].grid(linestyle=':')
axe[0].set_xlabel('t(毫秒）', fontproperties=chinese)
axe[0].set_ylabel('S(t)幅值', fontproperties=chinese)
axe[0].set_title('叠加信号图', fontproperties=chinese)
axe[0].set_label('S')
axe[0].legend()

S_ifft = fft.ifft(complex_array)
axe[1].plot(1000 * t[1:51], S_ifft[1: 51], label='S_ifft', color='orangered', linestyle='-.')
axe[1].set_xlabel('t(毫秒)', fontproperties=chinese)
axe[1].set_ylabel('S_ifft(t)幅值', fontproperties=chinese)
axe[1].set_title('ifft变换图', fontproperties=chinese)
axe[1].grid(linestyle=':')
axe[1].legend()

c = t[1] - t[0]
freqs = fft.fftfreq(t.size, t[1] - t[0])
pows = np.abs(complex_array)
a = freqs[freqs > 0]
b = pows[freqs > 0]
d = np.angle(complex_array[4])

axe[2].set_title('FFT变换,频谱图', fontproperties=chinese)
axe[2].set_xlabel('Frequency 频率', fontproperties=chinese)
axe[2].set_ylabel('Power 功率', fontproperties=chinese)
axe[2].tick_params(labelsize=10)
axe[2].grid(linestyle=':')
axe[2].plot(freqs[freqs > 0], pows[freqs > 0], c='orangered', label='Frequency')
axe[2].legend()

plt.tight_layout()
plt.show()
