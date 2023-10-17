---
layout: single
title: "[베어링 고장 진단 프로젝트#2]"
categories: 이상감지, 사이드 프로젝트
tag: [베어링, 이상감지, 사이드 프로젝트]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 베어링 고장 진단 프로젝트- 필터링

## 📈Filtering:

**대부분의 경우 베어링 주파수를 구할 때 베어링 결함 고유 주파수가 측정되는 것이 아닌 다른 요소에 의한 측대파로 나타난다.** 따라서 받은 측대파를 filtering 방법을 거쳐 고유 주파수를 구해주는 과정을 거쳐야 한다. 이때 사용되는 **Filtering 방법으로는 FFT, PSD, Auto correlation, Spectral Kurtosis + Hilbert transform이 있다.** Spectral Kurtosis 및 Hilbert transform 관련 내용은 추후에 다루기로 하고 나머지 filtering 방법에 대해서 다루겠다.

이러한 Filtering을 이해하기 앞서 기본적인 신호처리 방법에 대해 알 필요가 있다.

신호는 우리의 일상생활의 일부분이다. 오디오 신호, 사진, 비디오 모두 신호 형태로 있다. 이러한 신호는 연속적인 신호, 이산적인 신호 두가지로 나뉜다. 연속적인 신호는 우리가 자연 상태에서 들을 수 있는 신호들이다. 반면 이산적인 신호 같은 경우, 자연 상태의 신호를 디지털화 할 때 많이 사용된다. 머신러닝을 할때 측정된 데이터를 가지고 분석을 해야 하므로 이산신호를 다루게 될 것이다. 이렇게 신호를 가지고 분석을 하는 것을 신호처리라고 한다.

연속적인 신호를 이산적으로 바꾸는 과정에서 특정 sampling rate에서 sampling을 하는 과정을 거치게 된다.

![sampling_사진.png]({{site.url}}/images/2023-07-19-BearingProject/sampling_사진.png){: .align-center}

그림에서 보다시피, sample rate을 10Hz로 했을 때 더 많은 데이터를 뽑게 되고 더 정교하게 복원할 수 있다. 반면 sample rate을 5Hz로 했을 때 데이터를 적게 뽑게 되고 덜 정교한 그래프를 복원하게 된다. Sample rate을 나이퀴스트 rate보다 적게 뽑았을 시, 원본(자연 데이터)를 복원할 수 없게 된다. 따라서 샘플링을 할 때는 나이퀘스트 rate보다 크게 뽑아야 한다.

### 푸리에 변환(Fast Fourier Transform)

가장 대표적인 신호처리 방법이 **Fast Fourier Transform이다**. 푸리에 변환은 신호의 주기성을 공부할 때 사용되는 방법이다. **푸리에 변환은 신호들을 주파수 성분으로 분해하는 방법이다**.

**모든 신호는 더 간단한 형태의 신호 싸인 혹은 코싸인 형태의 신호의 합 형태로 분해가 가능하다**. 시간 영역에서 주파수 영역으로 변환하는 것을 푸리에 변환이라고 부른다. 반대 과정(주파수 영역에서 시간 영역으로 변환)하는 것을 역푸리에 변환이라고 한다.

```python

from scipy.fftpack import fft

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

t_n = 10
N = 1000
T = t_n / N
f_s = 1/T

f_values, fft_values = get_fft_values(composite_y_value, T, N, f_s)

plt.plot(f_values, fft_values, linestyle='-', color='blue')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title("Frequency domain of the signal", fontsize=16)
plt.show()
```

![image-20230719215524620]({{site.url}}/images/2023-07-19-BearingProject/image-20230719215524620.png){: .align-center}

위 그림 같은 경우 fs가 100Hz이기 때문에 FFT 스펙트럼은 fs/2인 50Hz이다. fs 값이 클수록 FFT에서 더 큰 주파수를 계산할 수 있게 된다.

위 get_fft_values 함수 같은 경우, 복잡한 신호의 주파수 벡터 값을 return하게 된다. 또한 우리가 관심을 갖는 것은 증폭의 정도이기 때문에 절댓값을 취한다.

FFT는 N points의 신호 값을 return 할 것이고 이의 N/2 값이 의미 있는 값이며 그 전 값들은 유의하지 않다.

#### PSD

PSD는 푸리에 변환과 메우 밀접한 연관이 있다. PSD 같은 경우, FFT값이 단순히 신호의 주파수 형태의 스펙트럼을 구한 것이라면, **PSD는 그것에 파워 정도도 구한 것이라고 할 수 있다**. 푸리에 변환과 거의 유사하나 각 신호의 peak 값의 높이 및 넓이가 다를 것이다.

![image-20230719222021425]({{site.url}}/images/2023-07-19-BearingProject/image-20230719222021425.png){: .align-center}

위 그림은 PSD를 시각화한 것이다.

Scipy에서 제공하기 때문에 PSD(Power Spectral Density)를 코딩으로 계산하는 것은 쉽다.

```python
from scipy.signal import welch

def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values


t_n = 10
N = 1000
T = t_n / N
f_s = 1/T

f_values, psd_values = get_psd_values(composite_y_value, T, N, f_s)

plt.plot(f_values, psd_values, linestyle='-', color='blue')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2 / Hz]')
plt.show()

```

### Auto-correlation(자기 상관)

**자기 상관함수는 신호와 시간 지연된 신호 자체의 상관계수를 계산한다**. 어떤 신호에 T sec동안 하나의 주기를 반복을 하면, 해당 신호와 해당 신호 T 시간 지연된 신호와 강한 상관관계가 있을 것이다. 이러한 원리를 활용하는 것이 자기 상관함수이다.

```python

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values

t_n = 10
N = 1000
T = t_n / N
f_s = 1/T

t_values, autocorr_values = get_autocorr_values(composite_y_value, T, N, f_s)

plt.plot(t_values, autocorr_values, linestyle='-', color='blue')
plt.xlabel('time delay [s]')
plt.ylabel('Autocorrelation amplitude')
plt.show()
```

![image-20230720145315485]({{site.url}}/images/2023-07-19-BearingProject/image-20230720145315485.png){: .align-center}

자기 상관함수를 통해 생성된 값의 Peak값의 시간 축을 주파수 축으로 변환하면 FFT와 같은 값과 같게 된다.

### ⚡Wavelet 방법:

앞서 푸리에 변환에 대해서 배운 바 있다. 푸리에 변환은 신호를 시간 차원에서 주파수 차원으로 변환하는 데 획기적인 기법이지만 **시간을 반영하지 못한다는 치명적인 단점이 존재한다.**

```python
t_n = 1
N = 100000
T = t_n / N
f_s = 1/T

xa = np.linspace(0, t_n, num=N)
xb = np.linspace(0, t_n/4, num=N/4)

frequencies = [4, 30, 60, 90]
y1a, y1b = np.sin(2*np.pi*frequencies[0]*xa), np.sin(2*np.pi*frequencies[0]*xb)
y2a, y2b = np.sin(2*np.pi*frequencies[1]*xa), np.sin(2*np.pi*frequencies[1]*xb)
y3a, y3b = np.sin(2*np.pi*frequencies[2]*xa), np.sin(2*np.pi*frequencies[2]*xb)
y4a, y4b = np.sin(2*np.pi*frequencies[3]*xa), np.sin(2*np.pi*frequencies[3]*xb)

composite_signal1 = y1a + y2a + y3a + y4a
composite_signal2 = np.concatenate([y1b, y2b, y3b, y4b])

f_values1, fft_values1 = get_fft_values(composite_signal1, T, N, f_s)
f_values2, fft_values2 = get_fft_values(composite_signal2, T, N, f_s)

fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
axarr[0,0].plot(xa, composite_signal1)
axarr[1,0].plot(xa, composite_signal2)
axarr[0,1].plot(f_values1, fft_values1)
axarr[1,1].plot(f_values2, fft_values2)
(...)
plt.tight_layout()
plt.show()
```

![image-20230720150430725]({{site.url}}/images/2023-07-19-BearingProject/image-20230720150430725.png){: .align-center}

위 그림에서도 볼 수 있듯이 Signal1과 Signal2는 다른 파형을 띄고 있다. **하지만 푸리에 변환 시 동일한 값을 구한다는 것을 알 수 있다**. 이는 푸리에 변환을 할 시 시간 차원을 고려하지 못하기 때문이다.

이를 해결하기 위해 STFT(short term Fourier Transform)이라는 기법이 사용되기도 한다. **해당 방법은 원본 신호를 동일한 길이의 window를 가지고 나눠서 푸리에 변환을 하는 것이다**. 예를 들어, 하나의 신호를 10개의 구간으로 나눈다고 할 때, 2번째 구간을 보려고 하면, 해당 주기의 2/10에서 3/10이 되는 구간을 찾으면 된다.

하지만 STFT 또한 푸리에 변환의 일환이기 때문에 **푸리에 변환의 불확실성 원칙이라는 문제에서 자유롭지 못하다**. STFT에서 윈도우 크기를 줄일수록 신호 위치를 파악하기 쉽지만 주파수의 값을 구하긴 어려워진다. 반면, 윈도우 크기를 키울 수록 주파수의 값을 구하긴 쉬워지지만 신호의 위치를 구하긴 어려워진다.

이를 해결하기 위한 방법으로 Wavelet Transform이 있다.

푸리에 변환은 싸인형태로 신호를 반환한다. 왜냐하면 하나의 신호가 싸인 신호의 선형식으로 존재하기 때문이다.

Wavelet은 싸인형태의 신호가 아닌 다양한 형태의 신호를 사용한다.

![image-20230720154304351]({{site.url}}/images/2023-07-19-BearingProject/image-20230720154304351.png){: .align-center}

싸인 신호와 Wavelet의 가장 큰 차이는 싸인 신호는 그 영역이 무한한 반면, Wavelet 같은 경우 지역(특정 시간)에 대해서 파형을 갖는다. 이러한 특성 때문에 푸리에 변환과 달리 시간적인 특성을 반영할 수 있다.

### 📝Wavelet의 이론:

푸리에 변환 같은 경우 싸인 형태의 파형을 활용하는 반면, Wavelet은 여러 형태의 Wavelet을 활용할 수 있다. 해당 Wavelet을 다 활용해 보고 그 중에서 가장 좋은 결과값을 구해내는 Wavelet을 선택하면 된다.

![image-20230721181614073]({{site.url}}/images/2023-07-19-BearingProject/image-20230721181614073.png){: .align-center}

다음은 다양한 형태의 wavelet을 나타낸다. 각각의 형태의 wavelet은 다른 특징을 가지고 있어 적합한 환경에 활용될 수 있다.

파이썬에서 제공하고 있는 wavelet 라이브러리 PyWavelets에서는 아래와 같은 다양한 형태의 wavelet을 제공한다.

```python
import pywt
print(pywt.families(short=False))
['Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal', 'Reverse biorthogonal',
'Discrete Meyer (FIR Approximation)', 'Gaussian', 'Mexican hat wavelet', 'Morlet wavelet',
'Complex Gaussian wavelets', 'Shannon wavelets', 'Frequency B-Spline wavelets', 'Complex Morlet wavelets']
```

Wavelet을 활용하여 원본 신호를 변형하는 과정은 크게 두가지 과정을 거친다.

1.  Shifting
2.  Scaling

이다.

Shifting- 시간에 따라 Wavelet을 이동시키면서 합성곱을 하게 된다. 이를 통해 시간을 반영할 수 있게 된다.

Scaling- 동일한 Wavelet을 원본 신호에 대하여 합성곱을 해도 그 크기에 따라 값이 다르다. CNN을 돌릴때도 Kernel 사이즈에 따라 합성곱 결과값이 달라지는 것과 같이 Scaling의 크기에 따라 Wavelet transform의 크기가 다르다.

Fourier 변환이 주로 주파수 차원으로 표현된다면, Wavelet 변환은 scale 차원으로 표현된다. scale 차원을 주파수 차원으로 변형하기 위해선 Wavelet의 중심 주파수에 scale을 나누면 주파수를 구할 수 있다.

Fa = Fc/a

\*Fa = pseudo-frequency

\*Fc = central frequency of Mother wavelet

\*a = scale

해당 식에 따르면, scale 값이 클수록(긴 wavelet 형태) 작은 주파수 값을 구할 수 있는 것을 알 수 있다.

scale 값이 작으면 짧은 형태의 wavelet이 생성되고 그럴수록 시간영역에서 더 정밀한 정보를 얻을 수 있다.

동일한 Wavelet 형태도 계수의 개수와 분해되는 정도에 따라 분류가 될 수 있다.

```python
import pywt
import matplotlib.pyplot as plt

db_wavelets = pywt.wavelist('db')[:5]
print(db_wavelets)
*** ['db1', 'db2', 'db3', 'db4', 'db5']

fig, axarr = plt.subplots(ncols=5, nrows=5, figsize=(20,16))
fig.suptitle('Daubechies family of wavelets', fontsize=16)
for col_no, waveletname in enumerate(db_wavelets):
    wavelet = pywt.Wavelet(waveletname)
    no_moments = wavelet.vanishing_moments_psi
    family_name = wavelet.family_name
    for row_no, level in enumerate(range(1,6)):
        wavelet_function, scaling_function, x_values = wavelet.wavefun(level = level)
        axarr[row_no, col_no].set_title("{} - level {}\n{} vanishing moments\n{} samples".format(
            waveletname, level, no_moments, len(x_values)), loc='left')
        axarr[row_no, col_no].plot(x_values, wavelet_function, 'bD--')
        axarr[row_no, col_no].set_yticks([])
        axarr[row_no, col_no].set_yticklabels([])
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
```

![image-20230721184056349]({{site.url}}/images/2023-07-19-BearingProject/image-20230721184056349.png){: .align-center}

위 그림은 동일한 'Daubechies' 형태의 Wavelet을 나타낸다. 첫번째 열은 db1, 두번째 열은 db2, 세번째 열은 db3, 네번째 열은 db4이다. db n 형태에서 n은 사라지는 구간의 수를 의미한다.

행들 간의 관계는 분해 정도를 의미한다.

위 그림에서 알 수 있듯이 사라지는 구간의 수가 증가할 수록 분해 정도는 증가하고 그래프는 더 완만해지는 것을 알 수 있다.

### 🌟DWT

Wavelet은 연속형태 혹은 이산형태로 나타난다. 해당 글에서는 이산형태인 DWT만 다루도록 하겠다.

DWT는 filter-bank 형태로 실행된다. 여기서 filter-bank은 high-pass와 low-pass filter를 활용하여 신호를 효율적으로 여러가지의 주파수 밴드 형태로 나누는 것을 의미한다.

DWT를 신호에 적용할 때, 가장 작은 scale 값에서 부터 시작한다. Fa = Fc/a 식에 따르면 scale 값이 작을수록 주파수 값이 커지므로 처음에 가장 높은 주파수 값을 분석하는 것이라고 할 수 있다. 두번째 스테이지에서는 scale 값이 2배 커지게 된다. 따라서 가장 높은 주파수의 절반에 해당하는 값을 분석하게 된다. 이런식의 계산은 최대 분해 정도를 다다를때까지 진행된다.

예를 들자면, 처음 신호의 주파수가 1000Hz라고 했을 때, 첫번째 stage에서는 신호를 low-frequency 부분과(0-500Hz) high-frequency(500Hz-1000Hz) 부분으로 나뉘게 된다. 두번째 stage에서는 low-frequency 부분의(0-500Hz)를 0-250Hz와 250-500Hz로 나뉜다. 이런식으로 진행되다 신호의 길이가 Wavelet의 크기 보다 작아질 때까지 진행된다(최대 분해 정도).

이를 시각화하면 다음과 같다.

```python
import pywt

x = np.linspace(0, 1, num=2048)
chirp_signal = np.sin(250 * np.pi * x**2)

fig, ax = plt.subplots(figsize=(6,1))
ax.set_title("Original Chirp Signal: ")
ax.plot(chirp_signal)
plt.show()

data = chirp_signal
waveletname = 'sym5'

fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(6,6))
for ii in range(5):
    (data, coeff_d) = pywt.dwt(data, waveletname)
    axarr[ii, 0].plot(data, 'r')
    axarr[ii, 1].plot(coeff_d, 'g')
    axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
    axarr[ii, 0].set_yticklabels([])
    if ii == 0:
        axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
        axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
    axarr[ii, 1].set_yticklabels([])
plt.tight_layout()
plt.show()

```

![image-20230721194641578]({{site.url}}/images/2023-07-19-BearingProject/image-20230721194641578.png){: .align-center}

위 코딩을 확인하면, DWT를 구하기 위해 pywt.dwt() 함수가 사용된다. DWT는 approximation coefficients, detail coefficient 두가지 종류의 계수를 반환한다. approximation coefficients는 low pass filter에 해당하고 detail coefficient는 high pass filter에 해당한다. 그전 단계의 DWT 값을 다시 적용하는 방식으로 진행된다.

### 💔DWT를 활용한 신호 분해

지금까지 DWT의 이론적 배경을 알아보았다. 그럼 DWT가 베어링 filtering에 어떤식으로 활용될 수 있을 까?

해당 방법은 딥러닝에서 Auto-encoder사 사용되는 방법과 유사하다. pywt.dwt() 함수를 사용하여 분해한 신호들을 다시 원본 신호를 회생시키는 과정에서 불필요한 신호들을 제거할 수 있다. 여기서 불필요한 신호는 detail coefficient에 해당한다. 이를 제거하는 과정은 pywt.threshold를 활용해 제거하는 방식이 있다.

NASA 데이터를 DWT로 신호를 분해한 후 다시 복구하는 과정을 코딩을 한 것이다.

```python
DATA_FOLDER = './FEMTO_bearing/Training_set/Bearing1_1/'
filename = 'acc_01210.csv'
df = pd.read_csv(DATA_FOLDER + filename, header=None)
signal = df[4].values

def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(signal, color="b", alpha=0.5, label='original signal')
rec = lowpassfilter(signal, 0.4)
ax.plot(rec, 'k', label='DWT smoothing}', linewidth=2)
ax.legend()
ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
ax.set_ylabel('Signal Amplitude', fontsize=16)
ax.set_xlabel('Sample No', fontsize=16)
plt.show()
```

![image-20230721202757223]({{site.url}}/images/2023-07-19-BearingProject/image-20230721202757223.png){: .align-center}

###### ❓많은 Wavelet 중 어떤 걸 활용해야 할까?

Wavelet은 정말 다양한 형태의 파형이 있다. 이 중 문제에 적합한 wavelet을 찾는 것이 주요하다. 이를 위해서는 여러 wavelet 파형 형태를 SVM classifier를 통해 분류를하고 그 정확도가 가장 좋은 파라미터를 선택하는 방식으로 진행할 수 있다.
