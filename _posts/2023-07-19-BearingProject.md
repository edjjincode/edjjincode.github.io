---
layout: single
title: "[베어링 고장 진단 프로젝트]/Nasa Dataset"
categories: 이상감지
tag: [이상감지, 신호처리]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 베어링 고장 진단 프로젝트- Nasa Bearing Dataset

## 🎯Domain Analysis:

도메인 분석(배경 분석)은 머신러닝 분석을 할 때 가장 중요한 과정이다. 분석하고자 하는 대상을 잘 이해해야 그에 적합한 데이터 전처리 및 분석 모델을 사용할 수 있기 때문이다. 이번 프로젝트에서 풀고자 하는 도메인은 베어링 예지 정비이다. 예지 정비란 이상치를 탐지하여 설비의 고장 이전에 정비를 하는 것을 말한다.

### ⚙ 베어링:

베어링이란 '회전운동을 하는 축을 일정한 위치에서 지지하여 운동을 제한하고 마찰을 줄여주는 기계요소'를 지칭한다. 베어링은 거의 모든 회전체에 포함된다고 할 수 있다. 우리가 흔히 알고 있는 자동차 바퀴에도 베어링이 사용된다. 베어링이 없이 바퀴축이 온전히 지탱하게 했을 시, 바퀴 축이 힘을 온전히 받아 빠르게 마모될 가능성이 있다. 이렇듯 회전체가 회전시에 마찰을 획기적으로 감소시키는 주요 설비이다.

베어링 종류는 슬리브 베어링, 구름 베어링 크게 두가지로 나뉜다. 그 중 Nasa Bearing Dataset에 사용되는 베어링은 구름 베어링이다. 구름 베어링의 구조는 Cage 내에 고정되는 구름요소, 안쪽 내륜, 그리고 바깥 쪽 외륜으로 구성되어 있다. 베어링 결함은 이러한 구성요소 어디에서든지 진행 될 수 있다.

베어링에 하나의 결함이 발생하면 결함 주파수가 생기게 되며 성장시 베어링 내에 다른 결함을 일으키게 할 수 있다. 이렇게 되면 어떤 주파수들이 다른 주파수를 더하거나 빼 주기도 한다. 실제로 베어링 결함이 발생할 시 기본 주파수만 출력되는 경우는 없다. 결함 주파수는 이미 존재하고 있는 다른 주파수들의 측대파로써 작용한다. 예를 들어, Cage에서 결함이 발생할 시, Cage 자체 결함 주파수는 발생하지 않는다. 대신, 외륜 및 내륜의 주파수 혹은 구름요소의 결함 주파수의 측대파로 나타나게 된다. 따라서 문제의 목적에 따라 구하고자 하는 고정 주파수를 구하기 위해서는 filtering 방법이 사용되어야 한다.

### 📈Filtering 방법:

대부분의 경우 베어링 주파수를 구할 때 베어링 결함 고유 주파수가 측정되는 것이 아닌 다른 요소에 의한 측대파로 나타난다. 따라서 받은 측대파를 filtering 방법을 거쳐 고유 주파수를 구해주는 과정을 거쳐야 한다. 이때 사용되는 Filtering 방법으로는 FFT, PSD, Auto correlation, Spectral Kurtosis + Hilbert transform이 있다. Spectral Kurtosis 및 Hilbert transform 관련 내용은 추후에 다루기로 하고 나머지 filtering 방법에 대해서 다루겠다.

이러한 Filtering을 이해하기 앞서 기본적인 신호처리 방법에 대해 알 필요가 있다.

신호는 우리의 일상생활의 일부분이다. 오디오 신호, 사진, 비디오 모두 신호 형태로 있다. 이러한 신호는 연속적인 신호, 이산적인 신호 두가지로 나뉜다. 연속적인 신호는 우리가 자연 상태에서 들을 수 있는 신호들이다. 반면 이산적인 신호 같은 경우, 자연 상태의 신호를 디지털화 할 때 많이 사용된다. 머신러닝을 할때 측정된 데이터를 가지고 분석을 해야 하므로 이산신호를 다루게 될 것이다. 이렇게 신호를 가지고 분석을 하는 것을 신호처리라고 한다.

연속적인 신호를 이산적으로 바꾸는 과정에서 특정 sampling rate에서 sampling을 하는 과정을 거치게 된다.

![sampling_사진.png]({{site.url}}/images/2023-07-19-BearingProject/sampling_사진.png){: .align-center}

그림에서 보다시피, sample rate을 10Hz로 했을 때 더 많은 데이터를 뽑게 되고 더 정교하게 복원할 수 있다. 반면 sample rate을 5Hz로 했을 때 데이터를 적게 뽑게 되고 덜 정교한 그래프를 복원하게 된다. Sample rate을 나이퀴스트 rate보다 적게 뽑았을 시, 원본(자연 데이터)를 복원할 수 없게 된다. 따라서 샘플링을 할 때는 나이퀘스트 rate보다 크게 뽑아야 한다.

#### 푸리에 변환(Fast Fourier Transform)

가장 대표적인 신호처리 방법이 Fast Fourier Transform이다. 푸리에 변환은 신호의 주기성을 공부할 때 사용되는 방법이다. 푸리에 변환은 신호들을 주파수 성분으로 분해하는 방법이다.

모든 신호는 더 간단한 형태의 신호 싸인 혹은 코싸인 형태의 신호의 합 형태로 분해가 가능하다. 시간 영역에서 주파수 영역으로 변환하는 것을 푸리에 변환이라고 부른다. 반대 과정(주파수 영역에서 시간 영역으로 변환)하는 것을 역푸리에 변환이라고 한다.

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

PSD는 푸리에 변환과 메우 밀접한 연관이 있다. PSD 같은 경우, FFT값이 단순히 신호의 주파수 형태의 스펙트럼을 구한 것이라면, PSD는 그것에 파워 정도도 구한 것이라고 할 수 있다. 푸리에 변환과 거의 유사하나 각 신호의 peak 값의 높이 및 넓이가 다를 것이다.

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

#### Auto-correlation(자기 상관)

자기 상관함수는 신호와 시간 지연된 신호 자체의 상관계수를 계산한다. 어떤 신호에 T sec동안 하나의 주기를 반복을 하면, 해당 신호와 해당 신호 T 시간 지연된 신호와 강한 상관관계가 있을 것이다. 이러한 원리를 활용하는 것이 자기 상관함수이다.

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

#### ⚡Wavelet 방법:

앞서 푸리에 변환에 대해서 배운 바 있다. 푸리에 변환은 신호를 시간 차원에서 주파수 차원으로 변환하는 데 획기적인 기법이지만 시간을 반영하지 못한다는 치명적인 단점이 존재한다.

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

위 그림에서도 볼 수 있듯이 Signal1과 Signal2는 다른 파형을 띄고 있다. 하지만 푸리에 변환 시 동일한 값을 구한다는 것을 알 수 있다. 이는 푸리에 변환을 할 시 시간 차원을 고려하지 못하기 때문이다.

이를 해결하기 위해 STFT(short term Fourier Transform)이라는 기법이 사용되기도 한다. 해당 방법은 원본 신호를 동일한 길이의 window를 가지고 나눠서 푸리에 변환을 하는 것이다. 예를 들어, 하나의 신호를 10개의 구간으로 나눈다고 할 때, 2번째 구간을 보려고 하면, 해당 주기의 2/10에서 3/10이 되는 구간을 찾으면 된다.

하지만 STFT 또한 푸리에 변환의 일환이기 때문에 푸리에 변환의 불확실성 원칙이라는 문제에서 자유롭지 못하다. STFT에서 윈도우 크기를 줄일수록 신호 위치를 파악하기 쉽지만 주파수의 값을 구하긴 어려워진다. 반면, 윈도우 크기를 키울 수록 주파수의 값을 구하긴 쉬워지지만 신호의 위치를 구하긴 어려워진다.

이를 해결하기 위한 방법으로 Wavelet Transform이 있다.

푸리에 변환은 싸인형태로 신호를 반환한다. 왜냐하면 하나의 신호가 싸인 신호의 선형식으로 존재하기 때문이다.

Wavelet은 싸인형태의 신호가 아닌 다양한 형태의 신호를 사용한다.

![image-20230720154304351]({{site.url}}/images/2023-07-19-BearingProject/image-20230720154304351.png){: .align-center}

싸인 신호와 Wavelet의 가장 큰 차이는 싸인 신호는 그 영역이 무한한 반면, Wavelet 같은 경우 지역(특정 시간)에 대해서 파형을 갖는다. 이러한 특성 때문에 푸리에 변환과 달리 시간적인 특성을 반영할 수 있다.

##### 📝Wavelet의 이론:

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

###### 🌟DWT

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

###### 💔DWT를 활용한 신호 분해

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

### 📣 이상 감지

#### 🌳 Isolation Forest를 이용한 Anomaly detection

##### Isolation Forest 개념:

Isolation Forest는 이상치가 적고 다르다는 점에서 착안 된 개념이다. 의사결정 나무를 지속적으로 분기시키면서 모든 데이터 관측치의 고립 정도 여부에 따라 이상치를 판별한다.

분기 되는 깊이가 낮을 수록 이상치에 가깝고 깊이가 높을 수록 정상 데이터에 가깝다.

![isolationForest]({{site.url}}/images/2023-07-19-BearingProject/IsolationForest.png){: .align-center}

이상치는 분기되는 깊이가 얕다.

![isolationForest2]({{site.url}}/images/2023-07-19-BearingProject/IsolationForest2.png){: .align-center}

정상 데이터는 분기되는 깊이가 깊다.

##### Isolation Forest 알고리즘:

1. 무작위 데이터가 주어질 때 데이터의 샘플이 이진 분류 트리에 할당된다.

2. 이진 분류는 랜덤한 피쳐를 선택하는 걸로 시작한다. 랜덤한 피쳐가 선택되면 선택된 피쳐에서 랜덤한 값을 선택하여 threshold로 선정한다.

3. 데이터 포인트가 threshold 보다 작을 경우 왼쪽 branch로 들어간다. threshold가 클 경우 오른쪽 branch로 들어간다.

4. 사용자가 지정해준 깊이나 데이터가 완전 괴립될 때까지 2번 과정을 반복한다.

이상치(anomaly)에 해당하는 값은 anomaly score -1을 갖게 되고 정상 데이터 같은 경우 1을 갖게 된다.

##### Isolation Forest Python 코드:

```python

```

#### ABOD를 활용한 이상감지

ABOD(Angle-based Outlier Detection)은 다변량 시계열 데이터에서 이상치를 감지할 때 유용하게 사용할 수 있는 머신러닝 기법이다.

##### ABOD 개념:

다변량 데이터에서 임의의 3개의 데이터가 생성하는 각도를 면밀히 관찰하는 것이 ABOD의 핵심 개념이다. 해당 각도의 분산은 이상치와 정상치가 다르게 나타난다. 보통의 경우, 정상치의 각도 분산 값이 이상치에 비해 더 크게 나타난다 ABOD는 다른 머신러닝 이상탐지 기법과 달리, 고차원에서도 감지를 잘한다는 특징이 있다.

![ABOD.png]({{site.url}}/images/2023-07-19-BearingProject/ABOD.png)

##### ABOD 알고리즘:

1. 각각의 데이터들에 대해서 조합을 생성한 후 해당 조합이 만들어내는 각도를 각도 리스트에 저장한다.
2. 해당 리스트에서 만들어진 각도의 분산 값을 구한다.
3. 특정 임계값 이하의 값은 이상치로 분류한다.

##### 이상 감지를 위한 머신러닝 모듈 pyod:

```python
def outliers_detection(model, name):
    clf = model
    clf.fit(Y)

    outliers = clf.predict(Y)

    Y_outliers = Y[np.where(outliers==1)]
    X_outliers = X[np.where(outliers==1)]

    Y_inliers = Y[np.where(outliers==0)]
    X_inliers = X[np.where(outliers==0)]
    print(X_outliers)


    plt.scatter(X_outliers, Y_outliers, edgecolor='black',color='red', label='outliers')
    plt.scatter(X_inliers, Y_inliers, edgecolor='black',color='green', label='inliers')
    plt.title(name)
    plt.legend()
    plt.grid()
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()

    anomaly_score = clf.decision_function(Y)
    min_outlier_anomaly_score = np.floor(np.min(anomaly_score[np.where(outliers==1)])*10)/10
    plt.hist(anomaly_score, bins=n_bins)
    plt.axvline(min_outlier_anomaly_score, c='k')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Number of data points')
    plt.show()
    return anomaly_score
```

pyod는 다양한 머신러닝 기법들을 제공하고 해당 식을 활용하면 쉽게 구현할 수 있다.

##### ABOD Python 코드(pyod를 활용한 코드):

```python

from pyod.models.abod import ABOD
clf = ABOD()
clf.fit(X_train)

#get outliers scores
y_train_scores = clf.decision_scores_#트레인 데이터에 대한 outlier score를 구한다
y_test_scores = clf.decision_function(X_test)#테스트 데이터에 대한 outlier score을 구한다.

```

##### Fast ABOD 개념:

ABOD는 굉장히 좋은 이상감지 알고리즘이나 시간 복잡도가 O(n^3)이 될 만큼 시간이 많이 걸리는 알고리즘이다. 이를 개선한 것이 Fast ABOD이다. Fast ABOD는 모든 점들에 대해 가능한 모든 조합을 찾는 대신에 KNN(K-nearest neighbor)를 활용하여 분산 값을 추정한다.

#### Mahalanobis 거리와 MCD를 이용한 이상감지

##### Mahalanobis 거리

[유클리드 거리]

데이터 분포에서 데이터 간의 거리를 구할 수 있는 방법은 다양하게 있다. 이 중에서 우리가 흔히 중, 고등학교 때 배웠던 거리를 구하는 방식은 유클리드 거리이다.

![uclidean.png]({{site.url}}/images/2023-07-19-BearingProject\유클리디안거리.png)

흔히 수학 시간에 배운 두 지점의 거리를 구하는 공식이다. 이를 벡터의 관점에서 생각을 해보면, 두 벡터의 내적으로 거리를 표현한 것이라고도 생각할 수 있다.

이러한 유클리드 거리의 단점은 맥락(혹은 관계?)을 파악하지 못한다는 점이다. 맥락을 이해하기 위해서는 마할라노비스 거리를 이해하면 된다.

[마할라노비스 거리]

마할라노비스는 유클리드 거리가 고려하지 못하는 "맥락"을 고려하여 거리를 재는 방식이다. 여기서 맥락이란 무슨 뜻일까? 마할라노비스 거리에서 "맥락"은 데이터 분포를 의미한다. 즉, 다변량 데이터에서 분포의 형태를 고려하여 거리를 재겠다는 이론이다.

![uclidean_distance.png]({{site.url}}/images/2023-07-19-BearingProject\마할라노비스거리.png)

위 식에서 노란색으로 칠해진 부분은, 공분산 행렬로 변수들간의 Correlation을 고려하겠다는 것이다. 해당 식에서 분산이 1로 정규화 되고 변수들이 서로 독립일 경우, 유클리디안 거리와 같아진다.

마할라노비스 거리를 시각화 하면,

![uclidean_distance_graph.png]({{site.url}}/images/2023-07-19-BearingProject\마할라노비스거리_분포.png)

위 그림을 보면, 유클리디안 거리로 판단하였을 경우, y축에 있는 별이 x축에 있는 별보다 가까이 있는 것처럼 보이지만, 마할라노비스 거리에서는 x축에 있는 별이 y축에 있는 별보다 가까이 있다.

등고선을 고려했을 때, x 축에 있는 별이 y 축에 있는 별보다 등고선 상 가까이 위치하기 떄문이다.

##### MCD

마할라노비스 거리는 변수들간의 공분산 값을 구하는 것이기 때문에 제곱을 취하게 된다. 하지만 제곱을 취할 시, outlier를 구하는 데 굉장히 취약해질 수 밖에 없다. 이를 개선하기 위한 방법이 바로 MCD이다.

##### MCD Python 코드:

```python
from pyod.models.mcd import MCD

...

```

#### OC-SVM(One-Class SVM)

OC-SVM을 이해하기 위새서는 기본적인 SVM 개념을 알아야한다. SVM 개념은 다음 블로그에 잘 나와 있으니 참고하도록 하자.

##### SVM 개념:

[SVM 개념](https://losskatsu.github.io/machine-learning/svm/#%EB%84%88%EB%B9%84width-%EC%B5%9C%EB%8C%80%ED%99%94)

##### OC-SVM 개념:

OC-SVM 개념 또한 해당 블로그에 잘 나와있다. 자세한 수학적인 증명을 알고 싶으면 해당 블로그를 참고하자.

[OC-SVM](https://losskatsu.github.io/machine-learning/oneclass-svm/#2-one-class-svm%EC%9D%98-%EB%AA%A9%EC%A0%81)

SVM의 수학적 의미를 알고 싶으면 위 블로그를 참고하도록 하고, 여기서는 간단하게 개념에 대해 넘어가도록 하겠다.

지도학습에 사용되는 SVM은 보통 라벨링이 되어 있는 데이터를 서포트 벡터를 이용해 분류하는 것이다.

OC-SVM(One Class SVM) 같은 경우, 지도 학습이 아닌 비지도 학습 형태의 SVM이다.

OC-SVM의 목적은 라벨링이 되어 있지 않은 데이터들을 가지고 정상 데이터와 비지도 학습으로 나누는 것이다. 이를 위해서 OC-SVM은 초평면을 생성하고 초평면과 이상치와의 거리가 최대가 될 수 있도록 하여 정상과 이상치를 구분한다.

![OC-SVM_1]({{site.url}}/images/2023-07-19-BearingProject/oc-svm_1.png){: .align-center}

위 그림에 있는 빨간색 선이 초평면이고, 초평면을 기준으로 왼쪽에 있는 값은 이상치이고, 오른쪽에 있는 데이터 값은 정상치이다. 빨간색 선에서 부터 이상치까지의 거리를 Ei라고 하자, 이때 정상치는 이상치가 아니므로 Ei 값은 0이 된다.

우리는 이 Ei 값을 최대로 하게 하면 된다.

![OC-SVM]({{site.url}}/images/2023-07-19-BearingProject/oc-svm.png){: .align-center}

위 그림을 보면, W는 원점에서 초명면간의 수직 벡터를 의미하고, Xi는 이상치와 원점간의 벡터를 의미한다. 이상치 같은 경우 W 벡터와 이상치간의 내적 값이 초평면을 넘어가지 못한다.

반면 초평면 오른쪽에 위치하는 정상 데이터 같은 경우 W 벡터와 이상치 간의 내적 값이 초평면을 넘게 된다.

그리고 W벡터와 이상치의 내적 값이 원점에서 초평면까지의 거리 P에서 초평면에서 이상치까지의 거리와 같다는 것을 알 수 있다.

이를 최적화 표현으로 바꾸면,

![OC-SVM_2]({{site.url}}/images/2023-07-19-BearingProject/oc-svm_2.png){: .align-center}

로 나타낼 수 있다. 식을 잘 보면, 구하고자 하는 것이 P-Ei값의 최대값이었는 데 이를 부호를 바꾸고 최솟값을 구하는 방식으로 했다는 것을 알 수 있다.

##### oc-svm 파이썬 코드 Pyod

pyod는 oc-svm 기능 또한 제공한다.

```python
from from pyod.models.ocsvm import OCSVM

```

#### DeepSVDD를 이용한 이상감지

해당 개념은 아래 링크를 참고하였습니다.

[DeepSVDD 개념](https://ffighting.net/deep-learning-paper-review/anomaly-detection/deep-svdd/)

##### SVDD 개념:

SVDD는 정상 데이터들만을 포함하는 하나의 구를 찾아내고 그것에 벗어난 값들을 다 이상치로 판단하는 방법이다. 해당 구에서 많이 벗어나 있을수록 비정상 데이터라고 판단하는 것이다.

![SVDD]({{site.url}}/images/2023-07-19-BearingProject/svdd_문제점.png){: .align-center}

하지만 SVDD는 위 그림과 같은 상황일 때 한계가 있다. 위 그림에서 파란색 큰 원은 정상 데이터를 포한하는 구이다. 이때 X1, X2 두 점을 보면, X1 같은 경우 정상 데이터라고 할 수 있는 범위에 있으나, X2 같은 경우 정상 데이터라고 하긴 어렵다. 하지만 SVDD를 사용할 시 그대로 정상으로 판단하게 된다.

이를 보완하고자 하는 것이 Deep SVDD이다.

##### DeepSVDD:

위 SVDD가 가지고 있는 한계에 대해 생각을 해보자, 정상 데이터 구에 포함되나 비교적 정상 데이터 클러스터와 멀리 있는 값 같은 경우를 다룰 수 없는 게 SVDD의 한계이다. 이를 해결하기 위해 고안한 방법이, CNN을 활용하여 넓게 분포된 데이터를 모아주는 방법이다.

![Deep_SVDD]({{site.url}}/images/2023-07-19-BearingProject/Deep_svdd.png){: .align-center}

CNN을 사용하여 정상 이미지 데이터를 저차원의 feature 공간으로 매핑할 것이다. 이때 매핑하는 방향은 모든 정상 이미지의 feature들이 한 점을 중심으로 모이도록 한다.

![Deep_SVDD_cnn]({{site.url}}/images/2023-07-19-BearingProject/deep_svdd_cnn.png){: .align-center}

CNN이 고차원의 공간을 저차원의 공간으로 매핑 해줄 수 있는 성질을 이용한 것입니다.

![Deep_SVDD_cnn_1]({{site.url}}/images/2023-07-19-BearingProject/cnn_deep.png){: .align-center}

##### DeepSVDD 파이썬 코드 Pyod:

```python
from pyod.models.deep_svdd import DeepSVDD

...

```

#### VAE를 활용한 이상감지

##### VAE와 AE의 차이점:

VAE는 Variational Auto-Encoder의 약자로 이름만 봐서는 Auto-Encoder와 비슷한 개념이라 생각이 들지만 전혀 다른 개념이다.

AE:

![AE]({{site.url}}/images/2023-07-19-BearingProject/vae-autoencoder.png){: .align-center}

Auto Encoder의 목적은 Encoder에 있다. AE는 Encoder의 학습을 위해 Decoder를 붙인 것이다.

latent vector가 어떤 하나의 값을 가지게 된다.

VAE:

![VAE]({{site.url}}/images/2023-07-19-BearingProject/vae_1.png){: .align-center}

Variational AutoEncoder의 목적은 Decoder에 있다. Decoder의 학습을 위해 Encoder를 붙인 것이다.

추출된 latent vector의 값을 하나의 숫자로 나타내는 것이 아니라 가우시안 확률 분포에 기반한 확률값으로 나타내게 된다.

![VAE_2]({{site.url}}/images/2023-07-19-BearingProject/vae_2.png){: .align-center}

##### VAE의 개념:

VAE는 Input image X를 잘 설명하는 feature를 추출하여 Latent vector z에 담고, 이 Latent vector z를 통해 X와 유사하지만 완전히 새로운 데이터를 생성해내는 것을 목표로 한다.

✔ 이때 추출되는 feature는 가우시안 분포를 따른다고 가정한다

✔ Latent vector z는 각 feature의 평균과 분산값을 나타낸다

✔ AE의 decoder처럼 latent vector로부터 이미지를 생성해낸다고 보면 된다. 하지만 AE와 다른 점은 latent vector에 있는 값이 값이 아니라 확률 값이라는 점이다.

예를 들어, 시츄 얼굴을 그리고자 한다면, 시츄의 눈, 코, 입 등의 feature를 평균 및 분산 형태로 Latent vector z에 담고, 그 z를 이용해 시츄의 얼굴을 그리게 된다.

#### AnoGan을 활용한 이상감지

---

## ⚙ NASA Bearing Dataset 프로젝트

### NASA Bearing Dataset 설명:

샤프트(회전하는 기계요소로, 보통 단면이 원형이며, 한 부분에서 다른 부분으로, 또는 전력을 생산하는 기계에서 전력을 흡수하는 기계로 전달하는 데 사용된다.)에 총 4개의 베어링이 있다. 회전 속도는 2000 RPM이며 스프링의 원리로 인해 축 중심의 수직으로 작용하는 하중이 6000ibs이다. 모든 베어링에는 윤활재가 발라져 있다.

NASA Bearing Dataset에는 크게 3가지 데이터들이 들어 있다. 각각의 데이터들은 정상에서 고장날 때까지의 데이터를 담고 있다. 각 데이터들은 1초에 한번 측정한 진동 데이터 셋 데이터를 가지고 있고 각각의 파일이 20,480 포인츠와 20kHz의 샘플링 레이트로 추출되었다.

🛢 데이터셋 1:

☑ 데이터셋 1은 1개의 베어링 당 2개의 엑셀로미터(x,y 축)을 나타낸 데이터 셋이다.

☑ 시기:
03.10.22~03.11.25

☑ 2156개의 파일

☑ 8개의 채널(4개의 베어링, 각 베어링 당 2개의 가속도계)

베어링 1: 1, 2 채널
베어링 2: 3, 4 채널
베어링 3: 5, 6 채널
베어링 4: 7, 8 채널

✅ 마지막에는 베어링 3, 베어링 4가 완전히 고장나는 구조

데이터셋 2, 3은 1개의 베어링 당 1개의 엑셀로미터를 나타낸 데이터 셋이다.

🛢 데이터 셋 2:

☑ 시기:
04.02.12~04.02.19

☑ 984개 파일

☑ 4개 채널(4개의 베어링, 각 1개의 가속도계)

베어링1: 채널 1
베어링2: 채널 2
베어링3: 채널 3
베어링4: 채널 4

✅ 결국 베어링 1에서 완전히 고장나는 데이터 셋

### 📊 데이터셋 2 분석

Nasa 데이터 셋에서 우선 각 베어링 별로 채널이 1개인 데이터 셋 2번을 분석하였다. Nasa Bearing Dataset 특성 상 정상이었던 베어링이 시간이 갈수록 마모가 되어 결함이 발생해 비정상 데이터로 바뀌는 과정을 담았으므로 가장 중요한 부분이 결함 발생 시점을 결정하는 것이 중요하다고 판단하였다.데이터 특성상 결함 과 비결함 부분을 labeling 하지 않았으므로 통계적인 방법 및 비지도 학습 방식을 활용하여 접근하였다.
