---
layout: single
title: [NASA Turbofan 잔여수명 예측]/잔여수명 예측"
categories: 잔여수명_예측
tag: [Python, RUL]
toc: true
author_profile: false
sidebar:
nav: "docs"
---

# Nasa TurboFan 잔여 수명 예측 Project

## 연구 문제 및 주제

고가의 장비들이 즐비한 공장에서 장비 이상으로 이한 공정이 중단하게 되면 장비 수리비와 함께 공정 중단으로 인한 막대한 손실이 발생하게 된다. 따라서 공정이 중단되기 전에 유지 보수를 수행해야 하는 시기를 추정하는 잔여 수명 예측은 공장 운영의 성패를 가를 수 있는 중요한 문제라고 할 수 있다.

본 블로그에서는 NASA에서 제공하는 NASA Turbofan Engine 데이터를 사용하여 잔여 수명 예측에 사용되는 대표적인 3가지 방법들을 활용하여 터보 엔진의 잔여 수명을 예측 해보고 각 모델을 서로 분석 해보고자 한다.

공장의 상태를 관리하기 위해 사용되는 대표적인 방법에 이상 감지가 있다. 이상 감지는 공장에서 상태를 점검하고 싶은 장비에 센서를 부착하여 비정상적인 동작이 감지되면 즉각적으로 조치를 취하는 방법이다. 이를 통해 생산하는 제품에 비정상적인 동작이 감지되면 제품 불량률을 줄이고 더 나은 제품을 생산하여 제품의 품질을 향상시킬 수 있고 미리 이상을 감지하여 조치를 취함으로써 장비 고장 및 생산 중단을 방지하여 비용을 절감 할 수 있다.

하지만 Nasa Turbofan Dataset과 같이 실시간 데이터가 아닌 데이터 셋에 이상 감지를 적용하는 것은 조심히 접근할 필요가 있다. 이상 감지는 통상적으로 실시간으로 이상 상태를 감지하고 조치를 취할 때 유의미한 결과를 얻어낼 수 있기 때문이다. 예를 들어, 반도체 공정에서 웨이퍼를 이송 시키는 oht에 이상이 생겨 작동을 멈췄다고 하자. 이때 반도체 공정은 스마트 팩토리 시스템이 적용되어 있기 때문에 즉각적으로 다른 oht의 운송 루트를 변경하거나, 해당 생산라인의 생산량을 줄이는 등의 조치를 취해 bottleneck이 발생하는 것을 줄일 수 있다. 하지만 Nasa Turbofan과 같이 비행 엔진에서는 반도체 공정과 달리 스마트 팩토리 시스템이 구비되어 있지 않아 즉각적으로 이상 상태에 대처하는 것이 쉽지 않다. 따라서 센서 데이터를 활용하여 이상치를 감지하여 조치를 하는 것보다 각 센서들을 통해 장비의 수명을 예측하여 초기 조치를 취하는 것이 더 적합하다고 판단하였다.

잔여 수명 예측은 통상적으로 3가지 방법론을 사용한다. 첫 번째 방법은 유사성 모델을 활용하는 것이다. 유사성 모델은 비슷한 동작을 보이는, 유사하거나 다른 구성 요소의 Run-To-Failure 데이터를 활용해 RUL을 추정하는 방식이다. 두 번째 방법은 생존 모델을 활용하는 것이다. 생존 모델은 수명 데이터가 주어졌을 때 사용된다. 마지막 세 번째 방법은 건전 지표를 활용하는 것이다. 이때는 도메인에 따라 규정된 임계 값이 주어졌을 때 사용될 수 있다. 해당 프로젝트에서는 3가지 방법들을 모두 적용시킨 후 성능을 비교하는 방식으로 진행하였다.

## 데이터 셋 가공

[사용한 라이브러리들에 대한 코드]

```Python

import pandas as pd
import numpy as np

#seaborn이나 matplotlib 사용하기
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import time

#sklearn 라이브러리를 사용하기
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import class_weight

%matplotlib inline
import matplotlib.gridspec as GridSpec

#pywt와  scipy라이브러리 사용하기
import pywt
from scipy import fftpack
from scipy import signal
from scipy import optimize
import itertools

```

[데이터 셋 불러오기]

```Python
#데이터 셋 불러오기(train 데이터, test 데이터, rul 데이터)
def prepare_data():
    dependent_var = ['RUL']
    index_columns_names =  ["UnitNumber","Cycle"]
    operational_settings_columns_names = ["OpSet"+str(i) for i in range(1,4)]
    sensor_measure_columns_names =["SensorMeasure"+str(i) for i in range(1,22)]
    input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names

    df_train = pd.read_csv('/content/drive/MyDrive/CMAPSS/train_FD001.txt',delim_whitespace=True,names=input_file_column_names)

    rul = pd.DataFrame(df_train.groupby('UnitNumber')['Cycle'].max()).reset_index()
    rul.columns = ['UnitNumber', 'max']
    df_train = df_train.merge(rul, on=['UnitNumber'], how='left')
    df_train['RUL'] = df_train['max'] - df_train['Cycle']
    df_train.drop('max', axis=1, inplace=True)

    df_test = pd.read_csv('/content/drive/MyDrive/CMAPSS/test_FD001.txt', delim_whitespace=True, names=input_file_column_names)

    y_true = pd.read_csv('/content/drive/MyDrive/CMAPSS/RUL_FD001.txt', delim_whitespace=True,names=["RUL"])
    y_true["UnitNumber"] = y_true.index + 1

    return df_train, df_test, y_true
```

[EDA]

[고유 기계 번호와 사이클 수 기술 통계량]

각 고유 기계 번호와 사이클 수의 통계 값을 통해 데이터 분포를 확인하였다.

```Python
index_names = ['UnitNumber', 'Cycle']
df_train[index_names].describe()
```

![description1]({{site.url}}/images/2023-10-15-NasaTurbofan/eda_decription.png){: .align-center}

- Count 값을 확인하면 데이터 집합의 행 수는 총 20631개라는 것을 알 수 있다.
- Min-Max 값을 통해 기계는 1에서 100개까지 있다는 것을 알 수 있다.
- 평균과 분위수가 기술 통계량과 깔끔하게 일치하지 않는 것을 확인 할 수 있다. 이는 각 기계마다 Max Cycle이 다르기 때문이다.

[고유 기계 별 사이클 수 기술 통계량]

고유 기계 별 사이클 수를 확인하는 과정을 거쳤다.

```Python
df_train[index_names].groupby("UnitNumber").max().describe()
```

![description2]({{site.url}}/images/2023-10-15-NasaTurbofan/eda_decription1.png){: .align-center}

- Min 값을 보면 가장 일찍 고장난 엔진은 128, 가장 오래 작동한 엔진은 362 사이클 후 고장 나는 것을 알 수 있다.

- 평균 엔진은 199 사이클과 206 사이클 사이에서 고장나지만 표준편차를 보면 46 사이클이 나는 것을 알 수 있다. 이는 비교적 표준편차가 크다고 볼 수 있다.

[각 센서 값의 기술 통계량]

```python
df_train[sensor_cols].describe().transpose()
```

![description3]({{site.url}}/images/2023-10-15-NasaTurbofan/eda_decription3.png){: .align-center}

해당 통계 값을 보면, 센서 1, 10, 18, 19가 변동하지 않는 것을 알 수 있다. 이는 유용한 정보를 전달하지 않는다고 판단할 수 있다. 또한 분위수를 통해 센서 5, 6, 16도 변동이 거의 없음을 알 수 있다. 이 또한 데이터를 한번 알아봐야 한다는 것을 알 수 있다.

[RUL 값의 히스토그램]

```Python
sc = MinMaxScaler(feature_range=(0,1))
df_train[sensor_cols] = sc.fit_transform(df_train[sensor_cols])
df_test[sensor_cols] = sc.transform(df_test[sensor_cols])
```

```Python
df_train.groupby("UnitNumber").size().plot(kind="hist")
plt.title("Number of samples per engine")
plt.show()
```

![histogram]({{site.url}}/images/2023-10-15-NasaTurbofan/histogram.png){: .align-center}

RUL 데이터 셋을 확인하면 좌측으로 skewed 되어 있다는 것을 알 수 있다.

## 방법론

유사성 모델, 건전 지표 활용 모델, 생존 모델 총 3가지 방법을 통해 Nasa Turbofan 데이터 셋의 잔여 수명 예측을 진행하고 그 중 가장 적합한 모델을 찾아내는 것을 목표로 한다.

## 이론

### Data Preprocessing(데이터 전처리)

대부분의 경우 한 공정 안에 있는 장비들의 주파수를 구할 때 각 장비의 고유 주파수가 측정되는 것이 아닌 다른 장비의 결함 요소로 인한 측대파로 나타난다. 따라서 받은 측대파를 filtering 방법을 거쳐 고유 주파수를 구해주는 과정을 거쳐야 한다. 이때 filtering 하는 방식으로는 급속 푸리에 변환(Fast Fourier Transform), 이동 평균 방법(Moving Average), 이산 웨이블릿 변환(Discrete wavelet Transform) 등이 있다. 기존 연구에서는 대부분 이동평균법을 활용하여 데이터를 전처리 하는 과정을 거쳤으나 본 보고서에는 Discrete wavelet Transform을 활용하여 데이터를 전처리한 방식으로 진행하였다.

### Fast Fourier Transform

신호처리를 할 수 있는 가장 대표적인 방법 중 하나가 바로 Fast Fourier Transform이다. 푸리에 변환은 신호의 주기성을 공부할 때 사용되는 방법이다. 푸리에 변환은 신호들을 주파수 성분으로 분해하는 방법이다.
모든 신호는 더 간단한 형태의 신호 싸인 혹은 코싸인 형태의 신호의 합 형태로 분해가 가능하다. 시간 영역에서 주파수 영역으로 변환하는 것을 푸리에 변환이라고 부른다. 반대 과정을 하는 것을 역푸리에 변환이라고 한다.

### Discrete Wavelet Transform

푸리에 변환은 신호를 시간 차원에서 주파수 차원으로 변환하는 데 획기적인 기법이지만 시간을 반영하지 못한다는 치명적인 단점이 존재한다. 이를 해결하기 위해 사용되는 방법이 STFT(Short Term Fourier Transform)이다. 해당 방법은 원본 신호를 동일한 길이의 window를 가지고 나눠서 푸리에 변환을 하는 것이다.
하지만 STFT 또한 푸리에 변환의 일환이기 때문에 푸리에 변환의 불확실성 원칙이라는 문제에서 자유롭지 못하다. STFT에서 윈도우 크기를 줄일수록 신호 위치를 파악하기 쉽지만 주파수의 값을 구하긴 어려워진다. 반면 윈도우 크기를 키울수록 주파수의 값을 구하긴 쉬워지지만 신호의 위치를 구하긴 어려워진다.
이를 해결하기 위한 방법으로 Wavelet Transform을 사용할 수 있다. 푸리에 변환은 싸인 형태로 신호를 반환한다. 왜냐하면 하나의 신호가 싸인 신호의 선형식으로 존재하기 때문이다. Wavelet은 싸인 형태의 신호가 아닌 다양한 형태의 신호를 사용한다.

![wavelet]({{site.url}}/images/2023-10-15-NasaTurbofan/wavelet.png){: .align-center}

싸인 신호와 Wavelet의 가장 큰 차이는 싸인 신호는 그 영역이 무한한 반면, Wavelet 같은 경우 특정 지역에 대한 파형을 갖는다. 이러한 특성 때문에 Wavelet은 푸리에 변환과 달리 시간적인 특성을 반영할 수 있다.
Wavelet은 여러 형태의 파형을 제공한다. 따라서 여러 Wavelet 중 가장 좋은 결과 값을 구해내는 Wavelet을 선택하면 된다. 다음은 wavelet의 종류이다.

![discrete_wavelet]({{site.url}}/images/2023-10-15-NasaTurbofan/dwt_wavelet.png){: .align-center}

[DWT]

Wavelet은 연속형태 혹은 이산형태로 나타난다. 해당 포스트에서는 이산 형태의 DWT만 다루도록 하겠다.
DWT는 filter-bank 형태로 실행된다. 여기서 filter-bank은 high-pass와 low-pass filter를 활용하여 신호를 효율적으로 여러 가지의 주파수 밴드 형태로 나누는 것을 의미한다.
DWT를 신호에 적용할 때, 가장 작은 scale 값에서부터 시작한다. Fa = Fc/a 식에 따르면 scale 값이 작을수록 주파수 값이 커지므로 처음에 가장 높은 주파수 값을 분석하는 것이라고 할 수 있다. 두 번째 스테이지에서는 scale 값이 2배 커지게 된다. 따라서 가장 높은 주파수의 절반에 해당하는 값을 분석하게 된다. 이런 식의 계산은 최대 분해 정도를 다 다를 때까지 진행된다.
예를 들자면, 처음 신호의 주파수가 1000Hz라고 했을 때, 첫 번째 stage에서는 신호를 low-frequency 부분과(0-500Hz) high-frequency(500Hz-1000Hz) 부분으로 나뉘게 된다. 두 번째 stage에서는 low-frequency 부분의(0-500Hz)를 0-250Hz와 250-500Hz로 나뉜다. 이런 식으로 진행되다 신호의 길이가 Wavelet의 크기 보다 작아질 때까지 진행된다. 이를 시각화하면 다음과 같다.

![dwt_신호처리]({{site.url}}/images/2023-10-15-NasaTurbofan/dwt신호처리.png){: .align-center}

[DWT를 활용한 신호 분해]

Dwt가 베어링 filtering에 활용되는 방식은 딥러닝에서 Auto-Encoder가 사용되는 방법과 유사하다. Pywt.dwt()함수를 사용하여 분해한 신호들을 다시 원본 신호를 회생시키는 과정에서 불필요한 신호들을 제거할 수 있다. 다음은 dwt를 활용하여 신호를 처리하기 전과 후를 나타내는 그림이다.

![dwt_신호분해]({{site.url}}/images/2023-10-15-NasaTurbofan/dwt신호분해.png){: .align-center}

## 잔여 수명 예측 방법론

### 유사성 모델(Similarity Model)

비슷한 동작을 보이는, 유사하거나 다른 구성요소의 Run-To-Fail 데이터를 담은 데이터 베이스가 있다면 유사성 모델을 활용하여 잔여 수명(RUL)을 예측할 수 있다. Nasa Turbofan 데이터 셋은 총 100개의 기계의 Run-To-Fail 데이터가 있기 때문에 유사성 모델을 활용하여 잔여 수명 예측을 할 수 있다.
유사성 모델은 이전 데이터와 현재 데이터의 유사성을 학습하고 예측을 수행하는 방법이다. 이전 데이터와의 유사성을 기반으로 시스템의 상태를 추론하고, 잔여 수명 예측을 한다.

### 유사성 모델(Similarity Model)의 Feature Selection

**[추세를 활용한 Feature Selection]**

모델을 생성할 때 모델에 적합한 특성을 선택하여 분석을 하는 것이 중요하다. 유사성 모델을 사용할 때는 추세를 활용하여 특성을 추출하였다. 각 센서 값과 시계열과의 상관관계를 구한 후 상관관계가 적은 값들을 제거하는 과정을 거쳤다.

```Python
def trendability(col):
    trend_vals = []
    for i in df_train.UnitNumber.unique():
        dff = df_train[df_train.UnitNumber == i]
        trend_vals.append(dff[['Cycle', col]].corr().iloc[0][1])
    return np.abs(np.mean(trend_vals))
```

시계열과의 상관성을 구할 수 있는 함수를 만들었다. trendability value 값이 클 수 록 시계열과 상관성이 높은 feature라고 할 수 있다.

```Python
trend_df_list = []
for col in sensor_cols:
    trend_df_list.append({'feature': col, 'trendability_val': trendability(col)})
trend_df = pd.DataFrame(trend_df_list, columns = ['feature', 'trendability_val'])

```

각 센서별로 trenability value를 구한 후 barplot을 구하였다.

```Python
fig, ax = plt.subplots(figsize = (7,10))
sns.barplot(y = trend_df.feature, x = trend_df.trendability_val)
```

![trendability_barplot]({{site.url}}/images/2023-10-15-NasaTurbofan/barplot.png){: .align-center}

이 중 trendability value 값이 0.2보다 큰 값을 선택하였다.

![trendability]({{site.url}}/images/2023-10-15-NasaTurbofan/trendability.png){: .align-center}

**[DWT를 활용한 신호처리]**

앞서 언급했듯이, 각 센서 데이터는 서로 영향을 주기 때문에 센서에 노이즈가 껴져 있다. 따라서 이를 신호처리를 활용하여 노이즈를 제거하는 과정을 거쳐야 한다.

```Python
df = df_train[feats]
```

Daubechies Wavelet을 사용하여 높은 주파수를 가진 파형을 처리하는 과정을 거쳤다.Threshold 값은 0.63으로 두었다.

```Python
def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal
```

만든 lowpassfilter의 출력 값을 dwt_list에 저장하는 과정을 거쳤다.

```Python
dwt_list = []
for i in range(len(feats)):
    dwt = lowpassfilter(df.iloc[:, i], 0.4)
    dwt_list.append(dwt)
```

dwt_list를 데이터 프레임화 시키는 과정을 거쳤다.

```Python
df_dwt = pd.DataFrame(dwt_list)
df_dwt = df_dwt.T
df_dwt.columns = df.columns

df_1 = df_train["UnitNumber"]
df = pd.concat([df_1, df_dwt], axis = 1)
df = df.dropna()
df_2 = df_train[["Cycle", "RUL"]]

df_new = pd.concat([df, df_2], axis = 1)
```

**[각 센서들을 전체를 대표할 수 있는 선형 열화 모델을 만든다]**

잔여 수명 값(RUL)을 Min Max Scaling을 통해 정규화를 해준 후 "HI"(Hazard Index) 칼럼을 생성한 후 값을 넣었다.

```Python
df_new['HI'] = df_new.groupby('UnitNumber').RUL.transform(lambda x: minmax_scale(x))
```

![HI]({{site.url}}/images/2023-10-15-NasaTurbofan/HI.png){: .align-center}

```Python
sns.lineplot(data= df_new[df_new.UnitNumber < 31], x = 'Cycle', y = 'HI', hue= 'UnitNumber')
```

![lineplot]({{site.url}}/images/2023-10-15-NasaTurbofan/lineplot.png){: .align-center}

각 기계들의 고유 번호에 대해 lineplot으로 시각화하면 선형적 열화 모델을 따른다는 것을 알 수 있다.

전체 센서 값들을 대표할 수 있는 하나의 열화 모델을 만들기 위해 추출된 특성을 독립변수로 놓고 “Hi” 값을 종속변수로 놓고 선형 회귀를 돌렸다. 해당 회귀 모델의 coefficient 값은 다음과 같다.

![HI_coefficient]({{site.url}}/images/2023-10-15-NasaTurbofan/HI_coefficient.png){: .align-center}

```Python
model = LinearRegression()
```

```Python
X = df_new[feats]
y = df_new.HI
X.shape, y.shape
```

```Python
model.fit(X, y)
model.score(X,y)
```

```Python
model.coef_
```

구한 선형 회귀 모델의 coefficient 값을 센서들의 데이터와 내적하여 모든 데이터를 대표할 수 있는 선형 열화 지수 “Hi_final”을 만든다.

```Python
df_new["HI_final"] = df_new[feats].dot(model.coef_)
df_new.HI_final.head()
```

![HI_final]({{site.url}}/images/2023-10-15-NasaTurbofan/HI_final.png){: .align-center}

![HI_final_1]({{site.url}}/images/2023-10-15-NasaTurbofan/HI_final_1.png){: .align-center}

다음은 x 값을 Cycle로 놓고 y 값을 HI_final로 놓은 후 시각화 한 그래프이다.

[이동 평균을 활용한 그래프]

```Python
sns.lineplot(data= df_new[df_new.UnitNumber < 31], x = 'Cycle', y = 'HI_final', hue= 'UnitNumber')
plt.ylabel('Health Indicator')
```

![이동평균_그래프]({{site.url}}/images/2023-10-15-NasaTurbofan/이동평균그래프.png){: .align-center}

```Python
for i in range(1,101):
    sns.lineplot(data= df_new[df_new.UnitNumber == i], x = 'Cycle', y = 'HI_final', color = 'green', lw = 0.2)
sns.scatterplot(data = df_new[df_new.HI == 0], x = 'Cycle', y = 'HI_final', label = 'Failure',
                marker = 'X', color = 'black')
plt.ylabel('Health Indicator')
```

![이동평균1]({{site.url}}/images/2023-10-15-NasaTurbofan/이동평균1.png){: .align-center}

[DWT를 활용한 그래프]

![dwt그래프]({{site.url}}/images/2023-10-15-NasaTurbofan/dwt를 활용한 그래프.png){: .align-center}

![dwt그래프1]({{site.url}}/images/2023-10-15-NasaTurbofan/dwt_graph.png){: .align-center}

[융합된 데이터 다차방정식에 피팅하기]

구한 “Hi_final” 지수를 가지고 다차방정식을 생성한다. 다차 방정식이 wavelet을 통과한 데이터와 적합이 되도록 차수를 맞춰줘야 한다. 전체적인 그래프의 추이를 보고 크게 2가지 차수(이차와 사차)를 선택하여 실험하였다. 이차 방정식을 선택한 이유는 그래프의 전체적인 추이가 점진적으로 감소하기 때문이다. 사차 방정식을 선택한 이유는 그래프가 wavelet을 통과하면서 두 번의 굴곡이 있는 사차 방정식 형태를 띠기 때문이다.
아래 그래프는 각각 기계 고유 번호가 1인 기계의 이차곡선 형태의 열화모델과 사차 곡선 형태의 열화모델의 그래프이다.

각 기계의 고유번호와 각 기계별 파라미터들을 하나의 데이터 프레임 params_df에 저장하였다. 해당 파라미터들은 주기에 따른 HI의 예측을 계산하는 데 사용된다.

```Python
params_list = []
for i in range(1,101):
    y = df_new.HI_final[df_new.UnitNumber == i]
    cycle = df_new.Cycle[df_new.UnitNumber == i]
    theta_2, theta_1, theta_0 = np.polyfit(cycle, y, 2)
    params_list.append({'UnitNumber':i, 'theta_0': theta_0, 'theta_1': theta_1, 'theta_2': theta_2})
params_df = pd.DataFrame(params_list, columns = ['UnitNumber', 'theta_2', 'theta_1', 'theta_0'])
```

```Python
params_df.head()
```

[이동평균법-2차 곡선 파이썬 코드]

```Python
HI = df_new.HI_final[df_new.UnitNumber == 1]
cycle = df_new.Cycle[df_new.UnitNumber == 1]
theta_0 = params_df.theta_0[params_df.UnitNumber == 1].values
theta_1 = params_df.theta_1[params_df.UnitNumber == 1].values
theta_2 = params_df.theta_2[params_df.UnitNumber == 1].values
HI_fit = theta_0 + theta_1*cycle + theta_2*cycle*cycle
```

```Python
plt.plot(cycle,HI, label = 'True')
plt.plot(cycle,HI_fit, label = 'Fitted')
plt.ylabel('Health Indicator')
plt.xlabel('Cycle')
plt.legend()
plt.title('Health Indicator of Unit 1');
```

[이동평균법-2차 곡선 형태]

![health_indicator]({{site.url}}/images/2023-10-15-NasaTurbofan/health_indicator.png){: .align-center}

[DWT-2차 곡선 형태]

![dwt_2차곡선]({{site.url}}/images/2023-10-15-NasaTurbofan/dwt_2차곡선.png){: .align-center}

[DWT-4차 곡선 형태]

![dwt_4차곡선]({{site.url}}/images/2023-10-15-NasaTurbofan/dwt_4차곡선.png){: .align-center}

**[테스트 데이터 준비]**

테스트 데이터 셋에 대해서도 훈련 데이터 셋과 동일하게 dwt를 통해 신호처리를 하였다.

```Python
dwt_test_list = []
for i in range(len(feats)):
    dwt = lowpassfilter(df_test[feats].iloc[:, i], 0.4)
    dwt_test_list.append(dwt)
```

테스트 데이터 셋에 대해서도 dwt를 활용하여 신호처리를 해 데이터를 전처리한다.

```Python
df_test_dwt = pd.DataFrame(dwt_test_list)
```

```Python
df_test_dwt  = df_test_dwt.T
```

```Python
df_test_dwt.columns = feats
```

훈련 데이터 셋에서 생성한 계수를 그대로 테스트 데이터 셋에 사용해 “HI”(Hazard Index) 변수를 생성하였다.

```Python
df_test['HI'] = df_test_dwt.dot(model.coef_)
df_test.HI.head()
```

**[유사성 점수 생성하기]**

테스트 데이터 셋에서 HI(Hazard Index)를 생성하였으니, 유사성 점수를 계산하여 더 적합한 모델을 찾아내야 한다. 유사성 점수를 구하기 위해서는 우선 각 기계의 HI 예측 값을 구해야 한다.

HI의 예측 값을 구하기 위해선 다음과 같은 과정을 거쳐야 한다.

1. 테스트 데이터 셋의 cycle 값을 구한다
2. 훈련 데이터 셋에서 생성한 모델의 파라미터 값(param_df의 파라미터)와 cycle(테스트 데이터 셋의 사이클)을 통해 다차방정식을 생성한다.
3. 생성한 다차방정식의 값이 바로 HI의 예측 값, Pred_HI이다.
   Pred_HI 값을 구했으면 이를 실제 HI 값과 비교하여 잔차를 구할 수 있다. 잔차는 예측과 관측 값 간의 차이를 의미한다. 구한 잔차를 통해 유사성 점수를 구하는 것은 쉽다. 잔차의 제곱을 exponential 시키면 된다. 잔차가 작을수록 유사성 점수가 높아지게 된다. 즉 모델의 예측이 관측값과 더 가까울수록 유사성 점수가 높다고 할 수 있다.

```Python
list_test_fit = []
for i in df_test.UnitNumber.unique():
    HI = df_test.HI[df_test.UnitNumber == i]
    cycle = df_test.Cycle[df_test.UnitNumber == i]
    for j in params_df.UnitNumber.unique():
        theta_0 = params_df.theta_0[params_df.UnitNumber == j].values
        theta_1 = params_df.theta_1[params_df.UnitNumber == j].values
        theta_2 = params_df.theta_2[params_df.UnitNumber == j].values
        pred_HI = theta_0 + theta_1*cycle + theta_2*cycle*cycle
        Residual = np.mean(np.abs(pred_HI - HI))
        total_life = df_new.Cycle[df_new.UnitNumber == j].max()
        similarity_score = np.exp(-Residual*Residual)
        list_test_fit.append({'UnitNumber':i, 'Model': j, 'Residual': Residual,
                              'similarity': similarity_score, 'total_life': total_life})
df_test_fit = pd.DataFrame(list_test_fit, columns=['UnitNumber', 'Model', 'Residual', 'similarity', 'total_life'])
```

```Python
df_test_fit.head()
```

![df_test_fit]({{site.url}}/images/2023-10-15-NasaTurbofan/df_test_fit.png){: .align-center}

df_test_fit 데이터 프레임에서 각 기계(UnitNumber) 마다 유사성 점수가 5번째로 높은 것까지 추출하여 result_df_5라는 새로운 데이터 프레임을 만들었다.

```Python
ind_5 = df_test_fit.groupby('UnitNumber')['similarity'].nlargest(5).reset_index()['level_1']
result_df_5 = df_test_fit.iloc[ind_5]
result_df_5.head()
```

![result_df_5]({{site.url}}/images/2023-10-15-NasaTurbofan/result_df_5.png){: .align-center}

이후 result_df_5에서 구한 각 기계 별 total_life의 평균과 테스트 데이터의 기계 별 사이클의 평균의 차이를 통해 잔여 수명 예측 값을 구한다.

```Python
y_true_5 = y_true.copy()

y_true_5['Pred_RUL'] = (result_df_5.groupby('UnitNumber')['total_life'].mean() - df_test.groupby('UnitNumber')['Cycle'].max()).values
y_true_5.head()
```

![pred_rul]({{site.url}}/images/2023-10-15-NasaTurbofan/pred_rul.png){: .align-center}

```Python
sns.regplot(x = y_true_5.Pred_RUL, y = y_true_5.RUL)
plt.xlabel('Predicted RUL')
plt.ylabel('True RUL')
```

```Python
fig, ax = plt.subplots(figsize = (15, 7))
sns.lineplot(x = y_true_5.UnitNumber, y = y_true_5.Pred_RUL, label = "Predicted RUL")
sns.lineplot(x = y_true_5.UnitNumber, y = y_true_5.RUL, label = "True RUL")
plt.xlabel("Unit Number")
plt.ylabel("Remaining Useful Life")
plt.legend(loc = 1)
```

```Python
mean_squared_error(y_true_5.RUL, y_true_5.Pred_RUL)
```

```Python
from sklearn.metrics import mean_squared_error, r2_score
```

```Python
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))
```

```Python
evaluate(y_true_5.RUL, y_true_5.Pred_RUL)
```

```Python
mean_absolute_error(y_true_5.RUL, y_true_5.Pred_RUL)
```

### 건전성 지표 모델(Health Indicator Model)

건전성 지표에 사용되는 데이터를 추출하는 과정은 앞 과정에서 다뤘으므로 생략하고 바로 피쳐 선택(Feature Selection) 단계를 진행하겠다.

**[단조 특성을 활용한 Feature Selection]**

각 센서들의 단조 특성 값을 구할 수 있는 함수를 설정한 뒤 각 센서들의 단조 특성 값을 구하였다.

```Python
def monotonicity(data):

    num_pos = data[data > 0].shape[0]
    num_neg = data[data < 0].shape[0]
    tot_n = data.shape[0] - 1

    mon_val = np.abs(num_pos - num_neg)/tot_n
    return mon_val
```

```Python
mon_df = pd.DataFrame(columns = ['feature', 'monotonicity_val'])

for col in sensor_cols:
    mon_val = []
    for unit in df_lag.UnitNumber.unique():
        mon_val.append(monotonicity(df_lag.loc[df_lag.UnitNumber == unit, col]))
    mon_df = mon_df.append({'feature': col, 'monotonicity_val': np.mean(mon_val)}, ignore_index = True)
```

```Python
mon_df = mon_df.sort_values(by = 'monotonicity_val', ascending = False)
mon_df.head()
```

```Python
fig, ax = plt.subplots(figsize = (7,10))

sns.barplot(y = mon_df.feature, x = mon_df.monotonicity_val)
```

```Python
feats = mon_df.feature[mon_df.monotonicity_val > 0.08]
feats
```

**[데이터 전처리]**

[이동 평균법을 활용한 데이터 전처리]

불필요한 노이즈를 제거하기 위해 이동평균법, 이산 웨이블릿 변환 기법 두 가지 방법을 모두 적용해보았다.

[DWT를 활용한 데이터 전처리]

Threshold를 0.63으로 두고 wavelet 종류를 db4로 설정한 후 DWT 신호 처리를 진행하였다.

[PCA]

주성분 분석을 통해 21개의 센서 값을 대표할 수 있는 주성분을 뽑아낸다. 주성분 분석 결과 pc1이 pc2, pc3에 비해 설명력이 더 높은 것을 확인할 수 있다.

```Python
from sklearn.decomposition import PCA
```

```Python
pca = PCA(n_components=3)

pca_data = pca.fit_transform(df_train_mean[feats])

pca_df = pd.DataFrame(pca_data, columns = ['pc1', 'pc2', 'pc3'])
pca_df['UnitNumber'] = df_train_mean.UnitNumber.values
pca_df['cycle'] = pca_df.groupby('UnitNumber').cumcount()+1
pca_df['RUL'] = pca_df.groupby('UnitNumber').cycle.transform('max') - pca_df.cycle
pca_df.head()
```

![pc_graph]({{site.url}}/images/2023-10-15-NasaTurbofan/pc_graph.png){: .align-center}

```Python
pcs = ['pc1', 'pc2', 'pc3']
```

처음 두 개의 주성분으로 구성된 공간에서 데이터를 시각화 하였다.

```Python
sns.pairplot(data = pca_df[pca_df.UnitNumber == 1], x_vars= pcs, y_vars = pcs)
```

```Python
sns.scatterplot(data = pca_df[pca_df.UnitNumber == 1], x = "pc1", y = "pc2", hue = "RUL")
```

![pc_scatterplot]({{site.url}}/images/2023-10-15-NasaTurbofan/pc_scatterplot.png){: .align-center}

```Python
fig, ax = plt.subplots()
sns.lineplot(data = pca_df[pca_df.UnitNumber == 1], x = "cycle", y = "pc1", ax = ax)
plt.axhline(pca_df[pca_df.UnitNumber == 1].pc1.max(), color = 'r')
ax.set_ylabel("Health Indicator")
```

다음은 기계 고유 번호가 1인 기계의 주성분 값을 시간에 따라 나타낸 것이다. 파란색 선은 주성분 값을 나타내고 빨간색 선은 주성분의 최대값을 의미한다. 주성분과 시간이 지수 관계에 있음을 확인할 수 있다.

![pc_lineplot]({{site.url}}/images/2023-10-15-NasaTurbofan/pc_lineplot.png){: .align-center}

다음은 pc1 값을 주성분으로 했을 때 RUL(잔여 수명) 값을 도식화 한 것이다. RUL 값이 0일 때 확실하게 분류가 되는 것을 확인할 수 있다.

```Python
fig, ax = plt.subplots(figsize = (8,6))
sns.distplot(pca_df.pc1[pca_df.RUL == 0], label= "RUL: 0")
sns.distplot(pca_df.pc1[pca_df.RUL == 30], label= "RUL: 30")
sns.distplot(pca_df.pc1[pca_df.RUL == 60], label= "RUL: 60")
sns.distplot(pca_df.pc1[pca_df.RUL == 90], label= "RUL: 90")
plt.xlabel("Health Indicator")
plt.legend()
plt.show()
```

![health_indicator_graph]({{site.url}}/images/2023-10-15-NasaTurbofan/health_indicator_graph.png){: .align-center}

```Python
threshold = pca_df.pc1[pca_df.RUL == 0].mean()
threshold
```

```Python
threshold_std = pca_df.pc1[pca_df.RUL == 0].std()
threshold_std
```

![threshold_pca]({{site.url}}/images/2023-10-15-NasaTurbofan/threshold_pca.png){: .align-center}

**[지수 성능 저하 모델을 활용한 건전성 지표 생성]**

지수 성능 저하 모델은 다음과 같이 정의된다.

![지수성능저하모델]({{site.url}}/images/2023-10-15-NasaTurbofan/지수성능저하모델.png){: .align-center}

여기서 h(t)는 건전성 지표로, 시간의 함수이다. Φ 는 절편 항으로, 상수로 간주된다. θ 와 β는 모델의 기울기를 결정하는 임의 파라미터이다. 여기서 θ는 로그 정규분포이고 β는 가우스 분포이다. 각 시간 스텝 t에서 θ와 β의 분포는 h(t)의 최선 관측 값을 기준으로 사후 확률로 업데이트 된다. ϵ 은 N(0,σ2) 을 따르는 가우스 백색 잡음이다.
해당 식을 데이터에 적용하기 위해 간소화 시키면 다음과 같은 수식을 세울 수 있다.

h(t) = Φ + θ*exp(β*cycle)

각 기계 별로 주성분을 바탕으로 지수 성능 저하 모델을 돌린 후 나온 파라미터 값을 exp_params_df 데이터 프레임에 저장하였다.

```Python
def exp_degradation(parameters, cycle):
    '''
    Calculate an exponetial degradation of the form:
    ht = phi + theta * exp(beta * cycle)
    '''
    phi = parameters[0]
    theta = parameters[1]
    beta = parameters[2]

    ht = phi + theta * np.exp(beta * cycle)
    return ht
```

```Python
def residuals(parameters, data, y_observed, func):
    '''
    Compute residuals of y_predicted - y_observed
    where:
    y_predicted = func(parameters,x_data)
    '''
    return func(parameters, data) - y_observed
```

```Python
param_0 = [-1, 0.01, 0.01]
```

```Python
exp_params_df = pd.DataFrame(columns = ['UnitNumber', 'phi', 'theta', 'beta'])

for i in range(1,101):

    ht = pca_df.pc1[pca_df.UnitNumber == i]
    cycle = pca_df.cycle[pca_df.UnitNumber == i]

    OptimizeResult = optimize.least_squares(residuals, param_0, args = (cycle, ht, exp_degradation))
    phi, theta, beta = OptimizeResult.x

    exp_params_df = exp_params_df.append({'UnitNumber':i, 'phi': phi, 'theta': theta, 'beta': beta}, ignore_index = True)
```

```Python
exp_params_df.head()
```

![exp_params_df]({{site.url}}/images/2023-10-15-NasaTurbofan/exp_params_df.png){: .align-center}

생성한 지수 성능 저하 모델이 잘나오는 지 확인하기 위해 1번 기계의 주성분 값과 지수 성능 저하 모델을 시각화하였다. 시각화 결과, 지수 성능 저하 모델이 잘 피팅이 되는 것을 알 수 있다.

```Python
phi = exp_params_df.phi[exp_params_df.UnitNumber == 1].values
theta = exp_params_df.theta[exp_params_df.UnitNumber == 1].values
beta = exp_params_df.beta[exp_params_df.UnitNumber == 1].values

cycles = pca_df.cycle[pca_df.UnitNumber == 1]
pred_ht = exp_degradation([phi, theta, beta], cycles)

fig, ax = plt.subplots()
sns.lineplot(data = pca_df[pca_df.UnitNumber == 1], x = "cycle", y = "pc1", ax = ax, label = "True HI")
sns.lineplot(y = pred_ht, x = cycles, ax = ax, color = "green", label = "Fitted Exponential HI")
ax.axhline(threshold, color = 'r')
ax.text(200,threshold - 0.01,'Failure Threshold',rotation=0)
ax.set_title("Unit: 1")
ax.set_xlabel("Cycles")
ax.set_ylabel("Health Indicator")
```

```Python
fig, ax = plt.subplots(nrows = 20, ncols = 5, figsize = (30,50))

ax = ax.ravel()

for i in range(0,100):

    phi = exp_params_df.phi[exp_params_df.UnitNumber == i+1].values
    theta = exp_params_df.theta[exp_params_df.UnitNumber == i+1].values
    beta = exp_params_df.beta[exp_params_df.UnitNumber == i+1].values

    cycles = pca_df.cycle[pca_df.UnitNumber == i+1]
    pred_ht = exp_degradation([phi, theta, beta], cycles)

    sns.lineplot(data = pca_df[pca_df.UnitNumber == i+1], x = "cycle", y = "pc1", ax = ax[i])
    sns.lineplot(y = pred_ht, x = cycles, ax = ax[i], color = "green")
    ax[i].axhline(threshold, color = 'r')
    ax[i].set_title("Unit:" + str(i+1))
    ax[i].set_xlabel("")
    ax[i].set_ylabel("")

plt.tight_layout()
```

```Python
fig, ax = plt.subplots(figsize = (10,3), nrows = 1, ncols = 3)
sns.distplot(exp_params_df.phi, ax = ax[0])
sns.distplot(exp_params_df.theta, ax = ax[1], color = "red")
sns.distplot(exp_params_df.beta, ax = ax[2], color = "green")
```

![exp_params_dist]({{site.url}}/images/2023-10-15-NasaTurbofan/exp_params_dist.png){: .align-center}

**[테스트 데이터 셋 적용]**

테스트 데이터 셋에 대해서도 PCA를 진행하고 PC1을 건전 지표로 설정하였다.

```Python
window = 5

df_test_mean = df_test.groupby('UnitNumber')[feats].rolling(window = window).mean()
df_test_mean = df_test_mean.reset_index()
df_test_mean.dropna(inplace = True)
df_test_mean.drop(['level_1'], axis = 1, inplace = True)
df_test_mean.head()
```

테스트 데이터 셋에 대해서도 PCA를 진행하고 PC1을 건전 지표로 설정하였다.

```Python
pca_test_data = pca.transform(df_test_mean[feats])

pca_test_df = pd.DataFrame(pca_test_data, columns = ['pc1', 'pc2', 'pc3'])
pca_test_df['UnitNumber'] = df_test_mean.UnitNumber.values
pca_test_df['cycle'] = pca_test_df.groupby('UnitNumber').cumcount()+1
pca_test_df.head()
```

![pc1_threshold]({{site.url}}/images/2023-10-15-NasaTurbofan/pc1_threshold.png){: .align-center}

앞서 지수 성능모델을 만드는 과정을 통해 생성한 파라미터들에 대해 백분위수를 활용하여 경계(bound)를 정의하였다. 하한선을 25%, 상한선을 75%로 설정하여 백분위수 범위를 설정하였다.

```Python
phi_vals = exp_params_df.phi
theta_vals = exp_params_df.theta
beta_vals = exp_params_df.beta
```

```Python
phi_vals.mean()
```

```Python
param_1 = [phi_vals.mean(), theta_vals.mean(), beta_vals.mean()]
param_1
```

```Python
lb = 25
ub = 75
phi_bounds = [np.percentile(phi_vals, lb), np.percentile(phi_vals, ub)]
theta_bounds = [np.percentile(theta_vals, lb), np.percentile(theta_vals, ub)]
beta_bounds = [np.percentile(beta_vals, lb), np.percentile(beta_vals, ub)]
```

```Python
bounds = ([phi_bounds[0], theta_bounds[0], beta_bounds[0]],
          [phi_bounds[1], theta_bounds[1], beta_bounds[1]])
bounds
```

**[잔여 수명 예측 값 구하기]**

앞서 구한 파라미터, bound 값, 테스트 데이터 셋의 건전 지표 값(주성분 값)을 활용하여 잔여 수명 예측 값을 구할 수 있다.
잔여 수명 예측 값을 구하기 위해서는 다음과 같은 과정을 거쳐야 한다.

1. 테스트 데이터 셋의 주성분 값
2. 테스트 데이터 셋의 cycle 값
3. 지수 성능 저하 모델의 파라미터의 경계 값
4. 테스트 데이터 셋의 주성분 값과 cycle 값, 파라미터의 경계값(bound) 값을 활용하여 지수 성능 저하 모델을 생성한다.
5. 생성한 모델을 잔차를 최소화 시키는 방향으로 최적화 시킨 후 최적화 된 파라 미터를 구한다. (phi, theta, beta)
6. 구한 최적의 파라미터를 통해 전체 사이클을 구한다. Total_cycles = log(threshold-phi)/theta)/beta
7. 전체 사이클과 테스트 데이터 셋의 사이클의 최대 값의 차이가 잔여 수명 예측 값이 된다.

```Python
result_test_df = pd.DataFrame(columns = ['UnitNumber', 'phi', 'theta', 'beta', 'Pred_RUL', 'True_RUL'])

for i in pca_test_df.UnitNumber.unique():

    ht = pca_test_df.pc1[pca_test_df.UnitNumber == i]
    cycle = pca_test_df.cycle[pca_test_df.UnitNumber == i]

    OptimizeResult = optimize.least_squares(residuals, param_1, bounds=bounds,
                                            args = (cycle, ht, exp_degradation))
    phi, theta, beta = OptimizeResult.x
    total_cycles = np.log((threshold - phi) / theta) / beta
    RUL = total_cycles - cycle.max()

    result_test_df = result_test_df.append({'UnitNumber':i, 'phi': phi, 'theta': theta, 'beta': beta,
                                         'Pred_RUL': RUL, 'True_RUL': y_true.RUL[y_true.UnitNumber == i].values[0]},
                                         ignore_index = True)
```

```Python
result_test_df.head()
```

![pred_rul1]({{site.url}}/images/2023-10-15-NasaTurbofan/pred_rul1.png){: .align-center}

```Python
fig, ax = plt.subplots(nrows = 20, ncols = 5, figsize = (30,50))

ax = ax.ravel()

for i in range(0,100):

    phi = result_test_df.phi[result_test_df.UnitNumber == i+1].values[0]
    theta = result_test_df.theta[result_test_df.UnitNumber == i+1].values[0]
    beta = result_test_df.beta[result_test_df.UnitNumber == i+1].values[0]
    Pred_RUL = result_test_df.Pred_RUL[result_test_df.UnitNumber == i+1].values[0]

    cycles = pca_test_df.cycle[pca_test_df.UnitNumber == i+1]
    total_cycles = [j for j in range(1, int(cycles.max() + Pred_RUL + 1))]
    pred_ht = exp_degradation([phi, theta, beta], pd.Series(total_cycles))

    sns.lineplot(data = pca_test_df[pca_test_df.UnitNumber == i+1], x = "cycle", y = "pc1", ax = ax[i])
    sns.lineplot(y = pred_ht, x = total_cycles, ax = ax[i], color = "green")
    ax[i].axhline(threshold, color = 'r')
    ax[i].set_title("Unit:" + str(i+1))
    ax[i].set_xlabel("")
    ax[i].set_ylabel("")

plt.tight_layout()
```

```Python
mean_squared_error(result_test_df.True_RUL, result_test_df.Pred_RUL)
```

```Python
mean_absolute_error(result_test_df.True_RUL, result_test_df.Pred_RUL)
```

```Python
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

```Python
mean_absolute_percentage_error(result_test_df.True_RUL, result_test_df.Pred_RUL)
```

### Survival Model(생존 모델)

**[데이터 전처리]**

생존 모델 같은 경우, 기존에 건전성 지표 모델과 유사성 모델과 달리 전처리하는 것부터 다시 시작하였다.

[사용한 라이브러리]

```Python
import pandas as pd
import numpy as np

#seaborn이나 matplotlib 사용하기
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import time

#sklearn 라이브러리를 사용하기
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import class_weight

%matplotlib inline
import matplotlib.gridspec as GridSpec

#pywt와  scipy라이브러리 사용하기
import pywt
from scipy import fftpack
from scipy import signal
from scipy import optimize
import itertools
```

```python
!pip install lifelines
```

```Python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from lifelines.utils import k_fold_cross_validation, median_survival_times
from lifelines import KaplanMeierFitter, CoxTimeVaryingFitter, NelsonAalenFitter,\
                      CoxPHFitter, WeibullAFTFitter, WeibullFitter, ExponentialFitter,\
                      LogNormalFitter, LogLogisticFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.plotting import plot_lifetimes
```

데이터 불러오기

```Python
dir_path = '/content/drive/MyDrive/CMAPSS/'
# define column names for easy indexing
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names
# read data
train = pd.read_csv((dir_path+'train_FD001.txt'), sep='\s+', header=None, names=col_names)
test = pd.read_csv((dir_path+'test_FD001.txt'), sep='\s+', header=None, names=col_names)
df_test = pd.read_csv((dir_path+'RUL_FD001.txt'), sep='\s+', header=None, names=['time_cycles'])
# inspect first few rows
train.head()
```

RUL 값을 추가하였다.

```Python
def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame

train = add_remaining_useful_life(train)
print(train.shape)
train[index_names+['RUL']].head()
```

유의미하지 않은 데이터를 제거하여 feature selection 하는 단계를 거쳤다.

```Python
# clip RUL max as 125 means values in column greater than 125 becomes 125
train['RUL'].clip(upper=125, inplace=True)

# drop non-informative features, derived from EDA
drop_sensors = ['s_1','s_5','s_6','s_10','s_16','s_18','s_19']
drop_labels = setting_names + drop_sensors
train.drop(labels=drop_labels, axis=1, inplace=True)

remaining_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11',
                     's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
train.shape
```

cycle 수가 200 아래 있는 것은 제거하는 과정을 거쳤다.

```Python
# cut off is the censoring cycle time line
cut_off = 200
train_censored = train[train['time_cycles'] <= cut_off].copy()
print(train_censored.shape)
train_censored[train_censored["unit_nr"] == 50].tail()
```

daubechies wavelet 형태의 이산 웨이블릿 변환을 활용하여 높은 주파수를 가지고 있는 신호를 제거하는 과정을 거쳤다.

```Python
def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal
```

```Python
dwt_list = []
for i in range(len(remaining_sensors)):
    dwt = lowpassfilter(df.iloc[:, i], 0.4)
    dwt_list.append(dwt)
```

```Python
df_dwt = pd.DataFrame(dwt_list)
```

```Python
df_dwt = df_dwt.T
```

```Python
df_dwt.columns = remaining_sensors
```

```Python
df_1 = train_censored["unit_nr"]
```

```Python
df = pd.concat([df_1, df_dwt], axis = 1)
```

```Python
df_2 = train_censored[["time_cycles", "RUL"]]
```

```Python
df_new = pd.concat([df, df_2], axis = 1)
```

```Python
df_new.dropna(inplace = True)
```

“breakdown”이라는 새로운 열을 생성하였다. 해당 열은 각 기계가 고장을 일으켰는지 여부를 나타낸다. 각 기계의 마지막 사이클에서 고장을 일으키기 때문에 해당 주기의 데이터 값을 1로 설정한다.

“start”라는 새로운 열을 생성한 후 time_cycles에서 1을 뺀 값을 저장하였다.

이후 타임사이클이 200개 이하인 데이터만을 추출하였다.

```Python
train_cols = index_names + remaining_sensors + ['start', 'breakdown']
predict_cols = ['time_cycles'] + remaining_sensors + ['start', 'breakdown']
```

**[Cox Time-Varying Model 방법]**

Cox 시간 변화 모델을 사용하여 데이터 셋의 생존 예측 모델을 구축하였다. Cox 시간 변화 모델은 시간에 따른 위험과 개별적인 요소들 간의 관계를 모델링한다.

```Python
ctv = CoxTimeVaryingFitter(penalizer=0.1)
ctv.fit(df_new[train_cols], id_col="unit_nr", event_col='breakdown',
        start_col='start', stop_col='time_cycles', show_progress=True)
```

![cox_time]({{site.url}}/images/2023-10-15-NasaTurbofan/cox_time.png){: .align-center}

Cox 시간 변화 모델의 요약 정보를 출력하고 모델의 그래프를 시각화하였다. Cox 시간 변화 모델의 요약 정보가 담겨있다. Partial Log-likelihood는 모델의 적합도를 평가하는 중요한 지표 중 하나이다. 값이 높을수록 모델이 데이터를 얼마나 잘 설명하는 지를 나타낸다.

```Python
ctv.print_summary()
plt.figure(figsize=(10,5))
ctv.plot()
plt.show()
plt.close()
```

![cox_time_summary]({{site.url}}/images/2023-10-15-NasaTurbofan/cox_time_summary.png){: .align-center}

Cox Time Varying model의 요약 정보에 대한 설명:

1. Coef: 모델의 각 입력 변수(설명 변수)에 대한 계수(coefficients)를 나타낸다. 이 값들은 해당 변수가 생존 분포에 미치는 영향을 나타낸다.
2. exp(coef): 계수(coef)의 지수값은 위험 비(risk ratio)를 나타낸다. 이 값이 1보다 크면 해당 변수가 위험을 증가시키는 요인으로 작용하며, 작으면 위험을 감소시키는 요인으로 작용한다.
3. z: z-값은 각 계수에 대한 표준화된 값으로, 계수가 표준 오차에 비해 얼마나 큰지를 나타낸다. 큰 z-값은 변수가 중요하다는 것을 의미한다
4. p: p-값은 계수가 통계적으로 유의미한지 여부를 나타낸다. 작은 p-값은 해당 변수가 유의미하다는 것이다.

![cox_tim_summary1]({{site.url}}/images/2023-10-15-NasaTurbofan/cox_tim_summary1.png){: .align-center}

![cox_time_summary2]({{site.url}}/images/2023-10-15-NasaTurbofan/cox_tim2_summary2.png){: .align-center}

Cox 시간 변화 모델(“ctv”)를 사용하여 각 엔진에 대한 로그 부분 위험 값을 예측한다. 예측된 결과 값을 prediction 데이터 프레임에 “predictions”라는 변수로 저장한다. 이후 실제 RUL 값을 데이터 프레임에 추가한다.

```Python
# get the last unit time series data frame
df = df_new.groupby("unit_nr").last()
df = df[df['breakdown'] == 0]  # get engines from dataset which are still functioning so we can predict their RUL
df_to_predict = df.copy().reset_index()
# predictions = pd.DataFrame(ctv.predict_log_partial_hazard(df_to_predict[predict_cols]), index=df_to_predict.index)
predictions = ctv.predict_log_partial_hazard(df_to_predict[predict_cols]).to_frame()
predictions.rename(columns={0: "predictions"}, inplace=True)
df_last = train.groupby('unit_nr').last()
predictions['RUL'] = df_to_predict['RUL']
predictions.head(10)
```

```Python
plt.figure(figsize=(12,5))
plt.plot(predictions['RUL'], predictions['predictions'], '.b')
xlim = plt.gca().get_xlim()
plt.xlim(xlim[1], xlim[0])
plt.xlabel('RUL')
plt.ylabel('log_partial_hazard')
plt.show()
```

![prediction_rul]({{site.url}}/images/2023-10-15-NasaTurbofan/prediction_rul.png){: .align-center}

Cox 시간 변화 모델을 사용하여 전체 학습 데이터셋에 대한 로그 부분 위험 값을 예측하고, 이를 데이터프레임에 추가하는 부분을 설명한다. 이렇게 한 후 각 주기(기록)에 대한 예측된 위험 값을 포함하는 새로운 열('hazard')을 생성한다.

```Python
# now lets look at some hazard trajectories
X = df_new.loc[df_new['unit_nr'].isin(df_to_predict.index)]
X_unique = len(X['unit_nr'].unique())
plt.figure(figsize=(12,5))
for i in range(1, X_unique, 2):
    X_sub = X.loc[X['unit_nr'] == i]
    predictions = ctv.predict_partial_hazard(X_sub).values
    plt.plot(X_sub['time_cycles'].values, np.log(predictions))

plt.xlabel('time_cycles')
plt.ylabel('log_partial_hazard')
plt.show()
```

![prediction_rul-df]({{site.url}}/images/2023-10-15-NasaTurbofan/prediction_rul_df.png){: .align-center}

```Python
df_hazard = df_new.copy().reset_index()
df_hazard['hazard'] = ctv.predict_log_partial_hazard(df_hazard)
df_hazard.head()
```

모델의 mse 값과 exponential model 값을 구하기 위해 함수를 만든다

```Python
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))

def exponential_model(z, a, b):
    return a * np.exp(-b * z)

popt, pcov = curve_fit(exponential_model, df_hazard['hazard'], df_hazard['RUL'])
print(popt)
```

```Python
#Idea of fit: The line that was actually fitted is less accurate as it takes data points of all engines into account.
# check specific unit_nr
y_hat = exponential_model(df_hazard.loc[df_hazard['unit_nr']==1, 'hazard'], 70, 0.1)
plt.plot(df_hazard.loc[df_hazard['unit_nr']==1, 'hazard'], df_hazard.loc[df_hazard['unit_nr']==1, 'RUL'], 'o',
         df_hazard.loc[df_hazard['unit_nr']==1, 'hazard'], y_hat)
plt.xlabel("log_partial_hazard")
plt.ylabel("RUL")
plt.show()
plt.close()
```

**[테스트 데이터 셋에도 적용하기]**

테스트 데이터도 트레인 데이터 셋에 적용했던 것처럼 breakdown 칼럼을 만들고 dwt를 활용하여 신호처리하는 과정을 거친다.

```Python
# prep test set
test = test.drop(labels=drop_labels, axis=1)
test['breakdown'] = 0
test['start'] = test['time_cycles'] - 1
```

```Python
df_t = test[remaining_sensors]
```

```Python
dwt_list = []
for i in range(len(remaining_sensors)):
    dwt = lowpassfilter(df_t.iloc[:, i], 0.4)
    dwt_list.append(dwt)
```

```Python
df_test_dwt = pd.DataFrame(dwt_list)
```

```Python
df_test_dwt = df_test_dwt.T
```

```Python
df_test_dwt.columns = remaining_sensors
```

```Python
# predict and evaluate
y_hat = exponential_model(df_hazard['hazard'], *popt)
evaluate(df_hazard['RUL'], y_hat, 'train')

y_pred = ctv.predict_log_partial_hazard(df_test_new.groupby('unit_nr').last())
y_hat = exponential_model(y_pred, *popt)
evaluate(df_test, y_hat)
```

```Python
def mse_evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    print('{} set MSE:{}'.format(label, mse))
```

```Python
mse_evaluate(df_test, y_hat)
```

**[Cox Time-Varying Model을 활용하여 잔여 수명 예측 값 구하기]**

앞서 구한 로그 부분 위험 값(“hazard”)을 사용하여 지수성능 저하 모델을 학습하고 최적의 파라미터 값을 구한다. 이 모델을 사용하면 로그 부분 위험 값에 기반하여 잔여 사용 예측할 수 있다.
잔여 수명 예측 값을 구하기 위해서는 다음과 같은 과정을 거쳐야 한다.

1. Ctv 함수를 통해 로그 부분 위험 값을 구한다.
2. 구한 로그 부분 위험 값을 지수 저하 모델에 적용한다.
3. 지수 저하 모델로 통해 나온 예측 값과 실제 예측 값의 차이를 구한다

```Python
ctv2 = CoxTimeVaryingFitter()
ctv2.fit(df_new [train_cols], id_col="unit_nr", event_col='breakdown',
         start_col='start', stop_col='time_cycles', show_progress=True)

train['hazard'] = ctv2.predict_log_partial_hazard(train)
popt2, pcov2 = curve_fit(exponential_model, train['hazard'], train['RUL'])

y_hat = exponential_model(train['hazard'], *popt2)
evaluate(train['RUL'], y_hat, 'train')

y_pred = ctv2.predict_log_partial_hazard(test.groupby('unit_nr').last())
y_hat = exponential_model(y_pred, *popt2)
evaluate(df_test, y_hat)
```
