---
layout: single
title: "[NASA Turbofan 프로젝트#2]/EDA"
categories: 사이드프로젝트
tag: [RUL]
toc: true
author_profile: false
sidebar:
nav: "docs"
---

## 데이터 셋 설명:

본 분석에 사용되는 데이터 셋은 Nasa에서 제공하는 CMAPss 데이터 셋이다. CMAPss 데이터 셋은 총 4개의 데이터 셋으로 이루어지고 그중 본 분석에 사용된 데이터 셋은 FD001 데이터 셋이다.

![데이터셋_설명]({{site.url}}/images/2023-10-15-NasaTurbofan/데이터셋_설명.png){: .align-center}

각 데이터 셋은 훈련, 테스트, RUL 세트로 나뉜다. 각 엔진은 초기 마모 및 제조 변동 정도가 다른 상태에서 시작한다. 엔진은 정상적으로 작동하다 어느 시점에서 고장이 발생한다. 훈련 셋에서는 시스템 고장이 발생할 때까지 선형적으로 고장 정도가 심해진다. 테스트 셋에서는 시계열이 시스템 고장보다 어느 정도 먼저 종료된다. 해당 분석 목적은 테스트 세트에서 고장이 발생하기 전 사이클 수, 즉 엔진이 계속 작동할 마지막 사이클 이후의 작동 사이클 수를 예측하는 것이다.

FD001 데이터 셋은 100개의 엔진을 작동시켜 21개의 센서 값을 통해 데이터를 측정하였다. 데이터 셋은 26개의 속성이 있고 엔진 고유 번호, 사이클 수(시간), 세가지 세팅 상태, 21개의 센서 값으로 이루어져 있다. 인스턴스는 훈련 데이터 셋 기준으로 20631개 있다.

[FD001 훈련/테스트 데이터 셋 Attribute]

UnitNumber: 엔진 고유 번호
Cycle: 사이클 수
OptSet1~3: 3가지 설정(해당 분석에는 활용하지 않을 예정이다)
Sensor1~21: 센서 값
RUL: 잔여 수명

[FD001 RUL 데이터 셋 Attribute]

RUL: 잔여 수명
UnitNumber: 고유 기계 번호

## 데이터 셋 가공

[사용한 라이브러리들에 대한 코드]

```python

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

```python
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

```python
index_names = ['UnitNumber', 'Cycle']
df_train[index_names].describe()
```

![description1]({{site.url}}/images/2023-10-15-NasaTurbofan/eda_decription.png){: .align-center}

- Count 값을 확인하면 데이터 집합의 행 수는 총 20631개라는 것을 알 수 있다.
- Min-Max 값을 통해 기계는 1에서 100개까지 있다는 것을 알 수 있다.
- 평균과 분위수가 기술 통계량과 깔끔하게 일치하지 않는 것을 확인 할 수 있다. 이는 각 기계마다 Max Cycle이 다르기 때문이다.

[고유 기계 별 사이클 수 기술 통계량]

고유 기계 별 사이클 수를 확인하는 과정을 거쳤다.

```python
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

```python
sc = MinMaxScaler(feature_range=(0,1))
df_train[sensor_cols] = sc.fit_transform(df_train[sensor_cols])
df_test[sensor_cols] = sc.transform(df_test[sensor_cols])
```

```python
df_train.groupby("UnitNumber").size().plot(kind="hist")
plt.title("Number of samples per engine")
plt.show()
```

![histogram]({{site.url}}/images/2023-10-15-NasaTurbofan/histogram.png){: .align-center}

RUL 데이터 셋을 확인하면 좌측으로 skewed 되어 있다는 것을 알 수 있다.
