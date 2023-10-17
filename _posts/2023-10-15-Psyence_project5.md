---
layout: single
title: "[NASA Turbofan 프로젝트#5]/생존 모델"
categories: 잔여수명_예측
tag: [RUL]
toc: true
author_profile: false
sidebar:
nav: "docs"
---

### Survival Model(생존 모델)

**[데이터 전처리]**

생존 모델 같은 경우, 기존에 건전성 지표 모델과 유사성 모델과 달리 전처리하는 것부터 다시 시작하였다.

[사용한 라이브러리]

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

```python
!pip install lifelines
```

```python
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

```python
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

```python
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

```python
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

```python
# cut off is the censoring cycle time line
cut_off = 200
train_censored = train[train['time_cycles'] <= cut_off].copy()
print(train_censored.shape)
train_censored[train_censored["unit_nr"] == 50].tail()
```

daubechies wavelet 형태의 이산 웨이블릿 변환을 활용하여 높은 주파수를 가지고 있는 신호를 제거하는 과정을 거쳤다.

```python
def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal
```

```python
dwt_list = []
for i in range(len(remaining_sensors)):
    dwt = lowpassfilter(df.iloc[:, i], 0.4)
    dwt_list.append(dwt)
```

```python
df_dwt = pd.DataFrame(dwt_list)
```

```python
df_dwt = df_dwt.T
```

```python
df_dwt.columns = remaining_sensors
```

```python
df_1 = train_censored["unit_nr"]
```

```python
df = pd.concat([df_1, df_dwt], axis = 1)
```

```python
df_2 = train_censored[["time_cycles", "RUL"]]
```

```python
df_new = pd.concat([df, df_2], axis = 1)
```

```python
df_new.dropna(inplace = True)
```

“breakdown”이라는 새로운 열을 생성하였다. 해당 열은 각 기계가 고장을 일으켰는지 여부를 나타낸다. 각 기계의 마지막 사이클에서 고장을 일으키기 때문에 해당 주기의 데이터 값을 1로 설정한다.

“start”라는 새로운 열을 생성한 후 time_cycles에서 1을 뺀 값을 저장하였다.

이후 타임사이클이 200개 이하인 데이터만을 추출하였다.

```python
train_cols = index_names + remaining_sensors + ['start', 'breakdown']
predict_cols = ['time_cycles'] + remaining_sensors + ['start', 'breakdown']
```

**[Cox Time-Varying Model 방법]**

Cox 시간 변화 모델을 사용하여 데이터 셋의 생존 예측 모델을 구축하였다. Cox 시간 변화 모델은 시간에 따른 위험과 개별적인 요소들 간의 관계를 모델링한다.

```python
ctv = CoxTimeVaryingFitter(penalizer=0.1)
ctv.fit(df_new[train_cols], id_col="unit_nr", event_col='breakdown',
        start_col='start', stop_col='time_cycles', show_progress=True)
```

![cox_time]({{site.url}}/images/2023-10-15-NasaTurbofan/cox_time.png){: .align-center}

Cox 시간 변화 모델의 요약 정보를 출력하고 모델의 그래프를 시각화하였다. Cox 시간 변화 모델의 요약 정보가 담겨있다. Partial Log-likelihood는 모델의 적합도를 평가하는 중요한 지표 중 하나이다. 값이 높을수록 모델이 데이터를 얼마나 잘 설명하는 지를 나타낸다.

```python
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

```python
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

```python
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

```python
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

```python
df_hazard = df_new.copy().reset_index()
df_hazard['hazard'] = ctv.predict_log_partial_hazard(df_hazard)
df_hazard.head()
```

모델의 mse 값과 exponential model 값을 구하기 위해 함수를 만든다

```python
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

```python
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

```python
# prep test set
test = test.drop(labels=drop_labels, axis=1)
test['breakdown'] = 0
test['start'] = test['time_cycles'] - 1
```

```python
df_t = test[remaining_sensors]
```

```python
dwt_list = []
for i in range(len(remaining_sensors)):
    dwt = lowpassfilter(df_t.iloc[:, i], 0.4)
    dwt_list.append(dwt)
```

```python
df_test_dwt = pd.DataFrame(dwt_list)
```

```python
df_test_dwt = df_test_dwt.T
```

```python
df_test_dwt.columns = remaining_sensors
```

```python
# predict and evaluate
y_hat = exponential_model(df_hazard['hazard'], *popt)
evaluate(df_hazard['RUL'], y_hat, 'train')

y_pred = ctv.predict_log_partial_hazard(df_test_new.groupby('unit_nr').last())
y_hat = exponential_model(y_pred, *popt)
evaluate(df_test, y_hat)
```

```python
def mse_evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    print('{} set MSE:{}'.format(label, mse))
```

```python
mse_evaluate(df_test, y_hat)
```

**[Cox Time-Varying Model을 활용하여 잔여 수명 예측 값 구하기]**

앞서 구한 로그 부분 위험 값(“hazard”)을 사용하여 지수성능 저하 모델을 학습하고 최적의 파라미터 값을 구한다. 이 모델을 사용하면 로그 부분 위험 값에 기반하여 잔여 사용 예측할 수 있다.
잔여 수명 예측 값을 구하기 위해서는 다음과 같은 과정을 거쳐야 한다.

1. Ctv 함수를 통해 로그 부분 위험 값을 구한다.
2. 구한 로그 부분 위험 값을 지수 저하 모델에 적용한다.
3. 지수 저하 모델로 통해 나온 예측 값과 실제 예측 값의 차이를 구한다

```python
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
