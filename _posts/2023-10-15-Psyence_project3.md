---
layout: single
title: "[NASA Turbofan 프로젝트#3]/Similarity Model"
categories: 사이드프로젝트
tag: [RUL]
toc: true
author_profile: false
sidebar:
nav: "docs"
---

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

```python
def trendability(col):
    trend_vals = []
    for i in df_train.UnitNumber.unique():
        dff = df_train[df_train.UnitNumber == i]
        trend_vals.append(dff[['Cycle', col]].corr().iloc[0][1])
    return np.abs(np.mean(trend_vals))
```

시계열과의 상관성을 구할 수 있는 함수를 만들었다. trendability value 값이 클 수 록 시계열과 상관성이 높은 feature라고 할 수 있다.

```python
trend_df_list = []
for col in sensor_cols:
    trend_df_list.append({'feature': col, 'trendability_val': trendability(col)})
trend_df = pd.DataFrame(trend_df_list, columns = ['feature', 'trendability_val'])

```

각 센서별로 trenability value를 구한 후 barplot을 구하였다.

```python
fig, ax = plt.subplots(figsize = (7,10))
sns.barplot(y = trend_df.feature, x = trend_df.trendability_val)
```

![trendability_barplot]({{site.url}}/images/2023-10-15-NasaTurbofan/barplot.png){: .align-center}

이 중 trendability value 값이 0.2보다 큰 값을 선택하였다.

![trendability]({{site.url}}/images/2023-10-15-NasaTurbofan/trendability.png){: .align-center}

**[DWT를 활용한 신호처리]**

앞서 언급했듯이, 각 센서 데이터는 서로 영향을 주기 때문에 센서에 노이즈가 껴져 있다. 따라서 이를 신호처리를 활용하여 노이즈를 제거하는 과정을 거쳐야 한다.

```python
df = df_train[feats]
```

Daubechies Wavelet을 사용하여 높은 주파수를 가진 파형을 처리하는 과정을 거쳤다.Threshold 값은 0.63으로 두었다.

```python
def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal
```

만든 lowpassfilter의 출력 값을 dwt_list에 저장하는 과정을 거쳤다.

```python
dwt_list = []
for i in range(len(feats)):
    dwt = lowpassfilter(df.iloc[:, i], 0.4)
    dwt_list.append(dwt)
```

dwt_list를 데이터 프레임화 시키는 과정을 거쳤다.

```python
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

```python
df_new['HI'] = df_new.groupby('UnitNumber').RUL.transform(lambda x: minmax_scale(x))
```

![HI]({{site.url}}/images/2023-10-15-NasaTurbofan/HI.png){: .align-center}

```python
sns.lineplot(data= df_new[df_new.UnitNumber < 31], x = 'Cycle', y = 'HI', hue= 'UnitNumber')
```

![lineplot]({{site.url}}/images/2023-10-15-NasaTurbofan/lineplot.png){: .align-center}

각 기계들의 고유 번호에 대해 lineplot으로 시각화하면 선형적 열화 모델을 따른다는 것을 알 수 있다.

전체 센서 값들을 대표할 수 있는 하나의 열화 모델을 만들기 위해 추출된 특성을 독립변수로 놓고 “Hi” 값을 종속변수로 놓고 선형 회귀를 돌렸다. 해당 회귀 모델의 coefficient 값은 다음과 같다.

![HI_coefficient]({{site.url}}/images/2023-10-15-NasaTurbofan/HI_coefficient.png){: .align-center}

```python
model = LinearRegression()
```

```python
X = df_new[feats]
y = df_new.HI
X.shape, y.shape
```

```python
model.fit(X, y)
model.score(X,y)
```

```python
model.coef_
```

구한 선형 회귀 모델의 coefficient 값을 센서들의 데이터와 내적하여 모든 데이터를 대표할 수 있는 선형 열화 지수 “Hi_final”을 만든다.

```python
df_new["HI_final"] = df_new[feats].dot(model.coef_)
df_new.HI_final.head()
```

![HI_final]({{site.url}}/images/2023-10-15-NasaTurbofan/HI_final.png){: .align-center}

![HI_final_1]({{site.url}}/images/2023-10-15-NasaTurbofan/HI_final_1.png){: .align-center}

다음은 x 값을 Cycle로 놓고 y 값을 HI_final로 놓은 후 시각화 한 그래프이다.

[이동 평균을 활용한 그래프]

```python
sns.lineplot(data= df_new[df_new.UnitNumber < 31], x = 'Cycle', y = 'HI_final', hue= 'UnitNumber')
plt.ylabel('Health Indicator')
```

![이동평균_그래프]({{site.url}}/images/2023-10-15-NasaTurbofan/이동평균그래프.png){: .align-center}

```python
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

```python
params_list = []
for i in range(1,101):
    y = df_new.HI_final[df_new.UnitNumber == i]
    cycle = df_new.Cycle[df_new.UnitNumber == i]
    theta_2, theta_1, theta_0 = np.polyfit(cycle, y, 2)
    params_list.append({'UnitNumber':i, 'theta_0': theta_0, 'theta_1': theta_1, 'theta_2': theta_2})
params_df = pd.DataFrame(params_list, columns = ['UnitNumber', 'theta_2', 'theta_1', 'theta_0'])
```

```python
params_df.head()
```

[이동평균법-2차 곡선 파이썬 코드]

```python
HI = df_new.HI_final[df_new.UnitNumber == 1]
cycle = df_new.Cycle[df_new.UnitNumber == 1]
theta_0 = params_df.theta_0[params_df.UnitNumber == 1].values
theta_1 = params_df.theta_1[params_df.UnitNumber == 1].values
theta_2 = params_df.theta_2[params_df.UnitNumber == 1].values
HI_fit = theta_0 + theta_1*cycle + theta_2*cycle*cycle
```

```python
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

```python
dwt_test_list = []
for i in range(len(feats)):
    dwt = lowpassfilter(df_test[feats].iloc[:, i], 0.4)
    dwt_test_list.append(dwt)
```

테스트 데이터 셋에 대해서도 dwt를 활용하여 신호처리를 해 데이터를 전처리한다.

```python
df_test_dwt = pd.DataFrame(dwt_test_list)
```

```python
df_test_dwt  = df_test_dwt.T
```

```python
df_test_dwt.columns = feats
```

훈련 데이터 셋에서 생성한 계수를 그대로 테스트 데이터 셋에 사용해 “HI”(Hazard Index) 변수를 생성하였다.

```python
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

```python
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

```python
df_test_fit.head()
```

![df_test_fit]({{site.url}}/images/2023-10-15-NasaTurbofan/df_test_fit.png){: .align-center}

df_test_fit 데이터 프레임에서 각 기계(UnitNumber) 마다 유사성 점수가 5번째로 높은 것까지 추출하여 result_df_5라는 새로운 데이터 프레임을 만들었다.

```python
ind_5 = df_test_fit.groupby('UnitNumber')['similarity'].nlargest(5).reset_index()['level_1']
result_df_5 = df_test_fit.iloc[ind_5]
result_df_5.head()
```

![result_df_5]({{site.url}}/images/2023-10-15-NasaTurbofan/result_df_5.png){: .align-center}

이후 result_df_5에서 구한 각 기계 별 total_life의 평균과 테스트 데이터의 기계 별 사이클의 평균의 차이를 통해 잔여 수명 예측 값을 구한다.

```python
y_true_5 = y_true.copy()

y_true_5['Pred_RUL'] = (result_df_5.groupby('UnitNumber')['total_life'].mean() - df_test.groupby('UnitNumber')['Cycle'].max()).values
y_true_5.head()
```

![pred_rul]({{site.url}}/images/2023-10-15-NasaTurbofan/pred_rul.png){: .align-center}

```python
sns.regplot(x = y_true_5.Pred_RUL, y = y_true_5.RUL)
plt.xlabel('Predicted RUL')
plt.ylabel('True RUL')
```

```python
fig, ax = plt.subplots(figsize = (15, 7))
sns.lineplot(x = y_true_5.UnitNumber, y = y_true_5.Pred_RUL, label = "Predicted RUL")
sns.lineplot(x = y_true_5.UnitNumber, y = y_true_5.RUL, label = "True RUL")
plt.xlabel("Unit Number")
plt.ylabel("Remaining Useful Life")
plt.legend(loc = 1)
```

```python
mean_squared_error(y_true_5.RUL, y_true_5.Pred_RUL)
```

```python
from sklearn.metrics import mean_squared_error, r2_score
```

```python
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))
```

```python
evaluate(y_true_5.RUL, y_true_5.Pred_RUL)
```

```python
mean_absolute_error(y_true_5.RUL, y_true_5.Pred_RUL)
```
