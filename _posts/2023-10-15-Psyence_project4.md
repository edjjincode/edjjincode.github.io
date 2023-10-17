---
layout: single
title: "[NASA Turbofan 프로젝트#4]/건전성 지표 모델"
categories: 사이드프로젝트
tag: [RUL]
toc: true
author_profile: false
sidebar:
nav: "docs"
---

### 건전성 지표 모델(Health Indicator Model)

건전성 지표에 사용되는 데이터를 추출하는 과정은 앞 과정에서 다뤘으므로 생략하고 바로 피쳐 선택(Feature Selection) 단계를 진행하겠다.

**[단조 특성을 활용한 Feature Selection]**

각 센서들의 단조 특성 값을 구할 수 있는 함수를 설정한 뒤 각 센서들의 단조 특성 값을 구하였다.

```python
def monotonicity(data):

    num_pos = data[data > 0].shape[0]
    num_neg = data[data < 0].shape[0]
    tot_n = data.shape[0] - 1

    mon_val = np.abs(num_pos - num_neg)/tot_n
    return mon_val
```

```python
mon_df = pd.DataFrame(columns = ['feature', 'monotonicity_val'])

for col in sensor_cols:
    mon_val = []
    for unit in df_lag.UnitNumber.unique():
        mon_val.append(monotonicity(df_lag.loc[df_lag.UnitNumber == unit, col]))
    mon_df = mon_df.append({'feature': col, 'monotonicity_val': np.mean(mon_val)}, ignore_index = True)
```

```python
mon_df = mon_df.sort_values(by = 'monotonicity_val', ascending = False)
mon_df.head()
```

```python
fig, ax = plt.subplots(figsize = (7,10))

sns.barplot(y = mon_df.feature, x = mon_df.monotonicity_val)
```

```python
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

```python
from sklearn.decomposition import PCA
```

```python
pca = PCA(n_components=3)

pca_data = pca.fit_transform(df_train_mean[feats])

pca_df = pd.DataFrame(pca_data, columns = ['pc1', 'pc2', 'pc3'])
pca_df['UnitNumber'] = df_train_mean.UnitNumber.values
pca_df['cycle'] = pca_df.groupby('UnitNumber').cumcount()+1
pca_df['RUL'] = pca_df.groupby('UnitNumber').cycle.transform('max') - pca_df.cycle
pca_df.head()
```

![pc_graph]({{site.url}}/images/2023-10-15-NasaTurbofan/pc_graph.png){: .align-center}

```python
pcs = ['pc1', 'pc2', 'pc3']
```

처음 두 개의 주성분으로 구성된 공간에서 데이터를 시각화 하였다.

```python
sns.pairplot(data = pca_df[pca_df.UnitNumber == 1], x_vars= pcs, y_vars = pcs)
```

```python
sns.scatterplot(data = pca_df[pca_df.UnitNumber == 1], x = "pc1", y = "pc2", hue = "RUL")
```

![pc_scatterplot]({{site.url}}/images/2023-10-15-NasaTurbofan/pc_scatterplot.png){: .align-center}

```python
fig, ax = plt.subplots()
sns.lineplot(data = pca_df[pca_df.UnitNumber == 1], x = "cycle", y = "pc1", ax = ax)
plt.axhline(pca_df[pca_df.UnitNumber == 1].pc1.max(), color = 'r')
ax.set_ylabel("Health Indicator")
```

다음은 기계 고유 번호가 1인 기계의 주성분 값을 시간에 따라 나타낸 것이다. 파란색 선은 주성분 값을 나타내고 빨간색 선은 주성분의 최대값을 의미한다. 주성분과 시간이 지수 관계에 있음을 확인할 수 있다.

![pc_lineplot]({{site.url}}/images/2023-10-15-NasaTurbofan/pc_lineplot.png){: .align-center}

다음은 pc1 값을 주성분으로 했을 때 RUL(잔여 수명) 값을 도식화 한 것이다. RUL 값이 0일 때 확실하게 분류가 되는 것을 확인할 수 있다.

```python
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

```python
threshold = pca_df.pc1[pca_df.RUL == 0].mean()
threshold
```

```python
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

```python
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

```python
def residuals(parameters, data, y_observed, func):
    '''
    Compute residuals of y_predicted - y_observed
    where:
    y_predicted = func(parameters,x_data)
    '''
    return func(parameters, data) - y_observed
```

```python
param_0 = [-1, 0.01, 0.01]
```

```python
exp_params_df = pd.DataFrame(columns = ['UnitNumber', 'phi', 'theta', 'beta'])

for i in range(1,101):

    ht = pca_df.pc1[pca_df.UnitNumber == i]
    cycle = pca_df.cycle[pca_df.UnitNumber == i]

    OptimizeResult = optimize.least_squares(residuals, param_0, args = (cycle, ht, exp_degradation))
    phi, theta, beta = OptimizeResult.x

    exp_params_df = exp_params_df.append({'UnitNumber':i, 'phi': phi, 'theta': theta, 'beta': beta}, ignore_index = True)
```

```python
exp_params_df.head()
```

![exp_params_df]({{site.url}}/images/2023-10-15-NasaTurbofan/exp_params_df.png){: .align-center}

생성한 지수 성능 저하 모델이 잘나오는 지 확인하기 위해 1번 기계의 주성분 값과 지수 성능 저하 모델을 시각화하였다. 시각화 결과, 지수 성능 저하 모델이 잘 피팅이 되는 것을 알 수 있다.

```python
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

```python
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

```python
fig, ax = plt.subplots(figsize = (10,3), nrows = 1, ncols = 3)
sns.distplot(exp_params_df.phi, ax = ax[0])
sns.distplot(exp_params_df.theta, ax = ax[1], color = "red")
sns.distplot(exp_params_df.beta, ax = ax[2], color = "green")
```

![exp_params_dist]({{site.url}}/images/2023-10-15-NasaTurbofan/exp_params_dist.png){: .align-center}

**[테스트 데이터 셋 적용]**

테스트 데이터 셋에 대해서도 PCA를 진행하고 PC1을 건전 지표로 설정하였다.

```python
window = 5

df_test_mean = df_test.groupby('UnitNumber')[feats].rolling(window = window).mean()
df_test_mean = df_test_mean.reset_index()
df_test_mean.dropna(inplace = True)
df_test_mean.drop(['level_1'], axis = 1, inplace = True)
df_test_mean.head()
```

테스트 데이터 셋에 대해서도 PCA를 진행하고 PC1을 건전 지표로 설정하였다.

```python
pca_test_data = pca.transform(df_test_mean[feats])

pca_test_df = pd.DataFrame(pca_test_data, columns = ['pc1', 'pc2', 'pc3'])
pca_test_df['UnitNumber'] = df_test_mean.UnitNumber.values
pca_test_df['cycle'] = pca_test_df.groupby('UnitNumber').cumcount()+1
pca_test_df.head()
```

![pc1_threshold]({{site.url}}/images/2023-10-15-NasaTurbofan/pc1_threshold.png){: .align-center}

앞서 지수 성능모델을 만드는 과정을 통해 생성한 파라미터들에 대해 백분위수를 활용하여 경계(bound)를 정의하였다. 하한선을 25%, 상한선을 75%로 설정하여 백분위수 범위를 설정하였다.

```python
phi_vals = exp_params_df.phi
theta_vals = exp_params_df.theta
beta_vals = exp_params_df.beta
```

```python
phi_vals.mean()
```

```python
param_1 = [phi_vals.mean(), theta_vals.mean(), beta_vals.mean()]
param_1
```

```python
lb = 25
ub = 75
phi_bounds = [np.percentile(phi_vals, lb), np.percentile(phi_vals, ub)]
theta_bounds = [np.percentile(theta_vals, lb), np.percentile(theta_vals, ub)]
beta_bounds = [np.percentile(beta_vals, lb), np.percentile(beta_vals, ub)]
```

```python
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

```python
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

```python
result_test_df.head()
```

![pred_rul1]({{site.url}}/images/2023-10-15-NasaTurbofan/pred_rul1.png){: .align-center}

```python
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

```python
mean_squared_error(result_test_df.True_RUL, result_test_df.Pred_RUL)
```

```python
mean_absolute_error(result_test_df.True_RUL, result_test_df.Pred_RUL)
```

```python
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

```python
mean_absolute_percentage_error(result_test_df.True_RUL, result_test_df.Pred_RUL)
```
