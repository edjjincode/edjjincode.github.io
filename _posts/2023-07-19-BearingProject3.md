---
layout: single
title: "[베어링 고장 진단 프로젝트#3/Isolation Forest, ABOD, Mahalanobis, MCD까지]"
categories: 이상감지 사이드프로젝트
tag:
  [베어링, 이상감지, 사이드 프로젝트, Isolation Forest, ABOD, Mahalanobis, MCD]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 베어링 고장 진단 프로젝트- 📣 이상감지

베어링의 고장을 감지할 때 머신 러닝 혹은 딥러닝을 활용한 이상감지 알고리즘이 활용될 수 있다. 다양한 이상감지 알고리즘들이 존재하나 이번 블로그에서는 Isolation Forest, ABOD, Mahalanobis, MCD, OC-SVM, SVDD, VAE, AnoGan에 대해 다룰 예정이다.

## 🌳 Isolation Forest를 이용한 Anomaly detection

### Isolation Forest 개념:

Isolation Forest는 이상치가 적고 다르다는 점에서 착안 된 개념이다. 의사결정 나무를 지속적으로 분기시키면서 모든 데이터 관측치의 고립 정도 여부에 따라 이상치를 판별한다.

분기 되는 깊이가 낮을 수록 이상치에 가깝고 깊이가 높을 수록 정상 데이터에 가깝다.

![isolationForest]({{site.url}}/images/2023-07-19-BearingProject/IsolationForest.png){: .align-center}

이상치는 분기되는 깊이가 얕다.

![isolationForest2]({{site.url}}/images/2023-07-19-BearingProject/IsolationForest2.png){: .align-center}

정상 데이터는 분기되는 깊이가 깊다.

### Isolation Forest 알고리즘:

1. 무작위 데이터가 주어질 때 데이터의 샘플이 이진 분류 트리에 할당된다.

2. 이진 분류는 랜덤한 피쳐를 선택하는 걸로 시작한다. 랜덤한 피쳐가 선택되면 선택된 피쳐에서 랜덤한 값을 선택하여 threshold로 선정한다.

3. 데이터 포인트가 threshold 보다 작을 경우 왼쪽 branch로 들어간다. threshold가 클 경우 오른쪽 branch로 들어간다.

4. 사용자가 지정해준 깊이나 데이터가 완전 괴립될 때까지 2번 과정을 반복한다.

이상치(anomaly)에 해당하는 값은 anomaly score -1을 갖게 되고 정상 데이터 같은 경우 1을 갖게 된다.

### Isolation Forest Python 코드:

```python

```

## 📐 ABOD를 활용한 이상감지

ABOD(Angle-based Outlier Detection)은 다변량 시계열 데이터에서 이상치를 감지할 때 유용하게 사용할 수 있는 머신러닝 기법이다.

### ABOD 개념:

다변량 데이터에서 임의의 3개의 데이터가 생성하는 각도를 면밀히 관찰하는 것이 ABOD의 핵심 개념이다. 해당 각도의 분산은 이상치와 정상치가 다르게 나타난다. 보통의 경우, 정상치의 각도 분산 값이 이상치에 비해 더 크게 나타난다 ABOD는 다른 머신러닝 이상탐지 기법과 달리, 고차원에서도 감지를 잘한다는 특징이 있다.

![ABOD.png]({{site.url}}/images/2023-07-19-BearingProject/ABOD.png)

### ABOD 알고리즘:

1. 각각의 데이터들에 대해서 조합을 생성한 후 해당 조합이 만들어내는 각도를 각도 리스트에 저장한다.
2. 해당 리스트에서 만들어진 각도의 분산 값을 구한다.
3. 특정 임계값 이하의 값은 이상치로 분류한다.

### 이상 감지를 위한 머신러닝 모듈 pyod:

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

### ABOD Python 코드(pyod를 활용한 코드):

```python

from pyod.models.abod import ABOD
clf = ABOD()
clf.fit(X_train)

#get outliers scores
y_train_scores = clf.decision_scores_#트레인 데이터에 대한 outlier score를 구한다
y_test_scores = clf.decision_function(X_test)#테스트 데이터에 대한 outlier score을 구한다.

```

### Fast ABOD 개념:

ABOD는 굉장히 좋은 이상감지 알고리즘이나 시간 복잡도가 O(n^3)이 될 만큼 시간이 많이 걸리는 알고리즘이다. 이를 개선한 것이 Fast ABOD이다. Fast ABOD는 모든 점들에 대해 가능한 모든 조합을 찾는 대신에 KNN(K-nearest neighbor)를 활용하여 분산 값을 추정한다.

## Mahalanobis 거리와 MCD를 이용한 이상감지

### Mahalanobis 거리

### [유클리드 거리]

데이터 분포에서 데이터 간의 거리를 구할 수 있는 방법은 다양하게 있다. 이 중에서 우리가 흔히 중, 고등학교 때 배웠던 거리를 구하는 방식은 유클리드 거리이다.

![uclidean.png]({{site.url}}/images/2023-07-19-BearingProject\유클리디안거리.png)

흔히 수학 시간에 배운 두 지점의 거리를 구하는 공식이다. 이를 벡터의 관점에서 생각을 해보면, 두 벡터의 내적으로 거리를 표현한 것이라고도 생각할 수 있다.

이러한 유클리드 거리의 단점은 맥락(혹은 관계?)을 파악하지 못한다는 점이다. 맥락을 이해하기 위해서는 마할라노비스 거리를 이해하면 된다.

### [마할라노비스 거리]

마할라노비스는 유클리드 거리가 고려하지 못하는 "맥락"을 고려하여 거리를 재는 방식이다. 여기서 맥락이란 무슨 뜻일까? 마할라노비스 거리에서 "맥락"은 데이터 분포를 의미한다. 즉, 다변량 데이터에서 분포의 형태를 고려하여 거리를 재겠다는 이론이다.

![uclidean_distance.png]({{site.url}}/images/2023-07-19-BearingProject\마할라노비스거리.png)

위 식에서 노란색으로 칠해진 부분은, 공분산 행렬로 변수들간의 Correlation을 고려하겠다는 것이다. 해당 식에서 분산이 1로 정규화 되고 변수들이 서로 독립일 경우, 유클리디안 거리와 같아진다.

마할라노비스 거리를 시각화 하면,

![uclidean_distance_graph.png]({{site.url}}/images/2023-07-19-BearingProject\마할라노비스거리_분포.png)

위 그림을 보면, 유클리디안 거리로 판단하였을 경우, y축에 있는 별이 x축에 있는 별보다 가까이 있는 것처럼 보이지만, 마할라노비스 거리에서는 x축에 있는 별이 y축에 있는 별보다 가까이 있다.

등고선을 고려했을 때, x 축에 있는 별이 y 축에 있는 별보다 등고선 상 가까이 위치하기 떄문이다.

### MCD

마할라노비스 거리는 변수들간의 공분산 값을 구하는 것이기 때문에 제곱을 취하게 된다. 하지만 제곱을 취할 시, outlier를 구하는 데 굉장히 취약해질 수 밖에 없다. 이를 개선하기 위한 방법이 바로 MCD이다.

### MCD Python 코드:

```python
from pyod.models.mcd import MCD

...

```

## OC-SVM(One-Class SVM)

OC-SVM을 이해하기 위새서는 기본적인 SVM 개념을 알아야한다. SVM 개념은 다음 블로그에 잘 나와 있으니 참고하도록 하자.

### SVM 개념:

[SVM 개념](https://losskatsu.github.io/machine-learning/svm/#%EB%84%88%EB%B9%84width-%EC%B5%9C%EB%8C%80%ED%99%94)

### OC-SVM 개념:

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

### oc-svm 파이썬 코드 Pyod

pyod는 oc-svm 기능 또한 제공한다.

```python
from from pyod.models.ocsvm import OCSVM

```

## DeepSVDD를 이용한 이상감지

해당 개념은 아래 링크를 참고하였습니다.

[DeepSVDD 개념](https://ffighting.net/deep-learning-paper-review/anomaly-detection/deep-svdd/)

### SVDD 개념:

SVDD는 정상 데이터들만을 포함하는 하나의 구를 찾아내고 그것에 벗어난 값들을 다 이상치로 판단하는 방법이다. 해당 구에서 많이 벗어나 있을수록 비정상 데이터라고 판단하는 것이다.

![SVDD]({{site.url}}/images/2023-07-19-BearingProject/svdd_문제점.png){: .align-center}

하지만 SVDD는 위 그림과 같은 상황일 때 한계가 있다. 위 그림에서 파란색 큰 원은 정상 데이터를 포한하는 구이다. 이때 X1, X2 두 점을 보면, X1 같은 경우 정상 데이터라고 할 수 있는 범위에 있으나, X2 같은 경우 정상 데이터라고 하긴 어렵다. 하지만 SVDD를 사용할 시 그대로 정상으로 판단하게 된다.

이를 보완하고자 하는 것이 Deep SVDD이다.

### DeepSVDD:

위 SVDD가 가지고 있는 한계에 대해 생각을 해보자, 정상 데이터 구에 포함되나 비교적 정상 데이터 클러스터와 멀리 있는 값 같은 경우를 다룰 수 없는 게 SVDD의 한계이다. 이를 해결하기 위해 고안한 방법이, CNN을 활용하여 넓게 분포된 데이터를 모아주는 방법이다.

![Deep_SVDD]({{site.url}}/images/2023-07-19-BearingProject/Deep_svdd.png){: .align-center}

CNN을 사용하여 정상 이미지 데이터를 저차원의 feature 공간으로 매핑할 것이다. 이때 매핑하는 방향은 모든 정상 이미지의 feature들이 한 점을 중심으로 모이도록 한다.

![Deep_SVDD_cnn]({{site.url}}/images/2023-07-19-BearingProject/deep_svdd_cnn.png){: .align-center}

CNN이 고차원의 공간을 저차원의 공간으로 매핑 해줄 수 있는 성질을 이용한 것입니다.

![Deep_SVDD_cnn_1]({{site.url}}/images/2023-07-19-BearingProject/cnn_deep.png){: .align-center}

### DeepSVDD 파이썬 코드 Pyod:

```python
from pyod.models.deep_svdd import DeepSVDD

...

```
