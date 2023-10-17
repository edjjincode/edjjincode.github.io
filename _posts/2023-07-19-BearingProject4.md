---
layout: single
title: "[베어링 고장 진단 프로젝트#4/딥러닝을 활용한 이상감지]"
categories: 이상감지, 사이드프로젝트
tag: [베어링, 이상감지, 사이드 프로젝트, VAE, AnoGan, DCGAN]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

## VAE를 활용한 이상감지

### VAE와 AE의 차이점:

VAE는 Variational Auto-Encoder의 약자로 이름만 봐서는 Auto-Encoder와 비슷한 개념이라 생각이 들지만 전혀 다른 개념이다.

다음 링크를 참고하였다. [VAE\_블로그](https://chickencat-jjanga.tistory.com/3)

AE:

![AE]({{site.url}}/images/2023-07-19-BearingProject/vae-autoencoder.png){: .align-center}

Auto Encoder의 목적은 Encoder에 있다. AE는 Encoder의 학습을 위해 Decoder를 붙인 것이다.

latent vector가 어떤 하나의 값을 가지게 된다.

VAE:

![VAE]({{site.url}}/images/2023-07-19-BearingProject/vae_1.png){: .align-center}

Variational AutoEncoder의 목적은 Decoder에 있다. Decoder의 학습을 위해 Encoder를 붙인 것이다.

추출된 latent vector의 값을 하나의 숫자로 나타내는 것이 아니라 가우시안 확률 분포에 기반한 확률값으로 나타내게 된다.

![VAE_2]({{site.url}}/images/2023-07-19-BearingProject/vae_2.png){: .align-center}

### VAE의 개념:

VAE는 Input image X를 잘 설명하는 feature를 추출하여 Latent vector z에 담고, 이 Latent vector z를 통해 X와 유사하지만 완전히 새로운 데이터를 생성해내는 것을 목표로 한다.

✔ 이때 추출되는 feature는 가우시안 분포를 따른다고 가정한다

✔ Latent vector z는 각 feature의 평균과 분산값을 나타낸다

✔ AE의 decoder처럼 latent vector로부터 이미지를 생성해낸다고 보면 된다. 하지만 AE와 다른 점은 latent vector에 있는 값이 값이 아니라 확률 값이라는 점이다.

예를 들어, 시츄 얼굴을 그리고자 한다면, 시츄의 눈, 코, 입 등의 feature를 평균 및 분산 형태로 Latent vector z에 담고, 그 z를 이용해 시츄의 얼굴을 그리게 된다.

VAE의 구조를 도식화 하면 다음과 같다.

![VAE_3]({{site.url}}/images/2023-07-19-BearingProject/vae_구조.png){: .align-center}

VAE는 Input Image가 들어오면, 그 이미지에서의 다양한 특징들이 각각의 확률 변수가 되는 어떤 확률 분포를 만들게 된다. 이런 확률 중에서 확률값이 높은 부분을 이용하면 실제에 있을법한 이미지를 새롭게 만들 수 있다.

![VAE_4]({{site.url}}/images/2023-07-19-BearingProject/vae_확률도식.png){: .align-center}

### VAE를 이용한 이상감지:

자, 이제 VAE의 개념을 알아봤으니, VAE가 이상감지에 어떻게 사용되는지 알아봐야 한다. VAE는 AE가 이상감지에 사용되는 방식과 동일한 원리로 사용된다.

우선 AE를 활용한 이상감지에 원리를 살펴보자. Auto Encoder는 Encoder와 Decoder로 나눠져있다. Encoder 같은 경우, 입력 값의 특징을 값 형태로 변환한다. 변환된 값을 통해 Decoder로 복원을 한다. 정상치가 Encoder에 들어가 생기는 값과 이상치가 Encoder에 들어가 생기는 값이 다르기 때문에 복원시 정상치와 이상치가 다르게 나타나게 된다. AE를 이상감지에 활용할 때는 이러한 방법을 사용한다.

VAE도 같은 원리로 이상감지에 활용된다. 정상치와 이상치가 Encoder에 들어가면 latent vector에 서로 다른 형태의 확률 분포를 갖게되고 이를 다시 Decoder를 통해 재생성하면 서로 다른 결과 값을 만들게 된다. 위와 같은 원리를 활용하여 VAE를 감지할 수 있다.

### VAE를 이용한 이상감지 python 코드:

1. pyod에서 제공하는 모듈 활용:

```python
from pyod.models.thresholds import VAE

```

2. python을 활용한 VAE(MIST 데이터 셋):

```python
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras import Model, layers
from matplotlib import pyplot as plt
```

```python
dataset = tfds.load('mnist', split = 'train')
```

데이터를 로드 한다.

```python
batch_size = 1024
train_data = dataset.map(lambda data: tf.cast(data['image'], tf.float32)/255.).batch(batch_size)
```

데이터를 전처리 해준다. 여기서 tf.cast 함수는 텐서의 자료형을 바꾸는 함수이다.

**<Encoder>**

```python
class Vanila_Encoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(512, activation = "relu")
            layers.Dense(256, activation = "relu"),
            layers.Dense(latent_dim * 2)
        ])

    def __call__(self, x):
        #self.encoder(x)가 도출한 각각의 값을 mu, logvar로 mapping
        #tf.split: value into a list of subentsors
        mu, logvar = tf.split(self.encoder(x), 2, axis = 1)
        return mu, logvar
```

**<Decoder>**

```python
class Vanila_Decoder(Model):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation = "relu"),
            layers.Dense(512, activation = "relu"),
            layers.Dense(784, activation = "sigmoid"),
            layers.Reshape((28, 28, 1))
        ])

    def __call__(self, z):
        return self.decoder(z)
```

### Sampling 및 train_step 구현

```python

def sample(mu, logvar):
    epsilon = tf.random.normal(mu.shape)
    sigma = tf.exp(0.5* logvar)
    return epsilon*sigma + mu
```

**<Train VAE>**

```python

def train_step(inputs):
    #Graient Tape에서 gradient 값들을 수집함
    with tf.GradientTape() as tape:
        # Encoder로부터 mu, logvar를 얻음: q(z/x)
        mu, logvar = encoder(inputs)
        #mu, logvar를 사용해서 reparameterization trick 생성
        z = sample(mu, logvar)
        # rparameterization tick을 Decoder에 넣어 reconstruct x를 얻는다.
        x_recon = decoder(z)
        #입력과 생성된 이미지의 차이
        reconstruction_error = tf.reduce_sum(tf.losses.binary_crossentopy(inputs, x_recon))
        #KL을 구한다
        kl = 0.5*tf.reduce_sum(tf.exp(logvar) + tf.square(mu) - 1. - logvar)
        # inputs.shape[0] # of sapmples
        loss = (kl + reconstruction_error) / inputs.shape[0]
        #get trainable parameter
        vars_ = encoder.trainable_variables + decoder.trainable_variables
        # get grads
        grads_ = tape.gradient(loss, vars_)
        # apply gradient descent
        optimizer.apply_gradients(zip(grads_, vars_))
    return loss, reconstruction_error, kl
```

### 모델 구성 및 학습

```python
# Set hyperparameters

n_epochs = 50
latent_dim = 2
learning_rate = 1e-3
log_interval = 10

encoder = Vanila_Encoder(latent_dim)
decoder = Vanila_Decoder(latent_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
```

```python

for epoch in range(1, n_epochs + 1):
    total_loss, total_recon, total_kl = 0, 0, 0
    for x in train_data:
        loss, recon, kl = train_step(x)
        #loss 저장
        total_loss += loss *x.shape[0]
        #error 저장
        total_recon += recon
        #total KL 저장
        total_kl += kl

    if epoch % log_interval == 0:
        print(
            f'{epoch:3d} iteration: ELBO {total_loss / len(dataset):.2f}, ' \
            f'Recon {total_recon / len(dataset):.2f}, ' \
            f'KL {total_kl / len(dataset):.2f}'
        )
```

## AnoGan을 활용한 이상감지

### AnoGan 개념:

해당 포스트는 [AnoGAN\_포스트](https://ffighting.net/deep-learning-paper-review/anomaly-detection/anogan/)를 참고하였다.

AnoGan은 이상 감지를 위해 Gan을 사용하는 모델으로 GAN과 유사하나 학습할 때 정상 이미지만 학습한다는 점에서 GAN과 다르다.

**[AnoGan 모델 Summary]**

![AnoGan]({{site.url}}/images/2023-07-19-BearingProject/anogan.png){: .align-center}

AnoGan에서는 학습 단계에서 정상 이미지만 학습을 한다. 이렇게 학습시, Generator는 정상 이미지에 대해서만 학습을 했기 때문에 생성할 수 있는 distribution 또한 정상 이미지와 유사한 distribution이다. 이런식으로 Generator가 정상 distribution만 생성할 수 있도록 학습을 시킨 상황에서 이상 이미지가 Generator가 들어가게 되면, 생성된 이미지와 테스트 이미지 사이의 차이가 크게 될 것이다.이런 원리를 활용하여 이상치를 감지하게 된다.

![AnoGan1]({{site.url}}/images/2023-07-19-BearingProject/anogan_1.png){: .align-center}

**[AnoGan train 단계]**

![AnoGan_train]({{site.url}}/images/2023-07-19-BearingProject/anogan_train.png){: .align-center}

AnoGan은 정상 데이터를 학습하게 된다. Generator는 latent vector z로부터 이미지 G(z)를 만들어내는 모듈이다. 학습 단계에서는 생성된 G(z)를 Discriminator가 정상 이미지인지, Generator가 만들어낸 이미지인지 구분하도록 학습을 하게 된다.

![AnoGan_train]({{site.url}}/images/2023-07-19-BearingProject/anogan_train1.png){: .align-center}

**[Inference 단계]**

![AnoGan_inference]({{site.url}}/images/2023-07-19-BearingProject/anogan_inference.png){: .align-center}

AnoGan에서는 단순히 정상 이미지를 생성하는 네트워크를 사용해서는 불량 유무를 판단할 수 없다. Generator가 입력으로 받은 이미지와 가장 유사한 이미지를 갖도록 추론 단계를 거쳐야 한다.

이를 위해서는 Generator가 생성한 이미지인 G(z)가 x와 유사해지도록 하는 z를 찾아야 한다. G(z)와 x의 차이가 최소가 되도록 loss function을 설계한 후 모델을 학습할 수 있도록 해야 한다.

**이때 사용되는 loss function은 두 가지이다.**

**첫번째는 입력 이미지(x)와 생성된 이미지가 image level에서 같아지도록 하는 loss function이다.**

![AnoGan_l1]({{site.url}}/images/2023-07-19-BearingProject/anogan_l1.png){: .align-center}

**두번째 loss는 생성한 이미지와 입력받은 이미지가 Discriminator의 feature level에서 같아지도록 유도해주는 loss이다.**

![AnoGan_l2]({{site.url}}/images/2023-07-19-BearingProject/anogan_l2.png){: .align-center}

**최종 loss는 두가지 loss로 구성해준다.**

![AnoGan_l3]({{site.url}}/images/2023-07-19-BearingProject/anogan_l3.png){: .align-center}

해당 최종 loss는 anomaly score로도 사용된다.

#### AnoGan 코드:

##### **[DCGAN + AE (CIFAR10 데이터)]**

##### **1.기본 라이브러리, 함수 Import**

```python
!pip install tensorboardX
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import skimage
import skimage.io

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False

writer = SummaryWriter(logdir='runs/DCGAN_training_5')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
```

##### **2.GAN 훈련 부분**

##### **1.Model Architecture**

##### **1-1.Generator(latent z를 통해 이미지 생성)**

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf = 28, channels=1, bias = True):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*8, kernel_size=4, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
```

##### **1-2.Discriminator(이미지가 Real인지 Fake인지 구별)**

```python
class Discriminator(nn.Module):
    def __init__(self, ndf=28, channels=1, bias=True):
        super().__init__()

        def discriminator_block(in_features, out_features, bn=True):
            if bn:
                block = [
                         nn.Conv2d(in_features, out_features, 4, 2, 1, bias=bias),
                         nn.BatchNorm2d(out_features),
                         nn.LeakyReLU(0.2, inplace=True) # Generator에 미분값을 더 잘 전달헤주기 위해 LeakyReLU 사용.
                ]
            else:
                block = [
                         nn.Conv2d(in_features, out_features, 4, 2, 1, bias=bias),
                         nn.LeakyReLU(0.2, inplace=True)
                ]
            return block

        self.features = nn.Sequential(
            *discriminator_block(channels, ndf, bn=False),
            *discriminator_block(ndf, ndf*2, bn=True),
            *discriminator_block(ndf*2, ndf*4, bn=True),
            *discriminator_block(ndf*4, ndf*8, bn=True)
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=bias),
            nn.Sigmoid()
        )

    def forward_features(self, x):
        features = self.features(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        validity = self.last_layer(features)
        return validity
```

##### **1-3.Encoder(latent z 생성)**

```python
class Encoder(nn.Module):
    def __init__(self, latent_dim=100, ndf=28, channels=1, bias=True):
        super().__init__()

        def encoder_block(in_features, out_features, bn=True):
            if bn:
                block = [
                         nn.Conv2d(in_features, out_features, 4, 2, 1, bias=bias),
                         nn.BatchNorm2d(out_features),
                         nn.LeakyReLU(0.2, inplace=True)
                ]
            else:
                block = [
                         nn.Conv2d(in_features, out_features, 4, 2, 1, bias=bias),
                         nn.LeakyReLU(0.2, inplace=True)
                ]
            return block

        self.features = nn.Sequential(
            *encoder_block(channels, ndf, bn=False),
            *encoder_block(ndf, ndf*2, bn=True),
            *encoder_block(ndf*2, ndf*4, bn=True),
            *encoder_block(ndf*4, ndf*8, bn=True),
            nn.Conv2d(ndf*8, latent_dim, 4, 1, 0, bias=bias),
            nn.Tanh()
        )

    def forward(self, x):
        validity = self.features(x)
        return validity
```

##### **2. Hyper Parameters**

```python
n_epochs = 200
batch_size = 128
lr = 0.0002
ndf = 64
ngf = 64
latent_dim = 100
img_size = 64
channels = 3
n_critic = 5
split_rate = 0.8
```

##### **Data**

```python

class SimpleDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.transpose(self.data[idx], (2, 0, 1))
        labels = self.labels[idx]

        if self.transform:
            data = self.transform(data)


        return data, labels
```

##### **Train Data, Data Loader 정의**

```python
# Train Data
train_dataset = CIFAR10('./', train=True, download=True)

_x_train = torch.ByteTensor(train_dataset.data[torch.IntTensor(train_dataset.targets) == 1])
x_train, x_test_normal = _x_train.split((int(len(_x_train) * split_rate)), dim=0)

train_dataset_target = torch.Tensor(train_dataset.targets)
_y_train = train_dataset_target[train_dataset_target == 1]
y_train, y_test_normal = _y_train.split((int(len(_y_train) * split_rate)), dim=0)

train_mnist = SimpleDataset(x_train, y_train,
                                transform=transforms.Compose(
                                    [transforms.ToPILImage(),
                                     transforms.Resize(img_size),
                                     transforms.ToTensor()])
                                )
train_dataloader = DataLoader(train_mnist, batch_size=batch_size, shuffle=True)

```

##### **DCGAN 학습**

##### **4-1.optimizer 함수, GAN 모델 정의**

##### **layer weight 초기화**

```python
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

```

```python
G = Generator(latent_dim = latent_dim, ngf=ngf, channels=channels, bias=False).to(device)
G.apply(weights_init)
D = Discriminator(ndf=ndf, channels=channels, bias=False).to(device)
D.apply(weights_init)

optimizer_G = optim.Adam(G.parameters(), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))
```
