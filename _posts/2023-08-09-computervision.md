---
layout: single
title: "[컴퓨터비전 전처리]"
categories: TIL
tag: [CV TIL]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# OpenCV

OpenCV란 Open Computer Vision의 약자로 영상처리에 사용할 수 있는 라이브러리다.

OpenCV는 오픈소스이나, BSD(Berkely Software Distribution) 라이센스를 따르기 때문에 상업적 목적을 사용할 수 있다.

## Packages:

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

def imshow(title = "Image", image = None, size = 16):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
```

```python
cv2.__version__
```

```python
!wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip
!unzip -qq images.zip
```

## Basic Image Data Operations

### Load Images

```python
%cd /content/images
```

```python
image = cv2.imread('castara.jpeg')
```

### Displaying Images

'castara.jpeg'에 해당하는 이미지는 다음과 같다

```python
from matplotlib import pyplot as plt

plt.imshow(image)
```

![castara]({{site.url}}/images/2023-08-09-opencv_1/castara.png){: .align-center}r
