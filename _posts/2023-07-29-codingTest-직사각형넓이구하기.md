---
layout: single
title: "[코딩테스트 준비]/직사각형 넓이 구하기"
categories: CodingTest
tag: [Python, 코딩테스트]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 직사각형 넓이 구하기

## 문제 설명:

![직사각형-문제설명.png]({{site.url}}/images/2023-07-29-codingTest-직사각형/문제설명.png)

### First Thought:

높이가 같은 것들 간에 가로축 차이를 구하고 가로축이 같은 것 중에 높이가 다른 것을 통해 높이를 구해 넓이를 구하는 방식으로 접근하였다.

### 문제 풀이:

```python

def solution(dots):
    hor = []
    ver = []
    a = 0
    b = 0
    x, y = dots[0]
    while len(dots) > 1:
        pop_x, pop_y = dots.pop()
        if y == pop_y:
            a = abs(x-pop_x)
        if x == pop_x:
            b = abs(y-pop_y)
    return a*b

```

### 문제 접근법:

✔️ 동일한 리스트에 있는 값 중에 같은 값을 가졌는 지 확인할 때 while len(리스트)> 1이라고 해 놓고 리스트에 있는 값을 하나씩 pop하면서 같은 지 확인한다.(리스트에 있는 모든 원소들은 적어도 x 혹은 y 값 중에 같은 값을 가지므로)
