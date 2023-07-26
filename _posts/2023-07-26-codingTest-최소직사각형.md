---
layout: single
title: "[코딩테스트 준비]/완전 탐색/최소 직사각형"
categories: CodingTest
tag: [완전탐색, Python 코딩테스트]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 최소 직사각형

## 문제 설명:

![문제설명.png]({{site.url}}/images/2023-07-26-codingTest-최소직사각형/문제설명.png)

## First Thought:

해당 문제를 처음 읽었을 때 문제 이해를 못해서 이해하는 데 한참 걸렸던 것 같다. 하지만 문제를 이해하고 난 후에는 빠르게 문제를 풀 수 있었다. 문제 해독력을 키우는 것도 굉장히 중요한 것 같다.

### 문제 풀이:

```python

    def solution(sizes):
        max_list = []
        min_list = []
        for size in sizes:
            x, y = size
            max_list.append(max(x,y))
            min_list.append(min(x,y))
        return max(max_list) * max(min_list)
```

### 문제 접근법:

✔️ 해당 문제는 문제를 이해하는 것 및 어떻게 접근하는 지가 굉장히 중요하다. 카드를 회전 할 수 있으므로 가로 길이, 세로 길이 중 긴 것과 짧은 것을 나눈 후 각각 최대값을 구하는 방식이 적절하다고 판단하였다.
