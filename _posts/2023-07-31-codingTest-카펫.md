---
layout: single
title: "[코딩테스트 준비]/카펫"
categories: CodingTest
tag: [완전탐색]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 카펫

## 문제 설명:

![카펫-문제설명.png]({{site.url}}/images/2023-07-31-codingTest-카펫/카펫-문제설명.png)

### First Thought:

브라운 개수와 옐로 타일 개수의 합을 구한 후, 그것의 인수 값을 구한 다음 조건을 만족하는 인수값을 활용해야 겠다고 판단하였다.

### 문제 풀이:

```python

def solution(brown, yellow):
    sum = brown + yellow
    cand_1 = []
    cand_2 = []
    for num in range(2, sum):
        if sum % num == 0:
            cand_1.append(num)
            cand_2.append(sum//num)
    ans = []
    for i in range(len(cand_1)):
        if (int(cand_1[i]-2))*(int(cand_2[i]-2)) == yellow:
            ans.append(cand_2[i])
    result = []
    result.append(max(ans))
    result.append(sum//max(ans))
    return result
```

### 문제 접근법:

✔ brown 타일 수 + yellow 타일 수가 전체 타일 수라는 점을 활용한다.

✔ 전체 타일 수의 인수들을 구한다. 인수가 곧 가로 혹은 세로의 숫자이다.

✔ yellow 타일은 중아에 있어야 하므로 전체 가로 혹은 세로에서 부터 2칸씩 떨어진 곳에서부터의 곱과 같다.

### 다른 사람 풀이:

```python

def solution(brown, yellow):
    for i in range(1, int(yellow**(1/2))+1):
        if yellow % i == 0:
            if 2*(i + yellow//i) == brown-4:
                return [yellow//i+2, i+2]

```
