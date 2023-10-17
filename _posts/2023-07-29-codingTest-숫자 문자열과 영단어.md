---
layout: single
title: "[코딩테스트 준비]/숫자 문자열과 영단어"
categories: CodingTest
tag: [Python 코딩테스트]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 숫자 문자열과 영단어

## 문제 설명:

![숫자문자열-문제설명.png]({{site.url}}/images/2023-07-29-codingTest-숫자문자열/숫자문자열-문제설명.png)

### First Thought:

처음에는 contain 함수가 있으면 좋곘다고 생각했으나, 파이썬에서는 제공을 안하므로 string input에 영단어가 존재할 시에 숫자로 바꿀 수 있도록 하였다.

### 문제 풀이:

```python

def solution(s):

    if "zero" in s:
        s = s.replace("zero", "0")
    if "one" in s:
        s = s.replace("one", "1")
    if "two" in s:
        s = s.replace("two", "2")
    if "three" in s:
        s = s.replace("three", "3")
    if "four" in s:
        s = s.replace("four", "4")
    if "five" in s:
        s = s.replace("five", "5")
    if "six" in s:
        s = s.replace("six", "6")
    if "seven" in s:
        s = s.replace("seven", "7")
    if "eight" in s:
        s = s.replace("eight", "8")
    if "nine" in s:
        s = s.replace("nine", "9")
    return int(s)

```

### 문제 접근법:

✔️ 문자로 표현된 숫자들을 숫자로 바꾸는 과정을 거쳤다. 이때 사용되는 함수는 replace 함수이다.
