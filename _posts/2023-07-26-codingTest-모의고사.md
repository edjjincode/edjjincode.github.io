---
layout: single
title: "[코딩테스트 준비]/완전 탐색/모의고사"
categories: CodingTest
tag: [완전탐색, Python 코딩테스트]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 모의고사

## 문제 설명:

![모의고사-문제설명][{{site.url}}/images/2023-07-26-codingTest-최소직사각형/모의고사-문제설명.png]

## First Thought:

문제를 처음 봤을 때 풀어봤던 문제라는 것을 바로 알 수 있었다. 당시 많은 시행착오를 겪어서 쉽게 풀 수 있었던 것 같다.

### 문제 풀이:

```python
    def solution(answers):
        n1 = [1, 2, 3, 4, 5]
        n2 = [2, 1, 2, 3, 2, 4, 2, 5]
        n3 = [3, 3, 1, 1, 2, 2, 4, 4, 5, 5]

        sol = [0, 0, 0]

        for i in range(len(answers)):
            if answers[i] == n1[i % 5]:
                sol[0] += 1
            if answers[i] == n2[i % 8]:
                sol[1] += 1
            if answers[i] == n3[i % 10]:
                sol[2] += 1
        ans = []

        for ind, num in enumerate(sol):
            if num >= max(sol):
                ans.append(ind+1)
        return ans

```

## 문제 접근법:

✔️ 이 문제의 가장 큰 핵심 포인트 중 하나는 각 번호별 수포자의 방식을 하나의 리스트 형태로 만드는 것이다.
✔️ 각 리스트의 숫자가 무한하게 반복되는 것을 인덱스에 리스트의 길이 만큼 나눈 후 그 나머지가 인덱스와 같을 시 값에 1을 추가하는 방식으로 진행된다.
✔️ 결과 출력을 리스트 형태로 진행해야 하므로 출력값을 sol = [0, 0, 0] 형태로 둔다.
✔️ enumerate 함수를 쓴 후 sol 중에서 숫자가 가장 큰 숫자의 인덱스에 1을 더한 것을 출력한다.
