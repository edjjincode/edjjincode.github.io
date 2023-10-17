---
layout: single
title: "[코딩테스트 준비]/피로도"
categories: CodingTest
tag: [완전탐색]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 피로도

## 문제 설명:

![피로도-문제설명.png]({{site.url}}/images/2023-08-06-codingTest-피로도/문제설명.png)

![피로도-문제설명_1.png]({{site.url}}/images/2023-08-06-codingTest-피로도/문제설명_1.png)

![피로도-문제설명_2.png]({{site.url}}/images/2023-08-06-codingTest-피로도/문제설명_2.png)

### First Thought

처음 문제를 보았을 때, 큐나 덱 개념으로 생각을 했다.

dungeons (need, spend)에서

1. need가 k 보다 클시, break 한다.
2. need와 k가 같을 때는 k-(need 값이 큰 값의 spend 값)
3. need가 k 보다 작을 시, k-(spend값이 가장 작은 값)

으로 접근하려고 하였으나, dungeon에서 need값이 최대인 spend 값을 지정하는 것이 너무 어려웠다.

### 문제 풀이:

```python

from itertools import permutations

def solution(k, dungeons):
    answer = []
    for p in permutations(dungeons, len(dungeons)):
        tmp = k
        cnt = 0
        for dungeon in p:
            x, y = dungeon
            if tmp >= x:
                tmp -= y
                cnt += 1
                answer.append(cnt)
    return max(answer)

```

### 문제 접근법:

한번에 최적의 문제를 푸는 방식으로 접근하는 것이 아닌, 모든 조합을 구한 후, 최고의 값을 구하는 방식으로 접근하였다.

### 다른 풀이:

```python

answer = 0

def dfs(k, cnt, dungeons, visited):
    global answer
    if cnt > answer:
        answer = cnt

    for i in range(len(dungeons)):
        if not visited[i] and k >= dungeons[i][0]:
            visited[i] = True
            dfs(k - dungeon[i][1], cnt + 1, dungeons, visited)
            visited[i] = False

def solution(k, dungeons):
    global answer
    visited = [False]*len(dungeons)
    dfs(k, 0, dungeons, visited)
    return answer
```
