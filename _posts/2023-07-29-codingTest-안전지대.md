---
layout: single
title: "[코딩테스트 준비]/BFS| DFS/안전 지대"
categories: CodingTest
tag: [BFS DFS Python 코딩테스트]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 안전 지대

## 문제 설명:

![안전지대-문제설명.png]({{site.url}}/images/2023-07-29-codingTest-안전지대/안전지대-문제설명.png)

### First Thought:

문제를 보자마자 게임 맵 최단거리 문제를 떠올렸다. 그래프에서 1을 찾고 그 주변에 8개의 방향으로 움직이면서 그 주변에 있는 값들을 위험 지역으로 설정하고 1일 시 다시 덱에 넣어 탐색을 하는 방식으로 접근하였다.

### 문제 풀이:

```python

from collections import deque

def solution(board):

    m = len(board)
    n = len(board[0])

    def bfs(y, x):

        q = deque()
        q.append((y,x))

        dx = [1, -1, 0, 0, 1, 1, -1, -1]
        dy = [0, 0, 1, -1, 1, -1, 1, -1]

        visited = [[False]*m for _ in range(n)]

        while q:

            y, x = q.popleft()
            visited[y][x] = True

            for i in range(8):
                nx = x + dx[i]
                ny = y + dy[i]

                if 0<= nx <m and 0<= ny <n:
                    if not visited[ny][nx]:
                        if board[ny][nx] == 1:
                            q.append((ny, nx))
                        else:
                            board[ny][nx] = 2

        for i in range(m):
            for j in range(n):
                if board[j][i] == 1:
                    bfs(j, i)


        result = 0

        for i in range(m):
            result += board[i].count(0)
        return result
```

### 문제 접근법:

✔️ BFS로 접근하여 폭탄이 있을 때 그 주변을 위험 지역이라고 해준다.
✔️ BFS로 접근하여 폭탄이 있을 때 다시 큐에 내용을 입력할 수 있도록 한다.
✔️ 폭탄 주변에 대각선도 고려해야 하므로 보통 BFS가 4개의 원소로 하는 것을 8개 원소로 늘려서 한다.
