---
layout: single
title: "[코딩테스트 준비]/BFS|DFS/게임 맵 최단거리"
categories: CodingTest
tag: [BFS, DFS, Python, 코딩테스트]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 게임 맴 최단 거리 문제

## 문제 설명:

![image-20230724175044305]({{site.url}}/images/2023-07-24-codingtest-게임 맵 최단거리/image-20230724175044305.png)

![image-20230724175253934]({{site.url}}/images/2023-07-24-codingtest-게임 맵 최단거리/image-20230724175253934.png)

![image-20230724175338179]({{site.url}}/images/2023-07-24-codingtest-게임 맵 최단거리/image-20230724175338179.png)

## First Thought

처음 문제를 보았을 때 최단 거리 문제이기 떄문에 BFS형태로 문제를 푸는 것이 좋을 것 같다고 판단하였다. BFS로 풀려고 하는 데 2차원 형태의 BFS 문제는 처음이라 접근하기 어려워 다른 풀이를 참고하여 풀 수 있었다.

## 문제 풀이:

```python
    from collections import deque
    def solutions(maps):
        m = len(maps)
        n = len(maps[0])
        visited = [[False]*m for _ in range(n)]

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        q = deque()
        q.append((0,0))
        visited[0][0] = True

        while q:

            y, x = q.popleft()

            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]

                if 0<= nx <m and 0<= ny <n and maps[ny][nx] == 1:
                    if not visited[ny][nx]:#접근하지 않았었다면
                        visited[ny][nx] = True #접근했었다는 흔적 남기기
                        q.append((ny, nx))
                        maps[ny][nx] = maps[y][x] + 1 #접근했던 값에 +1을 해준다.

            if maps[ny-1][nx-1] == 1:
                return -1

            else:
                return maps[ny-1][nx-1]

```

## 문제 접근 방법:

✔️ m = len(maps)는 행수를 의미한다. n = len(maps[0])은 열 수를 의미한다.

✔️ 접근 여부를 [False] 형태로 matrix를 만든다.
✔️ dx = [-1, 1, 0, 0], dy = [0, 0, -1, 1] 형태로 만들어서 index를 활용하여 상하좌우를 의미하도록 한다.
✔️ matrix의 조건들을 만족시킬 수 있도록 하고
✔️ 아직 접근하지 않았던 리스트에 대해서
✔️ 새로 움직인 위치를 visited 리스트에 추가하고 접근하였다라고 바꿔준다.
✔️ 새로 움직인 위치를 q 리스트에 넣어준다.
✔️ maps에 있는 값에 1을 추가해준다.
