---
layout: single
title: "[코딩테스트 준비]/BFS|DFS/타깃 넘버"
categories: CodingTest
tag: [BFS/DFS]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 타깃넘버[BFS/DFS]

## 문제정의:

![image-20230723120351484]({{site.url}}/images/2023-07-23-codingtest-타깃넘버/image-20230723120351484.png)

### 문제 해석:

주어진 숫자 리스트의 숫자를 음수 혹은 양수화 시킨 후 합계를 구해서 타깃 값과 일치하는 리스트 배열 숫자를 return 하는 문제이다.

## First Thought

문제를 처음 보았을 때 굉장히 난감하였다. 처음에는 숫자들의 조합이나 패턴을 찾으려고 하였으나 패턴 수가 너무 방대하였다. 다른 방식으로 문제를 풀 수 있는 방법을 고안하였고 BFS 방식을 생각하게 되었다.

🌕 BFS 방법을 고안하게 된 배경:

각 숫자 리스트의 원소 값들을 양수 값으로 바꾸거나 음수 값으로 바꿔가면서 그 합이 Target 값과 같아지는 갯수를 구하는 것이므로 각 숫자를 더하거나 뺴서 구할 수 있는 모든 경우의 수를 구한 후 조건을 만족하는 값을 구한다.

### BFS를 활용한 풀이:

```python

def solution(numbers, target):
    answer = 0
    leaves = [0] #숫자들을 더할 수 있게 leaves 리스트에 0을 넣는다.
    for num in numbers:
        tmp = [] #더한 값을 임시적으로 저장할 수 있는 리스틀 생성한다.
        for parent in leaves: #leaves에 있는 모든 값에 대하여 다음 숫자를 더하거나 뺸다.
            tmp.append(parent + num)
            tmp.append(parent - num)
        leaves = tmp #해당하는 값을 다시 leaves에 넣는다.
    for leaf in leaves:
        if leaf == target:
            answer += 1
    return answer

```

✔️ 값들의 합을 구해야 하므로 leaves 리스트에 [0]을 설정해준다.
✔️ 리스트에 있는 넘버에 대하여 뺴거나 더한 값을 tmp라는 임시 리스트에 넣어준 뒤 다시 그 값을 불러와서 숫자를 더하거나 빼는 방식으로 진행된다.

### 너비 우선 탐색(BFS, Breadth-First Search)

#### 너비 우선 탐색이란?

루트 노드에서 시작해서 인접한 노드를 먼저 탐색하는 방법을 말한다.

- 시작 정점으로부터 가까운 정점을 먼저 방문하고 멀리 떨어져 있는 정점을 나중에 방문하는 순회 방법이다.
- 두 노드 사이의 최단 경로 혹은 임의의 경로를 찾고 싶을 때 해당하는 방법을 사용한다.

#### 너비 우선 탐색 동작 과정

1. 탐색 시작 노드를 큐에 삽입하고 방문 처리를 한다.
2. 큐에서 노드를 꺼낸 뒤에 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리한다.

#### 너비 우선 탐색(BFS)의 특징

- 직관적이지 않은 면이 있다. BFS는 시작 노드에서 시작해서 거리에 따라 단계별로 탐색한다고 볼 수 있다.

- BFS는 재귀적으로 동작하지 않는다.
- BFS는 방문한 노드들을 차례로 저장한 후 꺼낼 수 있는 자료 구조인 큐를 사용한다.
- 선입선출(FIFO) 원칙으로 탐색한다.

#### 너비 우선 탐색 코드

```python
from collections import deque

def bfs(graph, start, visited):
    #큐(Queue) 구현을 위해 deque 라이브러리 사용
    queue = deque([start])
    #현재 노드를 방문 처리
    visited[start] = True
    #큐가 빌 때까지 반복
    while queue:
        #큐에서 하나의 원소를 뽑아 출력
        v = queue.popleft()
        print(v, end= ' ')
        #해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True

#각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

#각 노드가 방문된 정보를 리스트 자료형으로 표현(1차원 리스트)

visited = [False] * 9

#정의된 BFS 함수 호출

bfs(graph, 1, visited)
```

---

### DFS를 활용한 풀이:

```python

def dfs(numbers, target, idx, values):

    global cnt
    cnt = 0

    #깊이가 끝까지 닿았으면
    if idx == len(numbers) & values == target:
        cnt += 1
        return

    #끝까지 탐색했는데 sum이 target과 다르다면 그냥 넘어간다
    elif idx == len(numbers):
        return

    #재귀함수로 구현
    dfs(numbers, target, idx+1, values + numbers[idx])
    dfs(numbers, target, idx+1, values - numbers[idx])

def solution(numbers, target):

    global cnt
    dfs(numbers, target, 0, 0)

    return cnt
```

### DFS의 기본 원칙

DFS에서 데이터를 찾을 때는 항상 "앞으로 찾아야 가야할 노드"와 "이미 방문한 노드"를 기준으로 데이터를 탐색한다.

### DFS의 구현 방식

DFS를 구현할 때는 기본적으로 "스택/큐를 활용"할 수도 있고, "재귀함수를 통해 구현"할 수 있다.

##### deque을 활용한 DFS 구현

```python
def dfs2(graph, start_node):
    #deque 패키지 불러오기
    from collections import deque
    visited = []
    need_visited = deque()

    #시작 노드 설정해주기
    need_visited.append(start_node)

    #방문이 필요한 리스트가 아직 존재한다면
    while need_visited:
        node = need_visited.pop()

        ##만약 방문한 리스트에 없다면
        if node not in visited:

            ##방문 리스트에 노드를 추가
            visited.append(node)
            ##인접 노드들을 방문 예정 리스트에 추가
            need_visited.extend(graph[node])
    return visited
```

##### 재귀함수를 활용한 DFS 구현

```python
def dfs_recursive(graph, start, visited = []):
    visited.append(start)

    for node in graph[start]:
        if node not in visited:
            dfs_recursive(graph, node, visited)
    return visited
```
