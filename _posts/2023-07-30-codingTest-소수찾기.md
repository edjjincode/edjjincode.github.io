---
layout: single
title: "[코딩테스트 준비]/소수 찾기"
categories: CodingTest
tag: [구현]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 소수 찾기

## 문제 설명:

![소수찾기-문제설명.png]({{site.url}}/images/2023-07-30-codingTest-소수찾기/문제설명.png)

### First Thought:

문제를 보자마자 itertools에서 제공하는 permutations를 사용하겠다고 판단하였다. 또한 소수 n을 구할 때 n과 2에서부터 n까지의 숫자를 나눠서 나머지가 0인 것들은 소수가 아니라고 판단하였다.

### 문제 풀이:

```python

import itertools

def solution(numbers):
    ans = []
    comb = []
    for i in range(1, len(numbers)+1):
        comb += list(itertools.permutations(numbers, i))
    for tup in comb:
        A = "".join(tup)
        ans.append(int(A))
    ans = set(ans)
    result = []
    for num in ans:
        if num < 2:
            continue
        check = True
        for i in range(2, num):
            if num % i == 0:
                check = False
                break
        if check:
            result.append(num)
    return len(result)

```

### 문제 접근법:

✔ 숫자 리스트에서 모든 숫자의 조합을 구하려고 할 떄는

```python
comb = []
for i in range(1, len(numbers) + 1):
    comb += list(itertools.permutations(numebers, i))
```

를 활용한다.

리스트를 + 형태로 더할 시 여러 개의 리스트의 원소들을 하나의 리스트에 넣을 수 있다.

✔ 튜플 형태로 되어있는 것을 하나의 숫자로 만들고 싶을 때

```python
"".join(tuple)
```

을 활용하면 된다.

✔ 소수를 구하려고 할 때,

```python
check = True
for i in range(2, num):
    if num % i == 0:
        check = False #해당 숫자가 아니라는 것을 마킹하는 과정
        break
    if check: # 마킹이 안된 숫자에 한에서..
        new_list.append(num)

```

를 활용한다.
