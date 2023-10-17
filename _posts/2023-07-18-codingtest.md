---
layout: single
title: "[코딩테스트 준비]/정렬/가장 큰 수"
categories: CodingTest
tag: [정렬]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# 가장 큰 수[정렬]

## 문제 정의:

![image-20230718232337079]({{site.url}}\images\2023-07-18-third\image-20230718232337079.png)

## 문제 제한 사항:

![image-20230718232451688]({{site.url}}\images\2023-07-18-third\image-20230718232451688.png)

---

## First Thought:

1. 주어진 numbers list로 만들 수 있는 모든 조합을 만든다.
2. 만들어진 조합을 정렬시킨다.
3. 정렬된 조합 중 값이 가장 큰 값을 return한다.

## Code for execution of First Thought:

```python
from itertools import permutations

def solution(numbers):
    result_list = []
    combi = list(permutations(numbers, len(numbers)))
    for comb in combi:
        A = "".join(map(str, comb))
        result_list.append(int(A))
    return str(max(result_list))

```

- itertools 라이브러리에서 permutations 함수를 불러왔다.
- permutations(리스트, 조합의 길이)
- permutations를 통해 생성된 리스트 원소 값들을 ''.join 함수와 map 함수를 통해 str으로 바꾼다.
- 구한 리스트에서 최대 값을 구한다.

## 코드 실행 결과:

처음 실행 한 결과값들은 맞으나, 시간 초과로 인해 오답처리 되었다. 😟

---

## 올바른 풀이:

다른 방식으로 문제를 접근하려고 했으나, 시간이 많이 지체되는 관계로 다른 사람들의 풀이를 참고하고 다시 풀어 보았다.

## 다른 사람의 풀이:

```python
def solution(numbers):
    numbers = list(map(str, numbers))
    numbers.sort(key=lambda x:x*3, reverse=True)
    return str(int("".join(numbers)))
```

✔ lambda x: x\*3: num 인자 각각의 문자열을 3번 반복을 하는 것을 의미한다.

✔x: x\*3을 하는 이유: [3, 30, 34, 5, 9] 리스트가 주어졌을 때 단순히 정렬을 하면 30이 3보다 먼저 정렬된다. 이는 최적의 값이 아니므로 문자열에 3번씩 반복해서 만든 후 정렬을 하면 올바르게 정렬할 수 있다.

숫자 3을 3번 반복을 하게 될 시: 333이 되고, 34를 세번 반복할 시 343434, 30을 세번 반복할 시, 303030이다.

❗이때 제한 사항을 보면, numbers의 원소 값이 1000이하라는 것을 알 수 있다. 따라서 3의 값은 333, 34의 값은 343, 30의 값은 303으로 되며, 정렬하면, 34, 3, 30 순으로 되는 것을 알 수 있다.

## 배운점:

1. 제약 조건은 그냥 있는 것이 아니다

2. python 라이브러리에서 제공하는 permutation의 시간복잡도는 O(n!)으로 시간 복잡도가 굉장히 크다. 따라서 시간 복잡도 제한에 걸릴 가능성이 높다.

3. python 라이브러리를 활용하지 않은 상태에서 순열을 구현하고 싶을 때 사용하는 코딩을 짜보았다.

   ```py
   #재귀함수를 이용한 풀이

   def perm(arr, n):
       result = []
       if n > len(arr):
           return result
       if n == 1:
           for i in arr:
               result.append([i])
       elif n > 1:
           for i in range(len(arr)):
               ans = [i for i in arr]
               ans.remove(arr[i])
               for p in perm(ans, n-1):
                   result.append([arr[i]]+p)
      return result

   arr = [1, 2, 3]
   print(perm(arr, 3))

   ```
