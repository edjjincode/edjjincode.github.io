---
layout: single
title: "[자료구조 연결리스트]"
categories: CodingTest
tag: [Python 자료구조]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

# Linked List(연결리스트)

## 연결리스트란?

**연결리스트는 각 노드가 데이터와 포인터를 가지고 한 줄로 연결되어 있는 방식으로 데이터를 저장한다.** 연결리스트의 두드러지는 특징들을 배열과 비교해서 보면 좋다.

### 배열 vs 연결리스트

✔ 배열은 물리적인 메모리 주소가 연속적이고, **연결리스트는 물리 메모지 주소가 연속적이지 않고 랜덤이다**.
✔ 배열은 삽입/삭제가 O(n)의 시간이 걸리지만, 동적으로 **연결된 연결 리스트는 O(1) 시간이 걸린다**.
✔ 배열은 각 원소에 인덱스로 O(1)의 시간으로 손쉽게 접근이 가능하나 연결 리스트 같은 경우 O(n) 시간이 소요된다.(배열과 다르게 연결된 메모리가 아니기 때문에, 데이터를 찾기 위해서는 모든 노드를 거쳐서 탐색해야 한다.)

### 싱글 연결리스트

#### 노드 정의:

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

#### 연결리스트 생성:

보통 헤더를 선언하여 연결리스트를 생성하고, 헤더를 통해 다른 모든 노드를 탐색하고 참조할 수 있다.

```python
head = ListNode(0)

curr_node = head

new_node = ListNode(1)
curr_node.next = new_node
curr_node = curr_node.next

curr_node.next = ListNode(2)
curr_node = curr_node.next

curr_node.next = ListNode(3)
curr_node = curr_node.next

curr_node.next = ListNode(4)
curr_node  = curr_node.next

```

#### 전체 연결리스트 출력:

```python

node = head
while node:
    print(node.val)
    node = node.next

```

#### 노드 탐색하여 삭제:

```python
node = head
while node.next:
    if node.next.val == 2:
        next_node = node.next.next
        node.next = next_node
        break
    node = node.next

node = head
while node:
    print(node.val)
    node = node.next
```
