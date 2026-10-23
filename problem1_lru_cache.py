"""
Avaloka Take-home Assignment | Problem 1: LRU Cache
Solomon Khess | IIT Bhubaneswar

Run: python3 problem1_lru_cache.py
"""

import threading


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        # bool is a subclass of int in Python, so reject it explicitly.
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._map = {}
        # Sentinel nodes so insert/remove never need None checks.
        self._head = _Node()  # most recently used side
        self._tail = _Node()  # least recently used side
        self._head.next = self._tail
        self._tail.prev = self._head

    # ---- linked list helpers, all O(1) ----
    def _unlink(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node):
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node):
        self._unlink(node)
        self._push_front(node)

    # ---- public API ----
    def get(self, key):
        node = self._map.get(key)
        if node is None:
            return -1
        self._move_to_front(node)  # mark as recently used
        return node.value

    def put(self, key, value):
        node = self._map.get(key)
        if node is not None:  # update existing key
            node.value = value
            self._move_to_front(node)
            return
        if len(self._map) >= self.capacity:  # evict least recently used
            lru = self._tail.prev
            self._unlink(lru)
            del self._map[lru.key]
        node = _Node(key, value)
        self._map[key] = node
        self._push_front(node)

    def __len__(self):
        return len(self._map)

    def keys_in_order(self):
        """Keys from most to least recently used (for debugging and tests)."""
        out, node = [], self._head.next
        while node is not self._tail:
            out.append(node.key)
            node = node.next
        return out


class ThreadSafeLRUCache(LRUCache):
    """Same cache, with one lock around each operation.

    get() also rewrites the list (it moves the node to the front), so even
    reads need exclusive access; a reader-writer lock would not help here.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            return super().get(key)

    def put(self, key, value):
        with self._lock:
            super().put(key, value)

    def __len__(self):
        with self._lock:
            return super().__len__()

    def keys_in_order(self):
        with self._lock:
            return super().keys_in_order()


if __name__ == "__main__":
    cache = LRUCache(2)
    cache.put(1, "one")
    cache.put(2, "two")
    print("get(1) ->", cache.get(1))        # one; 1 is now most recent
    cache.put(3, "three")                    # evicts 2
    print("get(2) ->", cache.get(2))        # -1
    print("get(3) ->", cache.get(3))        # three
    print("order  ->", cache.keys_in_order())
