import random
import threading
import unittest

from problem1_lru_cache import LRUCache, ThreadSafeLRUCache
from problem2_event_scheduler import assign_rooms, can_attend_all, min_rooms_required, room_name


class TestLRUCache(unittest.TestCase):
    def test_basic_eviction(self):
        c = LRUCache(2)
        c.put(1, 1)
        c.put(2, 2)
        self.assertEqual(c.get(1), 1)
        c.put(3, 3)                      # evicts 2
        self.assertEqual(c.get(2), -1)
        c.put(4, 4)                      # evicts 1
        self.assertEqual(c.get(1), -1)
        self.assertEqual(c.get(3), 3)
        self.assertEqual(c.get(4), 4)

    def test_update_refreshes_recency(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("a", 10)                   # update makes "a" most recent
        c.put("c", 3)                    # so "b" is evicted
        self.assertEqual(c.get("b"), -1)
        self.assertEqual(c.get("a"), 10)
        self.assertEqual(len(c), 2)

    def test_capacity_one(self):
        c = LRUCache(1)
        c.put(1, 1)
        c.put(2, 2)
        self.assertEqual(c.get(1), -1)
        self.assertEqual(c.get(2), 2)

    def test_invalid_capacity(self):
        for bad in (0, -1, True, False, 2.0, "2", None):
            with self.assertRaises(ValueError):
                LRUCache(bad)

    def test_none_key_and_value(self):
        c = LRUCache(2)
        c.put(None, "n")
        c.put("x", None)
        self.assertEqual(c.get(None), "n")
        self.assertIsNone(c.get("x"))

    def test_repeated_put_same_key_does_not_evict(self):
        c = LRUCache(2)
        c.put(1, 1)
        c.put(2, 2)
        for i in range(10):
            c.put(1, i)
        self.assertEqual(c.get(2), 2)
        self.assertEqual(c.get(1), 9)
        self.assertEqual(len(c), 2)

    def test_get_miss_does_not_change_order(self):
        c = LRUCache(2)
        c.put(1, 1)
        c.put(2, 2)
        c.get(99)
        self.assertEqual(c.keys_in_order(), [2, 1])

    def test_matches_ordereddict(self):
        """Larger random run against collections.OrderedDict as the reference."""
        from collections import OrderedDict
        rng = random.Random(7)
        for cap in (1, 2, 3, 10, 100):
            c, ref = LRUCache(cap), OrderedDict()
            for _ in range(20000):
                k = rng.randint(0, cap * 3)
                if rng.random() < 0.5:
                    exp = -1
                    if k in ref:
                        ref.move_to_end(k)
                        exp = ref[k]
                    self.assertEqual(c.get(k), exp)
                else:
                    v = rng.random()
                    if k in ref:
                        ref.move_to_end(k)
                    elif len(ref) >= cap:
                        ref.popitem(last=False)
                    ref[k] = v
                    c.put(k, v)
            self.assertEqual(c.keys_in_order(), list(reversed(ref)))

    def test_matches_reference_model(self):
        """Random operations compared against a simple list-based model."""
        rng = random.Random(0)
        for _ in range(300):
            cap = rng.randint(1, 5)
            c, model = LRUCache(cap), []  # model: [(key, value)], most recent last
            for _ in range(200):
                k = rng.randint(0, 8)
                if rng.random() < 0.5:
                    hit = [kv for kv in model if kv[0] == k]
                    expected = hit[0][1] if hit else -1
                    if hit:
                        model.remove(hit[0])
                        model.append(hit[0])
                    self.assertEqual(c.get(k), expected)
                else:
                    v = rng.randint(0, 100)
                    model = [kv for kv in model if kv[0] != k]
                    if len(model) >= cap:
                        model.pop(0)
                    model.append((k, v))
                    c.put(k, v)
                self.assertEqual(c.keys_in_order(), [kv[0] for kv in reversed(model)])

    def test_thread_safe_stress(self):
        c = ThreadSafeLRUCache(50)

        def worker(seed):
            rng = random.Random(seed)
            for _ in range(5000):
                k = rng.randint(0, 200)
                if rng.random() < 0.5:
                    c.put(k, k)
                else:
                    v = c.get(k)
                    assert v in (-1, k)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(len(c), 50)
        self.assertEqual(len(c.keys_in_order()), len(c))  # list and map agree


def brute_force_rooms(events):
    """Peak number of events running at once, checked at every start time."""
    if not events:
        return 0
    return max(sum(1 for s, e in events if s <= t < e) for t, _ in events)


class TestScheduler(unittest.TestCase):
    def test_given_example(self):
        ev = [(9, 10), (10, 11), (11, 12)]
        self.assertTrue(can_attend_all(ev))
        self.assertEqual(min_rooms_required(ev), 1)

    def test_overlap(self):
        ev = [(9, 12), (10, 11), (11, 13), (12, 14)]
        self.assertFalse(can_attend_all(ev))
        self.assertEqual(min_rooms_required(ev), 2)

    def test_unsorted_input(self):
        ev = [(11, 12), (9, 10), (10, 11)]
        self.assertTrue(can_attend_all(ev))
        self.assertEqual(min_rooms_required(ev), 1)

    def test_all_overlap(self):
        ev = [(1, 10), (2, 9), (3, 8), (4, 7)]
        self.assertFalse(can_attend_all(ev))
        self.assertEqual(min_rooms_required(ev), 4)

    def test_empty_and_single(self):
        self.assertTrue(can_attend_all([]))
        self.assertEqual(min_rooms_required([]), 0)
        self.assertTrue(can_attend_all([(1, 2)]))
        self.assertEqual(min_rooms_required([(1, 2)]), 1)

    def test_invalid_event(self):
        with self.assertRaises(ValueError):
            min_rooms_required([(5, 5)])
        with self.assertRaises(ValueError):
            can_attend_all([(6, 5)])

    def test_random_against_brute_force(self):
        rng = random.Random(42)
        for _ in range(2000):
            n = rng.randint(0, 12)
            ev = []
            for _ in range(n):
                s = rng.randint(0, 20)
                ev.append((s, s + rng.randint(1, 6)))
            expected = brute_force_rooms(ev)
            self.assertEqual(min_rooms_required(ev), expected)
            self.assertEqual(can_attend_all(ev), expected <= 1)

            pairs = assign_rooms(ev)
            self.assertEqual([p[0] for p in pairs], ev)            # input order kept
            self.assertEqual(len({r for _, r in pairs}), expected)  # minimal rooms
            by_room = {}
            for (s, e), r in pairs:
                by_room.setdefault(r, []).append((s, e))
            for room_events in by_room.values():                   # no clash in a room
                self.assertTrue(can_attend_all(room_events))

    def test_room_names(self):
        self.assertEqual(room_name(0), "Room A")
        self.assertEqual(room_name(25), "Room Z")
        self.assertEqual(room_name(26), "Room AA")
        self.assertEqual(room_name(27), "Room AB")
        self.assertEqual(room_name(701), "Room ZZ")
        self.assertEqual(room_name(702), "Room AAA")
        with self.assertRaises(ValueError):
            room_name(-1)

    def test_generator_input(self):
        """Regression: generators used to be consumed by validation."""
        self.assertFalse(can_attend_all(x for x in [(1, 3), (2, 4)]))
        self.assertEqual(min_rooms_required(iter([(1, 3), (2, 4)])), 2)
        self.assertEqual(assign_rooms(iter([(1, 3), (2, 4)])),
                         [((1, 3), "Room A"), ((2, 4), "Room B")])

    def test_nan_rejected(self):
        nan = float("nan")
        for ev in ([(1, nan)], [(nan, 2)]):
            with self.assertRaises(ValueError):
                min_rooms_required(ev)

    def test_input_not_mutated(self):
        ev = [(11, 12), (9, 10), (10, 11)]
        before = list(ev)
        can_attend_all(ev)
        min_rooms_required(ev)
        assign_rooms(ev)
        self.assertEqual(ev, before)

    def test_floats_lists_duplicates_negatives(self):
        self.assertEqual(min_rooms_required([(9.5, 10.25), (10.25, 11), (10, 10.5)]), 2)
        self.assertEqual(min_rooms_required([[1, 3], [2, 4]]), 2)
        self.assertEqual(min_rooms_required([(1, 2)] * 5), 5)
        self.assertTrue(can_attend_all([(-5, -1), (-1, 3)]))

    def test_same_start_different_ends(self):
        ev = [(1, 5), (1, 2), (2, 3), (3, 5)]
        self.assertEqual(min_rooms_required(ev), 2)
        self.assertFalse(can_attend_all(ev))

    def test_large_input(self):
        rng = random.Random(1)
        ev = [(s, s + rng.randint(1, 50)) for s in (rng.randint(0, 10**6) for _ in range(200000))]
        self.assertGreaterEqual(min_rooms_required(ev), 1)
        pairs = assign_rooms(ev)
        self.assertEqual(len({r for _, r in pairs}), min_rooms_required(ev))


if __name__ == "__main__":
    unittest.main(verbosity=2)
