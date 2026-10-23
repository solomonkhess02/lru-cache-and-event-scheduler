"""
Avaloka Take-home Assignment | Problem 2: Event Scheduler
Solomon Khess | IIT Bhubaneswar

Run: python3 problem2_event_scheduler.py
"""

import heapq
from typing import Iterable, List, Tuple

Event = Tuple[int, int]


def _validated(events: Iterable[Event]) -> List[Event]:
    """Copy the input into a list (so generators work and are read only once)
    and check every event. `not (start < end)` also rejects NaN times."""
    events = list(events)
    for start, end in events:
        if not (start < end):
            raise ValueError(f"invalid event ({start}, {end}): start must be before end")
    return events


def can_attend_all(events: Iterable[Event]) -> bool:
    ordered = sorted(_validated(events))  # by start time
    for (_, prev_end), (start, _) in zip(ordered, ordered[1:]):
        if start < prev_end:  # strict: equal times are allowed
            return False
    return True


def min_rooms_required(events: Iterable[Event]) -> int:
    end_times: List[int] = []  # min-heap: end times of rooms currently in use
    for start, end in sorted(_validated(events)):
        if end_times and end_times[0] <= start:
            # The room that frees up earliest is free by now: reuse it.
            heapq.heapreplace(end_times, end)
        else:
            # Every room is still busy: open a new one.
            heapq.heappush(end_times, end)
    # The heap never shrinks, so its final size is the peak number of rooms.
    return len(end_times)


def room_name(index: int) -> str:
    """0 -> 'Room A', 25 -> 'Room Z', 26 -> 'Room AA' (spreadsheet-style)."""
    if index < 0:
        raise ValueError("room index must be non-negative")
    letters, n = "", index
    while True:
        n, r = divmod(n, 26)
        letters = chr(ord("A") + r) + letters
        if n == 0:
            break
        n -= 1
    return f"Room {letters}"


def assign_rooms(events: Iterable[Event]) -> List[Tuple[Event, str]]:
    """Give each event a room name, using the minimum number of rooms.

    Returns (event, room) pairs in the same order as the input.
    """
    events = _validated(events)
    order = sorted(range(len(events)), key=lambda i: events[i])
    busy: List[Tuple[int, int]] = []  # min-heap of (end_time, room_index)
    free: List[int] = []              # min-heap of free room indices
    rooms_opened = 0
    assigned = [""] * len(events)

    for i in order:
        start, end = events[i]
        # Release every room whose meeting has ended by this start time.
        while busy and busy[0][0] <= start:
            _, room = heapq.heappop(busy)
            heapq.heappush(free, room)
        if free:
            room = heapq.heappop(free)  # lowest-lettered free room
        else:
            room = rooms_opened
            rooms_opened += 1
        heapq.heappush(busy, (end, room))
        assigned[i] = room_name(room)

    return list(zip(events, assigned))


if __name__ == "__main__":
    a = [(9, 10), (10, 11), (11, 12)]
    b = [(9, 12), (10, 11), (11, 13), (12, 14)]
    print("can_attend_all(a)     ->", can_attend_all(a))
    print("min_rooms_required(a) ->", min_rooms_required(a))
    print("can_attend_all(b)     ->", can_attend_all(b))
    print("min_rooms_required(b) ->", min_rooms_required(b))
    for event, room in assign_rooms(b):
        print(f"  {event} -> {room}")
