"""Dividend buckets (Totem / dividend-swap style) and their allocation to ex-dates.

Totem dividend consensus is quoted per period (typically a calendar year, the
sum of dividends going ex in the period).  Forwards need dividends per
ex-date, so the controller spreads each bucket level over an intra-period
profile (normally the FO ex-date calendar).  A wrong profile is one of the
mechanisms that make implied repo curves irregular.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import Dividends, ZeroCurve


@dataclass(frozen=True)
class BucketSchedule:
    """Contiguous dividend periods ``(edges[y], edges[y+1]]`` in year fractions."""

    edges: np.ndarray

    def __post_init__(self) -> None:
        e = np.asarray(self.edges, dtype=float)
        if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0):
            raise ValueError("edges must be strictly increasing with at least two entries")
        object.__setattr__(self, "edges", e)

    @property
    def size(self) -> int:
        return self.edges.size - 1

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def index(self, t) -> np.ndarray:
        """Bucket index of each time (``-1`` outside the schedule)."""
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.edges, t, side="left") - 1
        return np.where((t > self.edges[0]) & (t <= self.edges[-1]), idx, -1)


def bucket_levels(divs: Dividends, schedule: BucketSchedule) -> np.ndarray:
    """Sum of cash dividends per bucket."""
    idx = schedule.index(divs.ex_times)
    out = np.zeros(schedule.size)
    np.add.at(out, idx[idx >= 0], divs.amounts[idx >= 0])
    return out


def _profile_weights(schedule: BucketSchedule, profile: Dividends):
    idx = schedule.index(profile.ex_times)
    totals = np.zeros(schedule.size)
    np.add.at(totals, idx[idx >= 0], profile.amounts[idx >= 0])
    return idx, totals


def allocate(schedule: BucketSchedule, levels, profile: Dividends) -> Dividends:
    """Spread bucket levels over the profile's ex-dates, pro rata to the profile amounts.

    A bucket without profile dividends puts its level at the bucket mid-point.
    """
    levels = np.asarray(levels, dtype=float)
    idx, totals = _profile_weights(schedule, profile)
    times, amounts = [], []
    for y in range(schedule.size):
        sel = idx == y
        if totals[y] > 0:
            times.append(profile.ex_times[sel])
            amounts.append(levels[y] * profile.amounts[sel] / totals[y])
        elif levels[y] != 0:
            times.append(np.array([schedule.mid[y]]))
            amounts.append(np.array([levels[y]]))
    if not times:
        return Dividends.empty()
    return Dividends(np.concatenate(times), np.concatenate(amounts))


def allocation_matrix(
    schedule: BucketSchedule,
    profile: Dividends,
    pillars,
    rates: ZeroCurve,
    repo: ZeroCurve,
) -> np.ndarray:
    """``A[k, y] = dD_k / dLevel_y``: carry-discounted share of bucket ``y`` ex before pillar ``k``."""
    T = np.asarray(pillars, dtype=float)
    idx, totals = _profile_weights(schedule, profile)
    A = np.zeros((T.size, schedule.size))
    for y in range(schedule.size):
        if totals[y] > 0:
            sel = idx == y
            t_i = profile.ex_times[sel]
            share = profile.amounts[sel] / totals[y]
        else:
            t_i = np.array([schedule.mid[y]])
            share = np.array([1.0])
        disc = np.exp(-(rates.cum(t_i) - repo.cum(t_i)))
        A[:, y] = ((t_i[None, :] <= T[:, None]) * (share * disc)[None, :]).sum(axis=1)
    return A
