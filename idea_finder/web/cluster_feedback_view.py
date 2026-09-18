"""Cluster feedback backend: labels + hidden filter (T315).

Schema note (task 315 constraint): ``db/repo.py`` is shared with other
concurrent streams, so this module keeps the two feedback UPDATE helpers
and the hidden-filter listing here instead of editing repo.py. The columns
themselves come from migration ``006_cluster_feedback.sql``.
TODO(repo): move :func:`set_feedback`, :func:`set_split_flag` and the
``include_hidden`` listing support into ``db/repo.py`` once the concurrent
streams merge; keep the signatures and SQL as-is.

Semantics: ``feedback`` is a three-state label — ``'interesting'``,
``'hidden'`` or ``None`` (label cleared). ``set_split_flag`` marks «это не
одна боль» without any destructive action: the cluster stays intact and the
flag only records the calibration signal (re-splitting is the C-flow's
job). Hidden clusters stay out of the default list; the page exposes the
``include_hidden`` switch that brings them back.
"""

from __future__ import annotations

from typing import Final, Literal

from psycopg import Connection

#: Allowed feedback labels (mirrors the 006 CHECK constraint).
FeedbackLabel = Literal["interesting", "hidden"]

_FEEDBACK_LABELS: Final[frozenset[str]] = frozenset({"interesting", "hidden"})


def set_feedback(
    conn: Connection,
    cluster_id: str,
    feedback: FeedbackLabel | None,
) -> None:
    """Set (or clear) the feedback label of cluster ``cluster_id``.

    ``None`` clears the label. Unknown ids match no row (no error, same
    no-op contract as ``sources_view.set_source_enabled``); invalid labels
    are rejected client-side — the DB CHECK is the second line of defence.
    """
    if feedback is not None and feedback not in _FEEDBACK_LABELS:
        msg = f"unknown feedback label: {feedback!r}"
        raise ValueError(msg)
    conn.execute(
        "UPDATE cluster SET feedback = %s WHERE id = %s",
        (feedback, cluster_id),
    )


def set_split_flag(conn: Connection, cluster_id: str, flag: bool) -> None:
    """Mark cluster ``cluster_id`` as «это не одна боль» (or clear it).

    Web-layer write (see module docstring): flips ``cluster.split_flag``
    only — no deletion, no pain reassignment.
    """
    conn.execute(
        "UPDATE cluster SET split_flag = %s WHERE id = %s",
        (flag, cluster_id),
    )
