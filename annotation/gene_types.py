"""Shared gene-type labels for GENCODE classification outputs.

Public ``gene_type_grouped`` stores the split lncRNA labels
(``novel_lncRNA`` / ``existing_lncRNA``). Analyses that want a single lncRNA
class should collapse with :func:`collapse_lncrna_grouped`.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

NOVEL_LNCRNA = "novel_lncRNA"
EXISTING_LNCRNA = "existing_lncRNA"
LNCRNA_GROUPED = frozenset({NOVEL_LNCRNA, EXISTING_LNCRNA})

_COLLAPSE = {NOVEL_LNCRNA: "lncRNA", EXISTING_LNCRNA: "lncRNA"}


def collapse_lncrna_grouped(values) -> pd.Series | str | None:
    """Map novel/existing lncRNA → ``lncRNA``; leave other labels unchanged."""
    if isinstance(values, pd.Series):
        return values.map(lambda x: _COLLAPSE.get(x, x))
    if values is None or (isinstance(values, float) and pd.isna(values)):
        return values
    s = str(values)
    return _COLLAPSE.get(s, s)


def is_lncrna_grouped(values) -> pd.Series | bool:
    """True for ``novel_lncRNA`` or ``existing_lncRNA``."""
    if isinstance(values, pd.Series):
        return values.isin(LNCRNA_GROUPED)
    return str(values) in LNCRNA_GROUPED


def collapse_lncrna_in_iterable(types: Iterable[str]) -> set[str]:
    """Collapse set/list membership labels the same way."""
    return {collapse_lncrna_grouped(t) for t in types}
