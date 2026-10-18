"""A Semantic Scholar search with no hits omits "data"; that is an empty result, not a malformed response."""

import pytest

from hybrid.literature.sources import semantic_scholar as s2


def test_zero_hit_search_is_empty():
    assert s2.parse_search('{"total": 0, "offset": 0}', "2026-10-17T00:00:00Z") == []


def test_missing_data_with_hits_is_still_malformed():
    with pytest.raises(ValueError, match="no 'data' array"):
        s2.parse_search('{"total": 3, "offset": 0}', "2026-10-17T00:00:00Z")
