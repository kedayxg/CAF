"""A historical dependency override must not silently claim the new method."""
import sys

import pytest

import dcmppln.clustering as clustering
from experiments._common import load_upstream_components


def test_external_legacy_fallback_is_rejected(monkeypatch,tmp_path):
    class Legacy:
        def __init__(self,**kwargs):
            self.method="louvain"  # historical unknown-method fallback
    monkeypatch.setattr(clustering,"Clustering",Legacy)
    monkeypatch.setattr(sys,"path",sys.path.copy())
    with pytest.raises(ImportError,match="silently rejected 'newman'"):
        load_upstream_components(tmp_path)
