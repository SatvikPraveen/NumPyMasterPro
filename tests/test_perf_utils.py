"""
Unit tests for perf_utils module
"""

import numpy as np
import pytest

from scripts.perf_utils import (
    Timer,
    TimingResult,
    chunked_apply,
    iter_chunks,
    memory_footprint,
    rolling_windows,
    strided_blocks,
    timeit_compare,
)


class TestTimer:
    def test_measures_elapsed(self, capsys):
        with Timer("demo", verbose=True) as t:
            _ = np.arange(1000).sum()
        assert t.elapsed >= 0
        assert "demo" in capsys.readouterr().out

    def test_silent_by_default(self, capsys):
        with Timer():
            pass
        assert capsys.readouterr().out == ""


class TestTimeitCompare:
    def test_returns_results_for_each_function(self):
        x = np.arange(1000.0)
        res = timeit_compare(
            {"vec": lambda a: a * 2, "loop": lambda a: np.array([v * 2 for v in a])},
            x,
            number=2,
            repeat=2,
        )
        assert list(res) == ["vec", "loop"]
        for r in res.values():
            assert isinstance(r, TimingResult)
            assert r.best >= 0
            assert r.best <= r.mean + 1e-12
            assert r.best_ms == pytest.approx(r.best * 1e3)
            assert r.runs == 2

    def test_detects_mismatched_outputs(self):
        with pytest.raises(AssertionError):
            timeit_compare({"a": lambda v: v, "b": lambda v: v + 1}, np.ones(3), number=1, repeat=1)

    def test_check_equal_can_be_disabled(self):
        res = timeit_compare(
            {"a": lambda v: v, "b": lambda v: v + 1},
            np.ones(3),
            number=1,
            repeat=1,
            check_equal=False,
        )
        assert len(res) == 2

    def test_validation(self):
        with pytest.raises(ValueError):
            timeit_compare({})
        with pytest.raises(ValueError):
            timeit_compare({"a": lambda: 1}, number=0)


class TestMemoryFootprint:
    def test_owner_vs_view(self):
        base = np.zeros(100)
        view = base[::2]
        owner = memory_footprint(base)
        viewer = memory_footprint(view)
        assert owner["owns_data"] is True
        assert owner["owned"] == 800
        assert viewer["owns_data"] is False
        assert viewer["owned"] == 0
        assert viewer["base_nbytes"] == 800
        assert viewer["contiguous"] is False


class TestRollingWindows:
    def test_shape_and_values(self):
        w = rolling_windows(np.arange(6), 3)
        assert w.shape == (4, 3)
        np.testing.assert_array_equal(w[1], [1, 2, 3])
        assert not w.flags.writeable

    def test_step(self):
        w = rolling_windows(np.arange(10), 4, step=3)
        np.testing.assert_array_equal(w[:, 0], [0, 3, 6])

    def test_is_zero_copy(self):
        x = np.arange(6)
        assert np.shares_memory(rolling_windows(x, 2), x)

    def test_validation(self):
        with pytest.raises(ValueError):
            rolling_windows(np.zeros((2, 2)), 1)
        with pytest.raises(ValueError):
            rolling_windows(np.arange(3), 4)
        with pytest.raises(ValueError):
            rolling_windows(np.arange(3), 2, step=0)


class TestStridedBlocks:
    def test_block_means_match_reshape(self):
        img = np.arange(64.0).reshape(8, 8)
        blocks = strided_blocks(img, (2, 2))
        assert blocks.shape == (4, 4, 2, 2)
        expected = img.reshape(4, 2, 4, 2).mean(axis=(1, 3))
        np.testing.assert_array_equal(blocks.mean(axis=(2, 3)), expected)
        np.testing.assert_array_equal(blocks[1, 2], img[2:4, 4:6])

    def test_zero_copy_and_readonly(self):
        img = np.zeros((4, 4))
        blocks = strided_blocks(img, (2, 2))
        assert np.shares_memory(blocks, img)
        assert not blocks.flags.writeable

    def test_validation(self):
        with pytest.raises(ValueError):
            strided_blocks(np.zeros(4), (2, 2))
        with pytest.raises(ValueError):
            strided_blocks(np.zeros((5, 4)), (2, 2))
        with pytest.raises(ValueError):
            strided_blocks(np.zeros((4, 4)), (0, 2))


class TestChunking:
    def test_iter_chunks_covers_array(self):
        x = np.arange(10)
        chunks = list(iter_chunks(x, 4))
        assert [c.tolist() for c in chunks] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        assert all(np.shares_memory(c, x) for c in chunks)

    def test_iter_chunks_axis(self):
        x = np.zeros((3, 7))
        assert [c.shape for c in iter_chunks(x, 3, axis=1)] == [(3, 3), (3, 3), (3, 1)]

    def test_chunked_apply_equals_full(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(23, 4))
        full = X @ X[:5].T
        out = chunked_apply(lambda c: c @ X[:5].T, X, chunk_size=6)
        np.testing.assert_allclose(out, full)

    def test_chunked_apply_empty(self):
        out = chunked_apply(lambda c: c * 2, np.zeros((0, 3)), chunk_size=2)
        assert out.shape == (0, 3)

    def test_validation(self):
        with pytest.raises(ValueError):
            list(iter_chunks(np.arange(3), 0))
