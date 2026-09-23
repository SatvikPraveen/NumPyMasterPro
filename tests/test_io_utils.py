"""
Unit tests for io_utils module (all file I/O goes through pytest's tmp_path)
"""

import numpy as np
import pytest

from scripts.io_utils import (
    create_memmap,
    load_genfromtxt,
    load_memmap,
    load_npy,
    load_npz,
    load_txt,
    save_npy,
    save_npz,
    save_txt,
    write_csv_with_missing_values,
)


class TestNpy:
    def test_roundtrip(self, tmp_path):
        arr = np.arange(12).reshape(3, 4)
        p = save_npy(tmp_path / "a.npy", arr)
        assert p.exists()
        np.testing.assert_array_equal(load_npy(p), arr)

    def test_extension_is_appended(self, tmp_path):
        p = save_npy(tmp_path / "noext", [1, 2, 3])
        assert p.name == "noext.npy"
        assert p.exists()

    def test_creates_parent_dirs(self, tmp_path):
        p = save_npy(tmp_path / "nested" / "deep" / "a.npy", np.ones(2))
        assert p.exists()

    def test_mmap_mode(self, tmp_path):
        p = save_npy(tmp_path / "big.npy", np.arange(100.0))
        m = load_npy(p, mmap_mode="r")
        assert isinstance(m, np.memmap)
        assert m[50] == 50.0


class TestNpz:
    def test_roundtrip_lazy_and_dict(self, tmp_path):
        p = save_npz(tmp_path / "multi.npz", x=np.arange(3), y=np.eye(2))
        lazy = load_npz(p)
        assert sorted(lazy.files) == ["x", "y"]
        np.testing.assert_array_equal(lazy["x"], [0, 1, 2])
        lazy.close()
        eager = load_npz(p, as_dict=True)
        assert isinstance(eager, dict)
        np.testing.assert_array_equal(eager["y"], np.eye(2))

    def test_compressed(self, tmp_path):
        data = np.zeros(10_000)
        p_plain = save_npz(tmp_path / "plain.npz", d=data)
        p_comp = save_npz(tmp_path / "comp.npz", compressed=True, d=data)
        assert p_comp.stat().st_size < p_plain.stat().st_size
        np.testing.assert_array_equal(load_npz(p_comp, as_dict=True)["d"], data)

    def test_requires_arrays(self, tmp_path):
        with pytest.raises(ValueError):
            save_npz(tmp_path / "empty.npz")


class TestText:
    def test_txt_roundtrip(self, tmp_path):
        arr = np.array([[1.5, 2.5], [3.5, 4.5]])
        p = save_txt(tmp_path / "a.csv", arr)
        np.testing.assert_allclose(load_txt(p), arr)

    def test_txt_with_header(self, tmp_path):
        p = save_txt(tmp_path / "h.csv", np.eye(2), fmt="%.1f", header="a,b")
        assert p.read_text().splitlines()[0] == "a,b"
        np.testing.assert_allclose(load_txt(p, skiprows=1), np.eye(2))

    def test_genfromtxt_missing_values(self, tmp_path):
        p = write_csv_with_missing_values(tmp_path / "missing.csv")
        arr = load_genfromtxt(p)
        assert arr.shape == (3, 3)
        assert np.isnan(arr[1, 1])
        filled = load_genfromtxt(p, filling_values=0.0)
        assert filled[1, 1] == 0.0


class TestMemmap:
    def test_random_fill_and_reload(self, tmp_path):
        p = tmp_path / "m.dat"
        m = create_memmap(p, shape=(4, 5), dtype="float32", seed=0)
        assert m.shape == (4, 5)
        assert ((m >= 0) & (m < 1)).all()
        del m
        r = load_memmap(p, shape=(4, 5), dtype="float32")
        assert r.shape == (4, 5)
        assert not r.flags.writeable

    def test_explicit_fill(self, tmp_path):
        p = tmp_path / "f.dat"
        create_memmap(p, shape=(2, 2), dtype="int32", fill=[[1, 2], [3, 4]])
        np.testing.assert_array_equal(load_memmap(p, shape=(2, 2), dtype="int32"), [[1, 2], [3, 4]])

    def test_seed_reproducible(self, tmp_path):
        a = np.array(create_memmap(tmp_path / "a.dat", shape=(3,), seed=1))
        b = np.array(create_memmap(tmp_path / "b.dat", shape=(3,), seed=1))
        np.testing.assert_array_equal(a, b)
