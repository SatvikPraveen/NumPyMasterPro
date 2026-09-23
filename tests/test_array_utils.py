"""
Unit tests for array_utils module
"""

import numpy as np
import pytest

from scripts.array_utils import (
    array_flags,
    array_summary_table,
    broadcast_result_shape,
    compare_arrays,
    create_identity_matrix,
    describe_array,
    explain_broadcast,
    flatten_or_ravel,
    generate_range,
    human_bytes,
    is_view_of,
)


class TestDescribeArray:
    """Tests for describe_array function"""

    def test_1d_array(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = describe_array(arr, verbose=False)

        assert result["shape"] == (5,)
        assert result["size"] == 5
        assert result["ndim"] == 1
        assert result["dtype"] == arr.dtype

    def test_2d_array(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        result = describe_array(arr, verbose=False)

        assert result["shape"] == (3, 2)
        assert result["size"] == 6
        assert result["ndim"] == 2

    def test_itemsize_calculation(self):
        arr_int32 = np.array([1, 2, 3], dtype=np.int32)
        result = describe_array(arr_int32, verbose=False)

        assert result["itemsize"] == 4  # int32 is 4 bytes
        assert result["nbytes"] == 12  # 3 elements * 4 bytes

    def test_view_detection_and_strides(self):
        base = np.arange(10)
        info = describe_array(base, verbose=False)
        assert info["is_view"] is False
        assert info["strides"] == (base.itemsize,)
        assert describe_array(base[2:5], verbose=False)["is_view"] is True

    def test_verbose_prints(self, capsys):
        describe_array(np.arange(3), verbose=True)
        assert "Array Summary" in capsys.readouterr().out


class TestArrayFlags:
    """Tests for array_flags function"""

    def test_c_contiguous(self):
        arr = np.array([[1, 2], [3, 4]])  # C-contiguous by default
        flags = array_flags(arr, verbose=False)

        assert flags["C_CONTIGUOUS"] is True

    def test_fortran_contiguous(self):
        arr = np.array([[1, 2], [3, 4]], order="F")
        flags = array_flags(arr, verbose=False)

        assert flags["F_CONTIGUOUS"] is True

    def test_writeable(self):
        arr = np.array([1, 2, 3])
        flags = array_flags(arr, verbose=False)

        assert flags["WRITEABLE"] is True

    def test_readonly_and_owndata(self):
        arr = np.arange(4)
        arr.setflags(write=False)
        flags = array_flags(arr, verbose=False)
        assert flags["WRITEABLE"] is False
        assert flags["OWNDATA"] is True
        assert array_flags(arr[1:], verbose=False)["OWNDATA"] is False


class TestFlattenOrRavel:
    """Tests for flatten_or_ravel function"""

    def test_flatten_creates_copy(self):
        arr = np.array([[1, 2], [3, 4]])
        result = flatten_or_ravel(arr, use_view=False)

        assert result.shape == (4,)
        # Modifying result should not affect original
        result[0] = 999
        assert arr[0, 0] == 1

    def test_ravel_creates_view(self):
        arr = np.array([[1, 2], [3, 4]])
        result = flatten_or_ravel(arr, use_view=True)

        assert result.shape == (4,)
        # Modifying result should affect original
        result[0] = 999
        assert arr[0, 0] == 999


class TestCompareArrays:
    """Tests for compare_arrays function"""

    def test_identical_arrays(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        result = compare_arrays(arr1, arr2)

        assert result["shape_equal"] is True
        assert result["dtype_equal"] is True
        assert result["elementwise_equal"] is True

    def test_different_shapes(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2])
        result = compare_arrays(arr1, arr2)

        assert result["shape_equal"] is False
        assert result["elementwise_equal"] is False

    def test_different_values(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 4])
        result = compare_arrays(arr1, arr2)

        assert result["shape_equal"] is True
        assert result["elementwise_equal"] is False

    def test_allclose_and_shared_memory(self):
        a = np.array([1.0, 2.0, 3.0])
        b = a + 1e-12
        result = compare_arrays(a, b)
        assert result["elementwise_equal"] is False
        assert result["allclose"] is True
        assert result["shares_memory"] is False
        assert compare_arrays(a, a[:])["shares_memory"] is True
        assert compare_arrays(a, np.array([1, 2]))["allclose"] is False


class TestViewsAndMemory:
    def test_is_view_of(self):
        base = np.arange(10)
        assert is_view_of(base[::2], base)
        assert not is_view_of(base.copy(), base)

    def test_human_bytes(self):
        assert human_bytes(512) == "512 B"
        assert human_bytes(1536) == "1.50 KiB"
        assert human_bytes(3 * 1024**2) == "3.00 MiB"
        assert human_bytes(2 * 1024**4) == "2.00 TiB"


class TestSummaryTable:
    def test_table_shape_and_names(self):
        df = array_summary_table(np.zeros(3), np.ones((2, 2)), names=["z", "o"])
        assert list(df["Name"]) == ["z", "o"]
        assert list(df["Size"]) == [3, 4]
        assert "Memory" in df.columns

    def test_default_names_and_validation(self):
        df = array_summary_table(np.zeros(1))
        assert df.loc[0, "Name"] == "Array 1"
        with pytest.raises(ValueError):
            array_summary_table(np.zeros(1), names=["a", "b"])


class TestBroadcasting:
    @pytest.mark.parametrize(
        "shapes",
        [((3, 1), (1, 4)), ((5,), (2, 5)), ((2, 3, 4), (4,)), ((1,), (1,)), ((), (3,))],
    )
    def test_matches_numpy(self, shapes):
        assert broadcast_result_shape(*shapes) == np.broadcast_shapes(*shapes)

    def test_incompatible(self):
        with pytest.raises(ValueError):
            broadcast_result_shape((3,), (4,))

    def test_explain_success_and_failure(self):
        text = explain_broadcast((3, 1), (4,))
        assert "result: (3, 4)" in text
        assert "stretch" in text
        bad = explain_broadcast((3,), (4,))
        assert "MISMATCH" in bad
        assert "incompatible" in bad


class TestCreateIdentityMatrix:
    """Tests for create_identity_matrix function"""

    def test_3x3_identity(self):
        result = create_identity_matrix(3)
        expected = np.eye(3)

        assert np.array_equal(result, expected)

    def test_5x5_identity(self):
        result = create_identity_matrix(5)

        assert result.shape == (5, 5)
        assert np.allclose(np.sum(result), 5.0)  # Sum of diagonal
        assert np.allclose(result[0, 0], 1.0)
        assert np.allclose(result[0, 1], 0.0)

    def test_negative_size(self):
        with pytest.raises(ValueError):
            create_identity_matrix(-1)


class TestGenerateRange:
    """Tests for generate_range function"""

    def test_basic_range(self):
        result = generate_range(0, 10, 2)
        expected = np.arange(0, 10, 2)

        assert np.array_equal(result, expected)

    def test_negative_step(self):
        result = generate_range(10, 0, -2)
        expected = np.arange(10, 0, -2)

        assert np.array_equal(result, expected)

    def test_float_range(self):
        result = generate_range(0.0, 1.0, 0.1)

        assert len(result) == 10
        assert np.allclose(result[0], 0.0)
        assert np.allclose(result[-1], 0.9)

    def test_zero_step(self):
        with pytest.raises(ValueError):
            generate_range(0, 5, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
