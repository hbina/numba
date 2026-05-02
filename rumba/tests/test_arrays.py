import numpy as np
import pytest

import rumba
from rumba import RumbaUnsupportedError


@rumba.njit
def _first(a):
    return a[0]


def test_sum_int64_array_with_len_and_indexing():
    @rumba.njit
    def total(a):
        acc = 0
        for i in range(len(a)):
            acc += a[i]
        return acc

    values = np.arange(6, dtype=np.int64)

    assert total(values) == 15
    assert total.signatures == [("array(int64, 1d, C)",)]


def test_sum_float64_array_with_len_and_indexing():
    @rumba.njit
    def total(a):
        acc = 0.0
        for i in range(len(a)):
            acc += a[i]
        return acc

    values = np.array([1.25, 2.5, 3.75], dtype=np.float64)

    assert total(values) == pytest.approx(7.5)


def test_mutate_int64_array_in_place():
    @rumba.njit
    def set_item(a, value):
        a[1] = value
        return a[1]

    values = np.array([1, 2, 3], dtype=np.int64)

    assert set_item(values, 99) == 99
    assert values.tolist() == [1, 99, 3]


def test_mutate_float64_array_in_place():
    @rumba.njit
    def set_item(a, value):
        a[2] = value
        return a[2]

    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert set_item(values, 9.5) == pytest.approx(9.5)
    assert values.tolist() == [1.0, 2.0, 9.5]


def test_mix_scalar_and_array_arguments():
    @rumba.njit
    def add_offset(a, offset):
        return a[0] + offset

    values = np.array([10, 20], dtype=np.int64)

    assert add_offset(values, 7) == 17


def test_helper_function_reads_array():
    @rumba.njit
    def use_helper(a):
        return _first(a) + a[1]

    values = np.array([4, 9], dtype=np.int64)

    assert use_helper(values) == 13
    assert "rumba_helper_0" in use_helper.inspect_c()


def test_len_builtin_intrinsic_returns_array_length():
    @rumba.njit
    def size(a):
        return len(a)

    values = np.array([4, 9, 16], dtype=np.int64)

    assert size(values) == 3
    call = size.inspect_typed_ast()["body"][0]["value"]
    assert call["kind"] == "IntrinsicCall"
    assert call["intrinsic"] == "len"
    assert call["type"] == "int64"


def test_numpy_reduction_intrinsics_for_int64_and_float64_arrays():
    @rumba.njit
    def int_reductions(a):
        return np.max(a) + np.min(a) + np.sum(a)

    @rumba.njit
    def float_reduction(a):
        return np.sum(a)

    ints = np.array([4, 1, 7], dtype=np.int64)
    floats = np.array([1.5, 2.25, 3.75], dtype=np.float64)

    assert int_reductions(ints) == 20
    assert float_reduction(floats) == pytest.approx(7.5)

    typed = int_reductions.inspect_typed_ast()
    value = typed["body"][0]["value"]
    assert value["left"]["left"]["kind"] == "IntrinsicCall"
    assert value["left"]["left"]["intrinsic"] == "numpy.max"
    assert value["left"]["right"]["intrinsic"] == "numpy.min"
    assert value["right"]["intrinsic"] == "numpy.sum"

    c_source = int_reductions.inspect_c()
    assert "static int64_t rumba_numpy_max_int64" in c_source
    assert "static int64_t rumba_numpy_min_int64" in c_source
    assert "static int64_t rumba_numpy_sum_int64" in c_source


def test_unsupported_numpy_constructor_call_raises():
    @rumba.njit
    def make_zeros(n):
        return np.zeros(n)

    with pytest.raises(RumbaUnsupportedError, match="attribute access"):
        make_zeros(3)


def test_shape_is_unsupported():
    @rumba.njit
    def use_shape(a):
        return a.shape[0]

    with pytest.raises(RumbaUnsupportedError, match="attribute access"):
        use_shape(np.array([1, 2], dtype=np.int64))


def test_2d_arrays_are_unsupported():
    @rumba.njit
    def first(a):
        return a[0]

    with pytest.raises(RumbaUnsupportedError, match="only 1D numpy arrays"):
        first(np.ones((2, 2), dtype=np.int64))


def test_non_contiguous_arrays_are_unsupported():
    @rumba.njit
    def first(a):
        return a[0]

    with pytest.raises(RumbaUnsupportedError, match="C-contiguous"):
        first(np.arange(6, dtype=np.int64)[::2])


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.bool_, object])
def test_unsupported_array_dtypes_raise(dtype):
    @rumba.njit
    def first(a):
        return a[0]

    with pytest.raises(RumbaUnsupportedError, match="unsupported numpy array dtype"):
        first(np.array([1, 2], dtype=dtype))


def test_list_arguments_are_not_array_arguments():
    @rumba.njit
    def first(a):
        return a[0]

    with pytest.raises(RumbaUnsupportedError, match="unsupported argument type"):
        first([1, 2, 3])


def test_array_returns_are_unsupported():
    @rumba.njit
    def identity(a):
        return a

    with pytest.raises(RumbaUnsupportedError, match="array return values"):
        identity(np.array([1, 2], dtype=np.int64))


def test_slicing_is_unsupported():
    @rumba.njit
    def head(a):
        return a[0:1]

    with pytest.raises(RumbaUnsupportedError, match="slicing"):
        head(np.array([1, 2], dtype=np.int64))


def test_tuple_indexing_is_unsupported():
    @rumba.njit
    def first(a):
        return a[0, 0]

    with pytest.raises(RumbaUnsupportedError):
        first(np.array([1, 2], dtype=np.int64))


def test_read_only_arrays_rejected_when_function_stores():
    @rumba.njit
    def set_item(a):
        a[0] = 10
        return a[0]

    values = np.array([1, 2], dtype=np.int64)
    values.flags.writeable = False

    with pytest.raises(RumbaUnsupportedError, match="read-only"):
        set_item(values)


def test_inspect_typed_ast_records_array_len_index_and_store_types():
    @rumba.njit
    def set_from_end(a):
        last = len(a) - 1
        a[0] = a[last]
        return a[0]

    values = np.array([1.25, 2.5], dtype=np.float64)

    assert set_from_end(values) == pytest.approx(2.5)
    typed = set_from_end.inspect_typed_ast()
    assign = typed["body"][0]
    store = typed["body"][1]
    ret = typed["body"][2]

    assert typed["locals"]["last"] == "int64"
    assert assign["target_type"] == "int64"
    assert assign["value"]["left"]["kind"] == "IntrinsicCall"
    assert assign["value"]["left"]["type"] == "int64"
    assert assign["value"]["left"]["intrinsic"] == "len"
    assert store["kind"] == "StoreIndex"
    assert store["element_type"] == "float64"
    assert store["value_type"] == "float64"
    assert store["value"]["kind"] == "Index"
    assert store["value"]["type"] == "float64"
    assert store["value"]["reason"] == "array_element"
    assert ret["value"]["kind"] == "Index"
    assert ret["value"]["type"] == "float64"
