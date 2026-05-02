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


@pytest.mark.parametrize(
    ("values", "offset", "expected"),
    [
        (np.array([10, 20], dtype=np.int64), 7, 17),
        (np.array([10, 20], dtype=np.int64), 2.5, 12.5),
        (np.array([1.25, 2.5], dtype=np.float64), 7, 8.25),
        (np.array([1.25, 2.5], dtype=np.float64), 2.5, 3.75),
    ],
)
def test_array_plus_scalar_arguments(values, offset, expected):
    @rumba.njit
    def add_offset(a, offset):
        return a[0] + offset

    assert add_offset(values, offset) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("offset", "values", "expected"),
    [
        (7, np.array([10, 20], dtype=np.int64), 17),
        (2.5, np.array([10, 20], dtype=np.int64), 12.5),
        (7, np.array([1.25, 2.5], dtype=np.float64), 8.25),
        (2.5, np.array([1.25, 2.5], dtype=np.float64), 3.75),
    ],
)
def test_scalar_plus_array_arguments(offset, values, expected):
    @rumba.njit
    def add_offset(offset, a):
        return offset + a[0]

    assert add_offset(offset, values) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("values", "replacement", "expected", "mutated"),
    [
        (np.array([1, 2, 3], dtype=np.int64), 99, 99, [1, 99, 3]),
        (np.array([1, 2, 3], dtype=np.int64), 9.5, 9, [1, 9, 3]),
        (np.array([1.0, 2.0, 3.0], dtype=np.float64), 99, 99.0, [1.0, 99.0, 3.0]),
        (np.array([1.0, 2.0, 3.0], dtype=np.float64), 9.5, 9.5, [1.0, 9.5, 3.0]),
    ],
)
def test_array_mutation_with_scalar_arguments(values, replacement, expected, mutated):
    @rumba.njit
    def set_item(a, value):
        a[1] = value
        return a[1]

    assert set_item(values, replacement) == pytest.approx(expected)
    assert values.tolist() == pytest.approx(mutated)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            np.array([10, 20], dtype=np.int64),
            np.array([7, 8], dtype=np.int64),
            17,
        ),
        (
            np.array([10, 20], dtype=np.int64),
            np.array([1.25, 2.5], dtype=np.float64),
            11.25,
        ),
        (
            np.array([1.25, 2.5], dtype=np.float64),
            np.array([7, 8], dtype=np.int64),
            8.25,
        ),
        (
            np.array([1.25, 2.5], dtype=np.float64),
            np.array([2.5, 5.0], dtype=np.float64),
            3.75,
        ),
    ],
)
def test_array_plus_array_arguments(left, right, expected):
    @rumba.njit
    def add_first(a, b):
        return a[0] + b[0]

    assert add_first(left, right) == pytest.approx(expected)


def test_mixed_array_scalar_inspect_c_signatures():
    @rumba.njit
    def array_then_scalar(a, x):
        return a[0] + x

    @rumba.njit
    def scalar_then_array(x, a):
        return x + a[0]

    @rumba.njit
    def arrays(a, b):
        return a[0] + b[0]

    ints = np.array([10, 20], dtype=np.int64)
    floats = np.array([1.25, 2.5], dtype=np.float64)

    assert array_then_scalar(ints, 2.5) == pytest.approx(12.5)
    assert scalar_then_array(2.5, floats) == pytest.approx(3.75)
    assert arrays(floats, ints) == pytest.approx(11.25)

    assert "double rumba_entry(rumba_array_i64 a, double x)" in array_then_scalar.inspect_c()
    assert "double rumba_entry(double x, rumba_array_f64 a)" in scalar_then_array.inspect_c()
    assert "double rumba_entry(rumba_array_f64 a, rumba_array_i64 b)" in arrays.inspect_c()
    assert "void rumba_call(void **args, void *ret)" in array_then_scalar.inspect_c()
    assert (
        "*(double *)ret = rumba_entry(*(rumba_array_i64 *)args[0], *(double *)args[1]);"
        in array_then_scalar.inspect_c()
    )
    assert (
        "*(double *)ret = rumba_entry(*(double *)args[0], *(rumba_array_f64 *)args[1]);"
        in scalar_then_array.inspect_c()
    )


def test_native_wrapper_handles_signature_without_rust_match_arm():
    @rumba.njit
    def float_array_len_plus_offset(a, offset):
        return len(a) + offset

    values = np.array([1.25, 2.5, 3.75], dtype=np.float64)

    assert float_array_len_plus_offset(values, 4) == 7
    c_source = float_array_len_plus_offset.inspect_c()
    assert "int64_t rumba_entry(rumba_array_f64 a, int64_t offset)" in c_source
    assert (
        "*(int64_t *)ret = rumba_entry(*(rumba_array_f64 *)args[0], *(int64_t *)args[1]);"
        in c_source
    )


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
    def set_item(a, value):
        a[0] = value
        return a[0]

    values = np.array([1, 2], dtype=np.int64)
    values.flags.writeable = False

    with pytest.raises(RumbaUnsupportedError, match="read-only"):
        set_item(values, 10)


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
