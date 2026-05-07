import numpy as np
import pytest

import rumba
from rumba import RumbaUnsupportedError

PCAP_HEADER_DTYPE = np.dtype(
    [
        ("ts_sec", np.uint32),
        ("ts_usec", np.uint32),
        ("incl_len", np.uint32),
        ("orig_len", np.uint32),
    ]
)
PACKET_VIEW_DTYPE = np.dtype(
    [
        ("captured_len", np.uint32),
        ("total_length", np.uint32),
        ("seq", np.uint64),
    ]
)
ALT_PACKET_VIEW_DTYPE = np.dtype(
    [
        ("captured_len", np.uint32),
        ("total_length", np.uint32),
        ("flags", np.uint64),
    ]
)
INT64_DTYPE = np.dtype("int64")
UNALIGNED_HEADER_DTYPE = np.dtype(
    {
        "names": ["pad", "incl_len"],
        "formats": ["u1", "u4"],
        "offsets": [0, 1],
        "itemsize": 5,
    }
)


@rumba.njit
def _first(a):
    return a[0]


@rumba.njit
def _yield_array(a):
    yield a


@rumba.njit
def _yield_array_alias(a):
    values = a
    yield values


@rumba.njit
def _yield_packet_views_named(buf, limit):
    offset = 0
    while offset < limit:
        packet = np.frombuffer(buf, PACKET_VIEW_DTYPE, 1, offset)
        yield packet
        offset += packet[0]["total_length"]


@rumba.njit
def _yield_packet_views_direct(buf, limit):
    offset = 0
    while offset < limit:
        yield np.frombuffer(buf, PACKET_VIEW_DTYPE, 1, offset)
        offset += 16


@rumba.njit
def _yield_int64_views(buf, limit):
    offset = 0
    while offset < limit:
        yield np.frombuffer(buf, INT64_DTYPE, 1, offset)
        offset += 8


@rumba.njit
def _yield_scalar_then_view(buf):
    yield 1
    yield np.frombuffer(buf, INT64_DTYPE, 1, 0)


@rumba.njit
def _yield_mixed_struct_views(buf):
    yield np.frombuffer(buf, PACKET_VIEW_DTYPE, 1, 0)
    yield np.frombuffer(buf, ALT_PACKET_VIEW_DTYPE, 1, 0)


@rumba.njit
def _yield_bad_int64_view(buf):
    yield np.frombuffer(buf, INT64_DTYPE, 1, 8)


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


def test_generator_yielding_array_is_rejected():
    @rumba.njit
    def total(a):
        acc = 0
        for value in _yield_array(a):
            acc += len(value)
        return acc

    values = np.arange(3, dtype=np.int64)
    with pytest.raises(RumbaUnsupportedError, match="np.frombuffer views"):
        total(values)


def test_generator_yielding_non_frombuffer_array_local_is_rejected():
    @rumba.njit
    def total(a):
        acc = 0
        for value in _yield_array_alias(a):
            acc += len(value)
        return acc

    values = np.arange(3, dtype=np.int64)
    with pytest.raises(RumbaUnsupportedError, match="np.frombuffer views"):
        total(values)


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


def test_while_mutates_int64_array_in_place():
    @rumba.njit
    def fill_prefix(a, n):
        i = 0
        while i < n:
            a[i] = i + 10
            i += 1
        return a[n - 1]

    values = np.zeros(5, dtype=np.int64)

    assert fill_prefix(values, 3) == 12
    assert values.tolist() == [10, 11, 12, 0, 0]


@pytest.mark.parametrize(
    ("values", "offset", "expected"),
    [
        (np.array([10, 20], dtype=np.int64), 7, 17),
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
            np.array([7, 8], dtype=np.int64),
            17,
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


def test_explicit_cast_array_scalar_inspect_c_signatures():
    @rumba.njit
    def array_then_scalar(a, x):
        return float(a[0]) + x

    @rumba.njit
    def scalar_then_array(x, a):
        return x + float(a[0])

    @rumba.njit
    def arrays(a, b):
        return a[0] + float(b[0])

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


@pytest.mark.parametrize(
    ("dtype", "field", "expected"),
    [
        (np.dtype([("x", np.bool_)]), "x", True),
        (np.dtype([("x", np.int8)]), "x", -3),
        (np.dtype([("x", np.int16)]), "x", -300),
        (np.dtype([("x", np.int32)]), "x", -30_000),
        (np.dtype([("x", np.int64)]), "x", -3_000_000_000),
        (np.dtype([("x", np.uint8)]), "x", 3),
        (np.dtype([("x", np.uint16)]), "x", 300),
        (np.dtype([("x", np.uint32)]), "x", 30_000),
        (np.dtype([("x", np.uint64)]), "x", 3_000_000_000),
        (np.dtype([("x", np.float32)]), "x", 1.5),
        (np.dtype([("x", np.float64)]), "x", 2.5),
    ],
)
def test_structured_array_field_reads(dtype, field, expected):
    @rumba.njit
    def first_field(a):
        return a[0]["x"]

    values = np.array([(expected,)], dtype=dtype)

    assert first_field(values) == pytest.approx(expected)


def test_structured_array_field_sum_with_len_and_scalar_argument():
    @rumba.njit
    def weighted(a, scale):
        total = 0.0
        for i in range(len(a)):
            total += float(a[i]["count"]) * scale + a[i]["weight"]
        return total

    dtype = np.dtype([("count", np.uint64), ("weight", np.float64)])
    values = np.array([(1, 1.25), (2, 2.5), (3, 3.75)], dtype=dtype)

    assert weighted(values, 2.5) == pytest.approx(22.5)


def test_structured_array_field_mutation_updates_numpy_buffer():
    @rumba.njit
    def set_field(a, value):
        a[1]["score"] = value
        return a[1]["score"]

    values = np.array([(1, 1.25), (2, 2.5)], dtype=[("id", np.int64), ("score", np.float64)])

    assert set_field(values, 9) == pytest.approx(9.0)
    assert values["score"].tolist() == [1.25, 9.0]


def test_read_only_structured_arrays_rejected_when_function_stores():
    @rumba.njit
    def set_field(a, value):
        a[0]["x"] = value
        return a[0]["x"]

    values = np.array([(1,), (2,)], dtype=[("x", np.int64)])
    values.flags.writeable = False

    with pytest.raises(RumbaUnsupportedError, match="read-only"):
        set_field(values, 10)


def test_structured_array_inspect_c_and_typed_ast():
    @rumba.njit
    def copy_field(a):
        a[1]["value"] = a[0]["value"]
        return a[1]["value"]

    dtype = np.dtype(
        {
            "names": ["tag", "value"],
            "formats": ["u1", "f8"],
            "offsets": [0, 8],
            "itemsize": 16,
        }
    )
    values = np.array([(4, 1.5), (5, 2.5)], dtype=dtype)

    assert copy_field(values) == pytest.approx(1.5)

    c_source = copy_field.inspect_c()
    assert "#include <stddef.h>" in c_source
    assert "typedef struct rumba_struct_tag_u8_value_f64" in c_source
    assert "uint8_t _pad0[7];" in c_source
    assert "_Static_assert(sizeof(rumba_struct_tag_u8_value_f64) == 16" in c_source
    assert "_Static_assert(offsetof(rumba_struct_tag_u8_value_f64, value) == 8" in c_source
    assert "a.data[1].value = a.data[0].value;" in c_source

    typed = copy_field.inspect_typed_ast()
    store = typed["body"][0]
    ret = typed["body"][1]
    assert store["kind"] == "StoreIndexField"
    assert store["field"] == "value"
    assert store["field_type"] == "float64"
    assert store["value"]["kind"] == "IndexField"
    assert ret["value"]["kind"] == "IndexField"
    assert ret["value"]["field"] == "value"


def test_while_reads_and_writes_structured_array_fields():
    @rumba.njit
    def scale_counts(a, n):
        i = 0
        total = 0.0
        while i < n:
            a[i]["weight"] = float(a[i]["count"]) * 2.0
            total += a[i]["weight"]
            i += 1
        return total

    values = np.array(
        [(1, 0.0), (2, 0.0), (3, 0.0)],
        dtype=np.dtype([("count", np.uint64), ("weight", np.float64)]),
    )

    assert scale_counts(values, 3) == pytest.approx(12.0)
    assert values["weight"].tolist() == pytest.approx([2.0, 4.0, 6.0])


def test_numpy_frombuffer_reads_structured_header_from_uint8_buffer():
    @rumba.njit
    def incl_len(buf):
        header = np.frombuffer(buf, PCAP_HEADER_DTYPE, 1, 0)
        return header[0]["incl_len"]

    records = np.array([(1, 2, 64, 128)], dtype=PCAP_HEADER_DTYPE)
    buf = records.view(np.uint8)

    assert incl_len(buf) == 64


def test_numpy_frombuffer_reads_structured_header_from_uint8_memmap(tmp_path):
    @rumba.njit
    def orig_len(buf):
        header = np.frombuffer(buf, PCAP_HEADER_DTYPE, 1, 0)
        return header[0]["orig_len"]

    path = tmp_path / "packet.bin"
    records = np.array([(1, 2, 64, 128)], dtype=PCAP_HEADER_DTYPE)
    path.write_bytes(records.view(np.uint8).tobytes())
    buf = np.memmap(path, dtype=np.uint8, mode="r")

    assert orig_len(buf) == 128


def test_numpy_frombuffer_dynamic_offsets_parse_multiple_records():
    @rumba.njit
    def incl_len_at(buf, index):
        offset = index * 16
        header = np.frombuffer(buf, PCAP_HEADER_DTYPE, 1, offset)
        return header[0]["incl_len"]

    records = np.array([(1, 2, 64, 128), (3, 4, 256, 512)], dtype=PCAP_HEADER_DTYPE)
    buf = records.view(np.uint8)

    assert incl_len_at(buf, 0) == 64
    assert incl_len_at(buf, 1) == 256


def test_generator_yields_named_structured_frombuffer_views_to_loop():
    @rumba.njit
    def captured_total(buf, limit):
        total = 0
        for packet in _yield_packet_views_named(buf, limit):
            total += packet[0]["captured_len"]
        return total

    records = np.array([(64, 16, 1), (128, 16, 2), (256, 16, 3)], dtype=PACKET_VIEW_DTYPE)
    buf = records.view(np.uint8)

    assert captured_total(buf, len(buf)) == 448
    typed = captured_total.inspect_typed_ast()
    assert typed["body"][1]["kind"] == "ForGenerator"
    assert (
        typed["body"][1]["target_type"]
        == "array(struct{captured_len:u32,total_length:u32,seq:u64}, 1d, C)"
    )


def test_generator_yields_direct_structured_frombuffer_views_to_loop():
    @rumba.njit
    def seq_total(buf, limit):
        total = 0
        for packet in _yield_packet_views_direct(buf, limit):
            total += packet[0]["seq"]
        return total

    records = np.array([(64, 16, 1), (128, 16, 2), (256, 16, 3)], dtype=PACKET_VIEW_DTYPE)
    buf = records.view(np.uint8)

    assert seq_total(buf, len(buf)) == 6


def test_generator_yields_scalar_frombuffer_views_to_loop():
    @rumba.njit
    def value_total(buf, limit):
        total = 0
        for values in _yield_int64_views(buf, limit):
            total += values[0]
        return total

    values = np.array([10, 20, 30], dtype=np.int64)
    buf = values.view(np.uint8)

    assert value_total(buf, len(buf)) == 60
    typed = value_total.inspect_typed_ast()
    assert typed["body"][1]["target_type"] == "array(int64, 1d, C)"


def test_generator_mixed_scalar_and_frombuffer_view_yields_raise():
    @rumba.njit
    def total(buf):
        acc = 0
        for value in _yield_scalar_then_view(buf):
            acc += len(value)
        return acc

    buf = np.zeros(8, dtype=np.uint8)

    with pytest.raises(RumbaUnsupportedError, match="generator yield values require exact matching"):
        total(buf)


def test_generator_mixed_structured_frombuffer_view_yields_raise():
    @rumba.njit
    def total(buf):
        acc = 0
        for packet in _yield_mixed_struct_views(buf):
            acc += packet[0]["captured_len"]
        return acc

    buf = np.zeros(16, dtype=np.uint8)

    with pytest.raises(RumbaUnsupportedError, match="generator yield values require exact matching"):
        total(buf)


def test_generator_frombuffer_runtime_error_skips_consumer_body():
    @rumba.njit
    def consume_bad_view(buf, marker):
        total = 0
        for values in _yield_bad_int64_view(buf):
            marker[0] = 1
            total += values[0]
        return total

    buf = np.zeros(8, dtype=np.uint8)
    marker = np.zeros(1, dtype=np.int64)

    with pytest.raises(RumbaUnsupportedError, match="offset/count"):
        consume_bad_view(buf, marker)
    assert marker[0] == 0


def test_numpy_frombuffer_inspects_as_intrinsic_and_pointer_cast_view():
    @rumba.njit
    def incl_len(buf):
        header = np.frombuffer(buf, PCAP_HEADER_DTYPE, 1, 0)
        return header[0]["incl_len"]

    records = np.array([(1, 2, 64, 128)], dtype=PCAP_HEADER_DTYPE)

    assert incl_len(records.view(np.uint8)) == 64
    typed = incl_len.inspect_typed_ast()
    assign = typed["body"][0]
    assert assign["value"]["kind"] == "IntrinsicCall"
    assert assign["value"]["intrinsic"] == "numpy.frombuffer"
    assert assign["target_type"] == "array(struct{ts_sec:u32,ts_usec:u32,incl_len:u32,orig_len:u32}, 1d, C)"

    c_source = incl_len.inspect_c()
    assert "static rumba_array_rumba_struct_ts_sec_u32_ts_usec_u32_incl_len_u32_orig_len_u32 rumba_numpy_frombuffer_" in c_source
    assert "(rumba_struct_ts_sec_u32_ts_usec_u32_incl_len_u32_orig_len_u32 *)(buf.data + offset)" in c_source
    assert "offset < 0 || count < 0" in c_source


def test_numpy_frombuffer_rejects_keyword_form():
    @rumba.njit
    def incl_len(buf):
        header = np.frombuffer(buf, dtype=PCAP_HEADER_DTYPE, count=1, offset=0)
        return header[0]["incl_len"]

    buf = np.zeros(16, dtype=np.uint8)

    with pytest.raises(RumbaUnsupportedError, match="keyword calls"):
        incl_len(buf)


def test_numpy_frombuffer_rejects_inline_dtype_constructor():
    @rumba.njit
    def incl_len(buf):
        header = np.frombuffer(buf, np.dtype([("incl_len", np.uint32)]), 1, 0)
        return header[0]["incl_len"]

    buf = np.zeros(4, dtype=np.uint8)

    with pytest.raises(RumbaUnsupportedError):
        incl_len(buf)


def test_numpy_frombuffer_rejects_non_uint8_source_buffer():
    @rumba.njit
    def first_value(buf):
        values = np.frombuffer(buf, INT64_DTYPE, 1, 0)
        return values[0]

    with pytest.raises(RumbaUnsupportedError, match="source must be a 1D uint8"):
        first_value(np.zeros(2, dtype=np.int64))


@pytest.mark.parametrize(("count", "offset"), [(-1, 0), (1, -1), (2, 0), (1, 8)])
def test_numpy_frombuffer_rejects_negative_or_out_of_bounds_views(count, offset):
    @rumba.njit
    def incl_len(buf, count, offset):
        header = np.frombuffer(buf, PCAP_HEADER_DTYPE, count, offset)
        return header[0]["incl_len"]

    buf = np.zeros(16, dtype=np.uint8)

    with pytest.raises(RumbaUnsupportedError, match="offset/count"):
        incl_len(buf, count, offset)


def test_numpy_frombuffer_rejects_unaligned_structured_dtype():
    @rumba.njit
    def read_value(buf):
        header = np.frombuffer(buf, UNALIGNED_HEADER_DTYPE, 1, 0)
        return header[0]["incl_len"]

    buf = np.zeros(5, dtype=np.uint8)

    with pytest.raises(RumbaUnsupportedError, match="alignment"):
        read_value(buf)


def test_numpy_frombuffer_returned_view_is_unsupported():
    @rumba.njit
    def view(buf):
        return np.frombuffer(buf, PCAP_HEADER_DTYPE, 1, 0)

    buf = np.zeros(16, dtype=np.uint8)

    with pytest.raises(RumbaUnsupportedError, match="array return values"):
        view(buf)


def test_numpy_frombuffer_assignment_through_view_is_unsupported():
    @rumba.njit
    def mutate(buf):
        header = np.frombuffer(buf, PCAP_HEADER_DTYPE, 1, 0)
        header[0]["incl_len"] = 9
        return header[0]["incl_len"]

    buf = np.zeros(16, dtype=np.uint8)

    with pytest.raises(RumbaUnsupportedError, match="assigning through np.frombuffer"):
        mutate(buf)


def test_structured_array_bare_record_read_is_unsupported():
    @rumba.njit
    def first_record(a):
        return a[0]

    values = np.array([(1,)], dtype=[("x", np.int64)])

    with pytest.raises(RumbaUnsupportedError, match=r"a\[i\]\['field'\]"):
        first_record(values)


def test_structured_array_dot_field_is_unsupported():
    @rumba.njit
    def dot_field(a):
        return a[0].x

    values = np.array([(1,)], dtype=[("x", np.int64)])

    with pytest.raises(RumbaUnsupportedError, match="attribute access"):
        dot_field(values)


def test_structured_array_missing_field_is_unsupported():
    @rumba.njit
    def missing(a):
        return a[0]["missing"]

    values = np.array([(1,)], dtype=[("x", np.int64)])

    with pytest.raises(RumbaUnsupportedError, match="does not exist"):
        missing(values)


def test_structured_array_non_string_field_key_is_unsupported():
    @rumba.njit
    def non_string(a):
        return a[0][0]

    values = np.array([(1,)], dtype=[("x", np.int64)])

    with pytest.raises(RumbaUnsupportedError, match=r"a\[i\]\['field'\]"):
        non_string(values)


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype([("bad-name", np.int64)]),
        np.dtype([("x", object)]),
        np.dtype([("x", np.complex128)]),
        np.dtype([("x", [("y", np.int64)])]),
        np.dtype([("x", np.int64, (2,))]),
        np.dtype(
            {
                "names": ["a", "b"],
                "formats": ["u1", "u8"],
                "offsets": [0, 1],
                "itemsize": 9,
            }
        ),
    ],
)
def test_unsupported_structured_dtypes_raise(dtype):
    @rumba.njit
    def first_field(a):
        return a[0]["x"]

    values = np.zeros(2, dtype=dtype)

    with pytest.raises(RumbaUnsupportedError):
        first_field(values)


def test_structured_array_2d_and_non_contiguous_are_unsupported():
    @rumba.njit
    def first_field(a):
        return a[0]["x"]

    dtype = np.dtype([("x", np.int64)])

    with pytest.raises(RumbaUnsupportedError, match="only 1D"):
        first_field(np.zeros((2, 2), dtype=dtype))

    with pytest.raises(RumbaUnsupportedError, match="C-contiguous"):
        first_field(np.zeros(4, dtype=dtype)[::2])


def test_numpy_reductions_on_structured_arrays_are_unsupported():
    @rumba.njit
    def sum_records(a):
        return np.sum(a)

    values = np.array([(1,), (2,)], dtype=[("x", np.int64)])

    with pytest.raises(RumbaUnsupportedError, match="structured arrays"):
        sum_records(values)
