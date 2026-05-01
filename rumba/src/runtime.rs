use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::artifact::CompiledArtifact;
use crate::errors::{compilation, unsupported};
use crate::types::{RumbaType, ScalarType};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ArrayI64View {
    data: *mut i64,
    len: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ArrayF64View {
    data: *mut f64,
    len: i64,
}

#[derive(Clone, Copy, Debug)]
enum NativeArg {
    I64(i64),
    F64(f64),
    Bool(bool),
    ArrayI64(ArrayI64View),
    ArrayF64(ArrayF64View),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AbiKind {
    I64,
    F64,
    Bool,
    ArrayI64,
    ArrayF64,
}

pub(crate) fn call_native(
    py: Python<'_>,
    artifact: &CompiledArtifact,
    args: &Bound<'_, PyTuple>,
    debug: bool,
) -> PyResult<PyObject> {
    let prepared = artifact
        .signature
        .iter()
        .zip(args.iter())
        .map(|(typ, arg)| prepare_arg(typ, &arg, artifact.requires_writable_arrays))
        .collect::<PyResult<Vec<_>>>()?;
    let kinds = artifact
        .signature
        .iter()
        .map(abi_kind)
        .collect::<PyResult<Vec<_>>>()?;
    if debug {
        eprintln!(
            "[rumba-debug] runtime: artifact signature: [{}]",
            crate::types::format_signature(&artifact.signature)
        );
        eprintln!(
            "[rumba-debug] runtime: return type: {}",
            artifact.return_type.name()
        );
        eprintln!("[rumba-debug] runtime: ABI kinds: {kinds:?}");
        eprintln!("[rumba-debug] runtime: prepared args: {prepared:?}");
        eprintln!(
            "[rumba-debug] runtime: library path: {}",
            artifact.library_path.display()
        );
    }

    unsafe {
        match artifact.return_type {
            ScalarType::Int64 => {
                call_i64(artifact, &kinds, &prepared, debug).map(|value| value.into_py(py))
            }
            ScalarType::Float64 => {
                call_f64(artifact, &kinds, &prepared, debug).map(|value| value.into_py(py))
            }
            ScalarType::Bool => {
                call_bool(artifact, &kinds, &prepared, debug).map(|value| value.into_py(py))
            }
        }
    }
}

unsafe fn call_i64(
    artifact: &CompiledArtifact,
    kinds: &[AbiKind],
    args: &[NativeArg],
    debug: bool,
) -> PyResult<i64> {
    match kinds {
        [AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0)))
        }
        [AbiKind::Bool] => {
            let f: libloading::Symbol<unsafe extern "C" fn(bool) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_bool(args, 0)))
        }
        [AbiKind::ArrayI64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(ArrayI64View) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_array_i64(args, 0)))
        }
        [AbiKind::ArrayF64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(ArrayF64View) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_array_f64(args, 0)))
        }
        [AbiKind::I64, AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0), arg_i64(args, 1)))
        }
        [AbiKind::I64, AbiKind::Bool] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64, bool) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0), arg_bool(args, 1)))
        }
        [AbiKind::Bool, AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(bool, i64) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_bool(args, 0), arg_i64(args, 1)))
        }
        [AbiKind::Bool, AbiKind::Bool] => {
            let f: libloading::Symbol<unsafe extern "C" fn(bool, bool) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_bool(args, 0), arg_bool(args, 1)))
        }
        [AbiKind::ArrayI64, AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(ArrayI64View, i64) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_array_i64(args, 0), arg_i64(args, 1)))
        }
        [AbiKind::I64, AbiKind::ArrayI64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64, ArrayI64View) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0), arg_array_i64(args, 1)))
        }
        [AbiKind::I64, AbiKind::I64, AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64, i64, i64) -> i64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0), arg_i64(args, 1), arg_i64(args, 2)))
        }
        _ => unsupported_native_signature("int64", kinds, debug),
    }
}

unsafe fn call_f64(
    artifact: &CompiledArtifact,
    kinds: &[AbiKind],
    args: &[NativeArg],
    debug: bool,
) -> PyResult<f64> {
    match kinds {
        [AbiKind::F64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(f64) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_f64(args, 0)))
        }
        [AbiKind::ArrayF64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(ArrayF64View) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_array_f64(args, 0)))
        }
        [AbiKind::F64, AbiKind::F64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(f64, f64) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_f64(args, 0), arg_f64(args, 1)))
        }
        [AbiKind::I64, AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0), arg_i64(args, 1)))
        }
        [AbiKind::I64, AbiKind::F64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64, f64) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0), arg_f64(args, 1)))
        }
        [AbiKind::F64, AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(f64, i64) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_f64(args, 0), arg_i64(args, 1)))
        }
        [AbiKind::ArrayF64, AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(ArrayF64View, i64) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_array_f64(args, 0), arg_i64(args, 1)))
        }
        [AbiKind::ArrayF64, AbiKind::F64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(ArrayF64View, f64) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_array_f64(args, 0), arg_f64(args, 1)))
        }
        [AbiKind::F64, AbiKind::ArrayF64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(f64, ArrayF64View) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_f64(args, 0), arg_array_f64(args, 1)))
        }
        [AbiKind::F64, AbiKind::F64, AbiKind::F64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(f64, f64, f64) -> f64> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_f64(args, 0), arg_f64(args, 1), arg_f64(args, 2)))
        }
        _ => unsupported_native_signature("float64", kinds, debug),
    }
}

unsafe fn call_bool(
    artifact: &CompiledArtifact,
    kinds: &[AbiKind],
    args: &[NativeArg],
    debug: bool,
) -> PyResult<bool> {
    match kinds {
        [AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0)))
        }
        [AbiKind::F64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(f64) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_f64(args, 0)))
        }
        [AbiKind::Bool] => {
            let f: libloading::Symbol<unsafe extern "C" fn(bool) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_bool(args, 0)))
        }
        [AbiKind::I64, AbiKind::F64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64, f64) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0), arg_f64(args, 1)))
        }
        [AbiKind::I64, AbiKind::Bool] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64, bool) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0), arg_bool(args, 1)))
        }
        [AbiKind::I64, AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_i64(args, 0), arg_i64(args, 1)))
        }
        [AbiKind::F64, AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(f64, i64) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_f64(args, 0), arg_i64(args, 1)))
        }
        [AbiKind::F64, AbiKind::F64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(f64, f64) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_f64(args, 0), arg_f64(args, 1)))
        }
        [AbiKind::F64, AbiKind::Bool] => {
            let f: libloading::Symbol<unsafe extern "C" fn(f64, bool) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_f64(args, 0), arg_bool(args, 1)))
        }
        [AbiKind::Bool, AbiKind::I64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(bool, i64) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_bool(args, 0), arg_i64(args, 1)))
        }
        [AbiKind::Bool, AbiKind::F64] => {
            let f: libloading::Symbol<unsafe extern "C" fn(bool, f64) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_bool(args, 0), arg_f64(args, 1)))
        }
        [AbiKind::Bool, AbiKind::Bool] => {
            let f: libloading::Symbol<unsafe extern "C" fn(bool, bool) -> bool> =
                artifact.library.get(b"rumba_entry").map_err(load_error)?;
            Ok(f(arg_bool(args, 0), arg_bool(args, 1)))
        }
        _ => unsupported_native_signature("bool", kinds, debug),
    }
}

fn unsupported_native_signature<T>(
    return_type: &str,
    kinds: &[AbiKind],
    debug: bool,
) -> PyResult<T> {
    if debug {
        eprintln!(
            "[rumba-debug] runtime: unsupported native ABI dispatch: return_type={return_type}, ABI kinds={kinds:?}"
        );
    }
    Err(unsupported(
        "native invocation does not support this signature",
    ))
}

fn prepare_arg(
    typ: &RumbaType,
    arg: &Bound<'_, PyAny>,
    require_writable_array: bool,
) -> PyResult<NativeArg> {
    match typ {
        RumbaType::Scalar(ScalarType::Int64) => Ok(NativeArg::I64(arg.extract()?)),
        RumbaType::Scalar(ScalarType::Float64) => Ok(NativeArg::F64(arg.extract()?)),
        RumbaType::Scalar(ScalarType::Bool) => Ok(NativeArg::Bool(arg.extract()?)),
        RumbaType::Array1D(ScalarType::Int64) => {
            let (data, len) = validate_array(arg, "int64", require_writable_array)?;
            Ok(NativeArg::ArrayI64(ArrayI64View {
                data: data as *mut i64,
                len,
            }))
        }
        RumbaType::Array1D(ScalarType::Float64) => {
            let (data, len) = validate_array(arg, "float64", require_writable_array)?;
            Ok(NativeArg::ArrayF64(ArrayF64View {
                data: data as *mut f64,
                len,
            }))
        }
        RumbaType::Array1D(ScalarType::Bool) => Err(unsupported("bool arrays are not supported")),
    }
}

fn abi_kind(typ: &RumbaType) -> PyResult<AbiKind> {
    match typ {
        RumbaType::Scalar(ScalarType::Int64) => Ok(AbiKind::I64),
        RumbaType::Scalar(ScalarType::Float64) => Ok(AbiKind::F64),
        RumbaType::Scalar(ScalarType::Bool) => Ok(AbiKind::Bool),
        RumbaType::Array1D(ScalarType::Int64) => Ok(AbiKind::ArrayI64),
        RumbaType::Array1D(ScalarType::Float64) => Ok(AbiKind::ArrayF64),
        RumbaType::Array1D(ScalarType::Bool) => Err(unsupported("bool arrays are not supported")),
    }
}

fn validate_array(
    arg: &Bound<'_, PyAny>,
    expected_dtype: &str,
    require_writable: bool,
) -> PyResult<(*mut std::ffi::c_void, i64)> {
    let py = arg.py();
    let numpy = py
        .import_bound("numpy")
        .map_err(|_| unsupported("numpy is required for array arguments"))?;
    let ndarray = numpy.getattr("ndarray")?;
    if !arg.is_instance(&ndarray)? {
        return Err(unsupported(
            "array arguments must be numpy.ndarray instances",
        ));
    }
    let ndim: i64 = arg.getattr("ndim")?.extract()?;
    if ndim != 1 {
        return Err(unsupported("only 1D numpy arrays are supported"));
    }
    let dtype = arg.getattr("dtype")?.str()?.to_str()?.to_string();
    if dtype != expected_dtype {
        return Err(unsupported(format!(
            "compiled signature expects numpy dtype {expected_dtype}, got {dtype}"
        )));
    }
    let flags = arg.getattr("flags")?;
    let c_contiguous: bool = flags.getattr("c_contiguous")?.extract()?;
    let aligned: bool = flags.getattr("aligned")?.extract()?;
    if !c_contiguous || !aligned {
        return Err(unsupported(
            "only C-contiguous aligned numpy arrays are supported",
        ));
    }
    if require_writable {
        let writeable: bool = flags.getattr("writeable")?.extract()?;
        if !writeable {
            return Err(unsupported(
                "read-only numpy arrays cannot be passed to functions that assign array elements",
            ));
        }
    }
    let len: i64 = arg.call_method0("__len__")?.extract()?;
    let data = arg.getattr("ctypes")?.getattr("data")?.extract::<usize>()?;
    Ok((data as *mut std::ffi::c_void, len))
}

fn arg_i64(args: &[NativeArg], index: usize) -> i64 {
    match args[index] {
        NativeArg::I64(value) => value,
        _ => unreachable!(),
    }
}

fn arg_f64(args: &[NativeArg], index: usize) -> f64 {
    match args[index] {
        NativeArg::F64(value) => value,
        _ => unreachable!(),
    }
}

fn arg_bool(args: &[NativeArg], index: usize) -> bool {
    match args[index] {
        NativeArg::Bool(value) => value,
        _ => unreachable!(),
    }
}

fn arg_array_i64(args: &[NativeArg], index: usize) -> ArrayI64View {
    match args[index] {
        NativeArg::ArrayI64(value) => value,
        _ => unreachable!(),
    }
}

fn arg_array_f64(args: &[NativeArg], index: usize) -> ArrayF64View {
    match args[index] {
        NativeArg::ArrayF64(value) => value,
        _ => unreachable!(),
    }
}

fn load_error(err: libloading::Error) -> PyErr {
    compilation(format!("failed to load compiled symbol: {err}"))
}
