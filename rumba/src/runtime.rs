use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::artifact::CompiledArtifact;
use crate::errors::{compilation, unsupported};
use crate::types::{RumbaType, ScalarType};
use std::ffi::c_void;

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
    if debug {
        eprintln!(
            "[rumba-debug] runtime: artifact signature: [{}]",
            crate::types::format_signature(&artifact.signature)
        );
        eprintln!(
            "[rumba-debug] runtime: return type: {}",
            artifact.return_type.name()
        );
        eprintln!("[rumba-debug] runtime: prepared args: {prepared:?}");
        eprintln!(
            "[rumba-debug] runtime: library path: {}",
            artifact.library_path.display()
        );
    }

    unsafe {
        let mut call = PreparedCall::new(prepared);
        if debug {
            eprintln!(
                "[rumba-debug] runtime: native wrapper argument count: {}",
                call.args.len()
            );
        }
        let f: libloading::Symbol<unsafe extern "C" fn(*mut *mut c_void, *mut c_void)> =
            artifact.library.get(b"rumba_call").map_err(load_error)?;
        match artifact.return_type {
            ScalarType::Int64 => {
                let mut ret = 0_i64;
                f(
                    call.arg_ptrs.as_mut_ptr(),
                    (&mut ret as *mut i64).cast::<c_void>(),
                );
                Ok(ret.into_py(py))
            }
            ScalarType::Float64 => {
                let mut ret = 0.0_f64;
                f(
                    call.arg_ptrs.as_mut_ptr(),
                    (&mut ret as *mut f64).cast::<c_void>(),
                );
                Ok(ret.into_py(py))
            }
            ScalarType::Bool => {
                let mut ret = false;
                f(
                    call.arg_ptrs.as_mut_ptr(),
                    (&mut ret as *mut bool).cast::<c_void>(),
                );
                Ok(ret.into_py(py))
            }
        }
    }
}

struct PreparedCall {
    args: Vec<NativeArg>,
    arg_ptrs: Vec<*mut c_void>,
}

impl PreparedCall {
    fn new(mut args: Vec<NativeArg>) -> Self {
        let arg_ptrs = args
            .iter_mut()
            .map(|arg| match arg {
                NativeArg::I64(value) => (value as *mut i64).cast::<c_void>(),
                NativeArg::F64(value) => (value as *mut f64).cast::<c_void>(),
                NativeArg::Bool(value) => (value as *mut bool).cast::<c_void>(),
                NativeArg::ArrayI64(value) => (value as *mut ArrayI64View).cast::<c_void>(),
                NativeArg::ArrayF64(value) => (value as *mut ArrayF64View).cast::<c_void>(),
            })
            .collect();
        Self { args, arg_ptrs }
    }
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

fn load_error(err: libloading::Error) -> PyErr {
    compilation(format!("failed to load compiled symbol: {err}"))
}
