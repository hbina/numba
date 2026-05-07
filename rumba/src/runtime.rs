use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::artifact::CompiledArtifact;
use crate::errors::{compilation, unsupported};
use crate::types::{FieldType, RumbaType, ScalarType, StructDtype};
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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ByteBufferView {
    data: *mut u8,
    len: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ArrayStructView {
    data: *mut c_void,
    len: i64,
}

#[derive(Clone, Copy, Debug)]
enum NativeArg {
    I64(i64),
    F64(f64),
    Bool(bool),
    ArrayI64(ArrayI64View),
    ArrayF64(ArrayF64View),
    ArrayStruct(ArrayStructView),
    ByteBuffer(ByteBufferView),
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
        let status_fn: libloading::Symbol<unsafe extern "C" fn() -> i32> = artifact
            .library
            .get(b"rumba_error_status")
            .map_err(load_error)?;
        match artifact.return_type {
            ScalarType::Int64 => {
                let mut ret = 0_i64;
                f(
                    call.arg_ptrs.as_mut_ptr(),
                    (&mut ret as *mut i64).cast::<c_void>(),
                );
                check_status(status_fn())?;
                Ok(ret.into_py(py))
            }
            ScalarType::Float64 => {
                let mut ret = 0.0_f64;
                f(
                    call.arg_ptrs.as_mut_ptr(),
                    (&mut ret as *mut f64).cast::<c_void>(),
                );
                check_status(status_fn())?;
                Ok(ret.into_py(py))
            }
            ScalarType::Bool => {
                let mut ret = false;
                f(
                    call.arg_ptrs.as_mut_ptr(),
                    (&mut ret as *mut bool).cast::<c_void>(),
                );
                check_status(status_fn())?;
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
                NativeArg::ArrayStruct(value) => (value as *mut ArrayStructView).cast::<c_void>(),
                NativeArg::ByteBuffer(value) => (value as *mut ByteBufferView).cast::<c_void>(),
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
        RumbaType::Array1DStruct(dtype) => {
            let (data, len) = validate_struct_array(arg, dtype, require_writable_array)?;
            Ok(NativeArg::ArrayStruct(ArrayStructView { data, len }))
        }
        RumbaType::ByteBuffer => {
            let (data, len) = validate_array(arg, "uint8", false)?;
            Ok(NativeArg::ByteBuffer(ByteBufferView {
                data: data as *mut u8,
                len,
            }))
        }
        RumbaType::Dtype(_) => Err(unsupported(
            "numpy dtype objects are only supported as np.frombuffer dtype arguments",
        )),
    }
}

fn check_status(status: i32) -> PyResult<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(unsupported(
            "np.frombuffer offset/count is negative or exceeds the source buffer length",
        ))
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

fn validate_struct_array(
    arg: &Bound<'_, PyAny>,
    expected: &StructDtype,
    require_writable: bool,
) -> PyResult<(*mut c_void, i64)> {
    let py = arg.py();
    let numpy = py
        .import_bound("numpy")
        .map_err(|_| unsupported("numpy is required for array arguments"))?;
    let ndarray = numpy.getattr("ndarray")?;
    if !arg.is_instance(&ndarray)? {
        return Err(unsupported(
            "structured array arguments must be numpy.ndarray instances",
        ));
    }
    let ndim: i64 = arg.getattr("ndim")?.extract()?;
    if ndim != 1 {
        return Err(unsupported("only 1D structured numpy arrays are supported"));
    }
    let flags = arg.getattr("flags")?;
    let c_contiguous: bool = flags.getattr("c_contiguous")?.extract()?;
    let aligned: bool = flags.getattr("aligned")?.extract()?;
    if !c_contiguous || !aligned {
        return Err(unsupported(
            "only C-contiguous aligned structured numpy arrays are supported",
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

    let dtype = arg.getattr("dtype")?;
    let names_obj = dtype.getattr("names")?;
    if names_obj.is_none() {
        return Err(unsupported(
            "compiled signature expects a structured numpy dtype",
        ));
    }
    let names: Vec<String> = names_obj.extract()?;
    let expected_names = expected
        .fields
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    if names != expected_names {
        return Err(unsupported(format!(
            "structured dtype field names/order mismatch; expected {:?}, got {:?}",
            expected_names, names
        )));
    }
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    if itemsize != expected.itemsize {
        return Err(unsupported(format!(
            "structured dtype itemsize mismatch; expected {}, got {}",
            expected.itemsize, itemsize
        )));
    }
    let fields_dict = dtype.getattr("fields")?;
    for ((name, expected_type), expected_offset) in
        expected.fields.iter().zip(expected.offsets.iter())
    {
        let field_tuple = fields_dict.get_item(name.as_str())?;
        let field_dtype = field_tuple.get_item(0)?;
        let offset: usize = field_tuple.get_item(1)?.extract()?;
        if offset != *expected_offset {
            return Err(unsupported(format!(
                "structured dtype field {name:?} offset mismatch; expected {}, got {}",
                expected_offset, offset
            )));
        }
        let actual_type = field_type_from_numpy_dtype(&field_dtype, name)?;
        if actual_type != *expected_type {
            return Err(unsupported(format!(
                "structured dtype field {name:?} dtype mismatch; expected {}, got {}",
                field_type_name(*expected_type),
                field_type_name(actual_type)
            )));
        }
    }

    let len: i64 = arg.call_method0("__len__")?.extract()?;
    let data = arg.getattr("ctypes")?.getattr("data")?.extract::<usize>()?;
    Ok((data as *mut c_void, len))
}

fn field_type_from_numpy_dtype(dtype: &Bound<'_, PyAny>, name: &str) -> PyResult<FieldType> {
    let subdtype = dtype.getattr("subdtype")?;
    if !subdtype.is_none() {
        return Err(unsupported(format!(
            "field {name:?}: subarray fields are not supported"
        )));
    }
    let field_names = dtype.getattr("names")?;
    if !field_names.is_none() {
        return Err(unsupported(format!(
            "field {name:?}: nested struct fields are not supported"
        )));
    }
    let kind_str: String = dtype.getattr("kind")?.extract()?;
    let kind = kind_str.chars().next().unwrap_or('\0');
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    FieldType::from_numpy_kind_size(kind, itemsize).ok_or_else(|| {
        unsupported(format!(
            "field {name:?}: unsupported field dtype (kind={kind:?}, itemsize={itemsize})"
        ))
    })
}

fn field_type_name(typ: FieldType) -> &'static str {
    match typ {
        FieldType::Bool => "bool",
        FieldType::Int8 => "int8",
        FieldType::Int16 => "int16",
        FieldType::Int32 => "int32",
        FieldType::Int64 => "int64",
        FieldType::UInt8 => "uint8",
        FieldType::UInt16 => "uint16",
        FieldType::UInt32 => "uint32",
        FieldType::UInt64 => "uint64",
        FieldType::Float32 => "float32",
        FieldType::Float64 => "float64",
    }
}

fn load_error(err: libloading::Error) -> PyErr {
    compilation(format!("failed to load compiled symbol: {err}"))
}
