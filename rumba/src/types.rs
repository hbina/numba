use std::sync::Arc;

use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PyString, PyTuple};

use crate::errors::unsupported;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ScalarType {
    Int64,
    Float64,
    Bool,
}

impl ScalarType {
    pub(crate) fn from_name(name: &str) -> PyResult<Self> {
        match name {
            "int64" => Ok(Self::Int64),
            "float64" => Ok(Self::Float64),
            "bool" => Ok(Self::Bool),
            other => Err(unsupported(format!("unsupported signature type {other:?}"))),
        }
    }

    pub(crate) fn from_arg(arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        if arg.downcast::<PyBool>().is_ok() {
            Ok(Self::Bool)
        } else if arg.extract::<i64>().is_ok() {
            Ok(Self::Int64)
        } else if arg.extract::<f64>().is_ok() {
            Ok(Self::Float64)
        } else {
            Err(unsupported(format!(
                "unsupported argument type {}; supported scalar types are int, float, and bool",
                arg.get_type().name()?
            )))
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Int64 => "int64",
            Self::Float64 => "float64",
            Self::Bool => "bool",
        }
    }

    pub(crate) fn c_type(self) -> &'static str {
        match self {
            Self::Int64 => "int64_t",
            Self::Float64 => "double",
            Self::Bool => "bool",
        }
    }
}

/// Primitive types that can appear as fields inside a structured NumPy dtype.
/// This is a superset of `ScalarType` — it covers all standard NumPy numeric
/// primitives. Standalone function arguments and returns still use `ScalarType`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum FieldType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

impl FieldType {
    /// Parse from numpy dtype `kind` (single char) and `itemsize` (bytes).
    pub(crate) fn from_numpy_kind_size(kind: char, itemsize: usize) -> Option<Self> {
        match (kind, itemsize) {
            ('b', 1) => Some(Self::Bool),
            ('i', 1) => Some(Self::Int8),
            ('i', 2) => Some(Self::Int16),
            ('i', 4) => Some(Self::Int32),
            ('i', 8) => Some(Self::Int64),
            ('u', 1) => Some(Self::UInt8),
            ('u', 2) => Some(Self::UInt16),
            ('u', 4) => Some(Self::UInt32),
            ('u', 8) => Some(Self::UInt64),
            ('f', 4) => Some(Self::Float32),
            ('f', 8) => Some(Self::Float64),
            _ => None,
        }
    }

    pub(crate) fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 => 8,
        }
    }

    pub(crate) fn c_alignment(self) -> usize {
        self.byte_size()
    }

    pub(crate) fn c_type(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int8 => "int8_t",
            Self::Int16 => "int16_t",
            Self::Int32 => "int32_t",
            Self::Int64 => "int64_t",
            Self::UInt8 => "uint8_t",
            Self::UInt16 => "uint16_t",
            Self::UInt32 => "uint32_t",
            Self::UInt64 => "uint64_t",
            Self::Float32 => "float",
            Self::Float64 => "double",
        }
    }

    /// Short identifier used in generated C struct names, e.g. "i64", "f32".
    pub(crate) fn c_name_part(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int8 => "i8",
            Self::Int16 => "i16",
            Self::Int32 => "i32",
            Self::Int64 => "i64",
            Self::UInt8 => "u8",
            Self::UInt16 => "u16",
            Self::UInt32 => "u32",
            Self::UInt64 => "u64",
            Self::Float32 => "f32",
            Self::Float64 => "f64",
        }
    }

    /// Widen to the nearest `ScalarType` for arithmetic within Rumba.
    /// Int/UInt → Int64, Float → Float64, Bool → Bool.
    pub(crate) fn to_scalar_type(self) -> ScalarType {
        match self {
            Self::Bool => ScalarType::Bool,
            Self::Float32 | Self::Float64 => ScalarType::Float64,
            Self::Int8
            | Self::Int16
            | Self::Int32
            | Self::Int64
            | Self::UInt8
            | Self::UInt16
            | Self::UInt32
            | Self::UInt64 => ScalarType::Int64,
        }
    }
}

/// Description of a structured NumPy dtype as seen by Rumba.
/// Fields are in declaration order and carry numpy's actual byte offsets.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StructDtype {
    /// Ordered field list: (name, type).
    pub(crate) fields: Vec<(String, FieldType)>,
    /// Byte offset of each field as reported by numpy (parallel to `fields`).
    pub(crate) offsets: Vec<usize>,
    /// Total bytes per element as reported by `dtype.itemsize`.
    pub(crate) itemsize: usize,
    /// Generated C struct name, e.g. `rumba_struct_a_i64_b_f32`.
    pub(crate) c_struct_name: String,
    /// Generated C array-wrapper name, e.g. `rumba_array_struct_a_i64_b_f32`.
    pub(crate) c_array_name: String,
}

impl StructDtype {
    /// Build from a numpy dtype object (must be a structured dtype).
    pub(crate) fn from_numpy_dtype(dtype: &Bound<'_, PyAny>) -> PyResult<Arc<Self>> {
        let names_obj = dtype.getattr("names")?;
        if names_obj.is_none() {
            return Err(unsupported("not a structured numpy dtype"));
        }
        let names: Vec<String> = names_obj.extract()?;
        let fields_dict = dtype.getattr("fields")?;
        let itemsize: usize = dtype.getattr("itemsize")?.extract()?;

        let mut fields = Vec::with_capacity(names.len());
        let mut offsets = Vec::with_capacity(names.len());

        for name in &names {
            let field_tuple = fields_dict.get_item(name.as_str())?;
            let field_dtype = field_tuple.get_item(0)?;
            let offset: usize = field_tuple.get_item(1)?.extract()?;

            let kind_str: String = field_dtype.getattr("kind")?.extract()?;
            let kind = kind_str.chars().next().unwrap_or('\0');
            let field_itemsize: usize = field_dtype.getattr("itemsize")?.extract()?;

            // Reject unsupported field dtypes (nested structs, subarrays, complex, objects…)
            let subdtype = field_dtype.getattr("subdtype")?;
            if !subdtype.is_none() {
                return Err(unsupported(format!(
                    "field {name:?}: subarray fields are not supported"
                )));
            }
            let field_names = field_dtype.getattr("names")?;
            if !field_names.is_none() {
                return Err(unsupported(format!(
                    "field {name:?}: nested struct fields are not supported"
                )));
            }

            let field_type =
                FieldType::from_numpy_kind_size(kind, field_itemsize).ok_or_else(|| {
                    unsupported(format!(
                        "field {name:?}: unsupported field dtype (kind={kind:?}, itemsize={field_itemsize}); \
                         supported field types are bool, int8/16/32/64, uint8/16/32/64, float32, float64"
                    ))
                })?;

            fields.push((name.clone(), field_type));
            offsets.push(offset);
        }

        validate_c_struct_layout(&fields, &offsets, itemsize)?;

        let c_struct_name = Self::make_c_struct_name(&fields);
        let c_array_name = format!("rumba_array_{c_struct_name}");

        Ok(Arc::new(Self {
            fields,
            offsets,
            itemsize,
            c_struct_name,
            c_array_name,
        }))
    }

    fn make_c_struct_name(fields: &[(String, FieldType)]) -> String {
        let parts = fields
            .iter()
            .map(|(name, ft)| format!("{}_{}", name, ft.c_name_part()))
            .collect::<Vec<_>>()
            .join("_");
        format!("rumba_struct_{parts}")
    }

    /// Look up a field by name; returns (FieldType, byte_offset).
    pub(crate) fn field(&self, name: &str) -> Option<(FieldType, usize)> {
        self.fields
            .iter()
            .zip(self.offsets.iter())
            .find_map(|((fname, ftype), &offset)| {
                if fname == name {
                    Some((*ftype, offset))
                } else {
                    None
                }
            })
    }
}

fn validate_c_struct_layout(
    fields: &[(String, FieldType)],
    offsets: &[usize],
    itemsize: usize,
) -> PyResult<()> {
    let mut cursor = 0_usize;
    let mut max_align = 1_usize;
    for ((name, field_type), offset) in fields.iter().zip(offsets.iter()) {
        if !is_safe_c_identifier(name) {
            return Err(unsupported(format!(
                "structured dtype field {name:?} cannot be emitted as a C identifier"
            )));
        }
        let align = field_type.c_alignment();
        max_align = max_align.max(align);
        if offset % align != 0 {
            return Err(unsupported(format!(
                "field {name:?} offset {offset} violates C alignment {align}"
            )));
        }
        if *offset < cursor {
            return Err(unsupported(format!(
                "field {name:?} overlaps a previous structured dtype field"
            )));
        }
        cursor = offset + field_type.byte_size();
    }
    if cursor > itemsize {
        return Err(unsupported(
            "structured dtype fields extend beyond dtype itemsize",
        ));
    }
    if itemsize % max_align != 0 {
        return Err(unsupported(format!(
            "structured dtype itemsize {itemsize} is not representable with C struct alignment {max_align}"
        )));
    }
    Ok(())
}

fn is_safe_c_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    if !chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric()) {
        return false;
    }
    !matches!(
        name,
        "auto"
            | "break"
            | "case"
            | "char"
            | "const"
            | "continue"
            | "default"
            | "do"
            | "double"
            | "else"
            | "enum"
            | "extern"
            | "float"
            | "for"
            | "goto"
            | "if"
            | "inline"
            | "int"
            | "long"
            | "register"
            | "restrict"
            | "return"
            | "short"
            | "signed"
            | "sizeof"
            | "static"
            | "struct"
            | "switch"
            | "typedef"
            | "union"
            | "unsigned"
            | "void"
            | "volatile"
            | "while"
            | "_Alignas"
            | "_Alignof"
            | "_Atomic"
            | "_Bool"
            | "_Complex"
            | "_Generic"
            | "_Imaginary"
            | "_Noreturn"
            | "_Static_assert"
            | "_Thread_local"
    )
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RumbaType {
    Scalar(ScalarType),
    Array1D(ScalarType),
    Array1DStruct(Arc<StructDtype>),
}

impl RumbaType {
    pub(crate) fn from_name(name: &str) -> PyResult<Self> {
        match name {
            "int64" | "float64" | "bool" => Ok(Self::Scalar(ScalarType::from_name(name)?)),
            "array(int64, 1d, C)" => Ok(Self::Array1D(ScalarType::Int64)),
            "array(float64, 1d, C)" => Ok(Self::Array1D(ScalarType::Float64)),
            other => Err(unsupported(format!("unsupported signature type {other:?}"))),
        }
    }

    pub(crate) fn from_arg(arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = arg.py();
        let numpy = match py.import_bound("numpy") {
            Ok(numpy) => numpy,
            Err(_) => return ScalarType::from_arg(arg).map(Self::Scalar),
        };
        let ndarray = numpy.getattr("ndarray")?;
        if arg.is_instance(&ndarray)? {
            let dtype = arg.getattr("dtype")?;
            let names = dtype.getattr("names")?;
            if !names.is_none() {
                // Structured array: infer StructDtype from the dtype.
                let struct_dtype = StructDtype::from_numpy_dtype(&dtype)?;
                return Ok(Self::Array1DStruct(struct_dtype));
            }
            let dtype_str = dtype.str()?.to_str()?.to_string();
            return match dtype_str.as_str() {
                "int64" => Ok(Self::Array1D(ScalarType::Int64)),
                "float64" => Ok(Self::Array1D(ScalarType::Float64)),
                other => Err(unsupported(format!(
                    "unsupported numpy array dtype {other}; supported array dtypes are int64, float64, and structured dtypes"
                ))),
            };
        }
        ScalarType::from_arg(arg).map(Self::Scalar)
    }

    pub(crate) fn name(&self) -> String {
        match self {
            Self::Scalar(typ) => typ.name().to_string(),
            Self::Array1D(typ) => format!("array({}, 1d, C)", typ.name()),
            Self::Array1DStruct(dtype) => {
                let fields = dtype
                    .fields
                    .iter()
                    .map(|(name, ft)| format!("{name}:{}", ft.c_name_part()))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("array(struct{{{fields}}}, 1d, C)")
            }
        }
    }

    pub(crate) fn c_type(&self) -> String {
        match self {
            Self::Scalar(typ) => typ.c_type().to_string(),
            Self::Array1D(ScalarType::Int64) => "rumba_array_i64".to_string(),
            Self::Array1D(ScalarType::Float64) => "rumba_array_f64".to_string(),
            Self::Array1D(ScalarType::Bool) => "rumba_array_bool".to_string(),
            Self::Array1DStruct(dtype) => dtype.c_array_name.clone(),
        }
    }

    pub(crate) fn as_scalar(&self) -> Option<ScalarType> {
        match self {
            Self::Scalar(typ) => Some(*typ),
            Self::Array1D(_) | Self::Array1DStruct(_) => None,
        }
    }
}

#[pyclass(name = "ScalarType", frozen)]
#[derive(Clone)]
pub(crate) struct PyScalarType {
    pub(crate) typ: ScalarType,
}

#[pymethods]
impl PyScalarType {
    #[getter]
    fn name(&self) -> &'static str {
        self.typ.name()
    }

    fn __repr__(&self) -> String {
        format!("rumba.{}", self.typ.name())
    }

    fn __richcmp__(&self, other: PyRef<'_, PyScalarType>, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.typ == other.typ,
            CompareOp::Ne => self.typ != other.typ,
            _ => false,
        }
    }

    fn __hash__(&self) -> isize {
        match self.typ {
            ScalarType::Int64 => 1,
            ScalarType::Float64 => 2,
            ScalarType::Bool => 3,
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScalarType>()?;
    Ok(())
}

pub(crate) fn parse_signature_value(value: &Bound<'_, PyAny>) -> PyResult<Vec<RumbaType>> {
    if let Ok(value) = value.downcast::<PyString>() {
        let value = value.to_str()?;
        return value
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(RumbaType::from_name)
            .collect();
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        return tuple
            .iter()
            .map(|part| parse_signature_part(&part))
            .collect();
    }
    if let Ok(list) = value.downcast::<PyList>() {
        return list
            .iter()
            .map(|part| parse_signature_part(&part))
            .collect();
    }
    Err(unsupported("signature must be a string, tuple, or list"))
}

pub(crate) fn parse_signature_part(value: &Bound<'_, PyAny>) -> PyResult<RumbaType> {
    if let Ok(value) = value.extract::<PyRef<'_, PyScalarType>>() {
        return Ok(RumbaType::Scalar(value.typ));
    }
    if let Ok(value) = value.downcast::<PyString>() {
        return RumbaType::from_name(value.to_str()?);
    }
    let repr = value.repr()?.extract::<String>()?;
    match repr.as_str() {
        "<class 'int'>" => Ok(RumbaType::Scalar(ScalarType::Int64)),
        "<class 'float'>" => Ok(RumbaType::Scalar(ScalarType::Float64)),
        "<class 'bool'>" => Ok(RumbaType::Scalar(ScalarType::Bool)),
        _ => Err(unsupported(format!("unsupported signature type {repr}"))),
    }
}

pub(crate) fn parse_signature_tuple(tuple: &Bound<'_, PyTuple>) -> PyResult<Vec<RumbaType>> {
    tuple
        .iter()
        .map(|item| parse_signature_part(&item))
        .collect()
}

pub(crate) fn signature_tuple(py: Python<'_>, signature: &[RumbaType]) -> PyResult<PyObject> {
    let items = signature
        .iter()
        .map(|typ| match typ {
            RumbaType::Scalar(typ) => {
                Py::new(py, PyScalarType { typ: *typ }).map(|obj| obj.into_py(py))
            }
            RumbaType::Array1D(_) | RumbaType::Array1DStruct(_) => Ok(typ.name().into_py(py)),
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new_bound(py, items).into())
}

pub(crate) fn format_signature(signature: &[RumbaType]) -> String {
    signature
        .iter()
        .map(|typ| typ.name())
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn promote_numeric(
    left: ScalarType,
    right: ScalarType,
    op: impl AsRef<str>,
) -> ScalarType {
    match op.as_ref() {
        "Eq" | "NotEq" | "Lt" | "LtE" | "Gt" | "GtE" => ScalarType::Bool,
        "Div" => ScalarType::Float64,
        _ if left == ScalarType::Float64 || right == ScalarType::Float64 => ScalarType::Float64,
        _ if left == ScalarType::Bool && right == ScalarType::Bool => ScalarType::Bool,
        _ => ScalarType::Int64,
    }
}
