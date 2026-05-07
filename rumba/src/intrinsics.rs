use pyo3::prelude::*;

use crate::errors::unsupported;
use crate::types::{RumbaType, ScalarType};
use crate::typing::TypedExpr;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum IntrinsicId {
    BuiltinLen,
    BuiltinMin,
    BuiltinMax,
    BuiltinAbs,
    BuiltinInt,
    BuiltinFloat,
    BuiltinBool,
    MathSqrt,
    MathSin,
    MathCos,
    MathTan,
    MathExp,
    MathLog,
    MathFloor,
    MathCeil,
    NumpyMax,
    NumpyMin,
    NumpySum,
    NumpyFromBuffer,
}

impl IntrinsicId {
    pub(crate) fn from_builtin(name: &str) -> Option<Self> {
        match name {
            "len" => Some(Self::BuiltinLen),
            "min" => Some(Self::BuiltinMin),
            "max" => Some(Self::BuiltinMax),
            "abs" => Some(Self::BuiltinAbs),
            "int" => Some(Self::BuiltinInt),
            "float" => Some(Self::BuiltinFloat),
            "bool" => Some(Self::BuiltinBool),
            _ => None,
        }
    }

    pub(crate) fn from_module_attr(module: &str, attr: &str) -> Option<Self> {
        match (module, attr) {
            ("math", "sqrt") => Some(Self::MathSqrt),
            ("math", "sin") => Some(Self::MathSin),
            ("math", "cos") => Some(Self::MathCos),
            ("math", "tan") => Some(Self::MathTan),
            ("math", "exp") => Some(Self::MathExp),
            ("math", "log") => Some(Self::MathLog),
            ("math", "floor") => Some(Self::MathFloor),
            ("math", "ceil") => Some(Self::MathCeil),
            ("numpy", "max") => Some(Self::NumpyMax),
            ("numpy", "min") => Some(Self::NumpyMin),
            ("numpy", "sum") => Some(Self::NumpySum),
            ("numpy", "frombuffer") => Some(Self::NumpyFromBuffer),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::BuiltinLen => "len",
            Self::BuiltinMin => "min",
            Self::BuiltinMax => "max",
            Self::BuiltinAbs => "abs",
            Self::BuiltinInt => "int",
            Self::BuiltinFloat => "float",
            Self::BuiltinBool => "bool",
            Self::MathSqrt => "math.sqrt",
            Self::MathSin => "math.sin",
            Self::MathCos => "math.cos",
            Self::MathTan => "math.tan",
            Self::MathExp => "math.exp",
            Self::MathLog => "math.log",
            Self::MathFloor => "math.floor",
            Self::MathCeil => "math.ceil",
            Self::NumpyMax => "numpy.max",
            Self::NumpyMin => "numpy.min",
            Self::NumpySum => "numpy.sum",
            Self::NumpyFromBuffer => "numpy.frombuffer",
        }
    }

    pub(crate) fn type_call(self, args: &[TypedExpr]) -> PyResult<RumbaType> {
        match self {
            Self::BuiltinLen => {
                require_arg_count(self, args, 1)?;
                match args[0].typ {
                    RumbaType::Array1D(_) | RumbaType::Array1DStruct(_) => {
                        Ok(RumbaType::Scalar(ScalarType::Int64))
                    }
                    RumbaType::Scalar(_) | RumbaType::ByteBuffer | RumbaType::Dtype(_) => {
                        Err(unsupported("len expects a 1D numpy array"))
                    }
                }
            }
            Self::BuiltinMin | Self::BuiltinMax => {
                if args.len() < 2 {
                    return Err(unsupported(format!(
                        "{} expects at least two scalar arguments",
                        self.name()
                    )));
                }
                let out = scalar_arg(self, &args[0])?;
                if out == ScalarType::Bool {
                    return Err(unsupported(format!(
                        "{} does not support bool arguments",
                        self.name()
                    )));
                }
                for arg in &args[1..] {
                    let typ = scalar_arg(self, arg)?;
                    if typ != out {
                        return Err(unsupported(format!(
                            "{} requires exact matching scalar argument types, got {} and {}; use an explicit cast",
                            self.name(),
                            out.name(),
                            typ.name()
                        )));
                    }
                }
                Ok(RumbaType::Scalar(out))
            }
            Self::BuiltinAbs => {
                require_arg_count(self, args, 1)?;
                let typ = scalar_arg(self, &args[0])?;
                if typ == ScalarType::Bool {
                    return Err(unsupported("abs does not support bool arguments"));
                }
                Ok(RumbaType::Scalar(typ))
            }
            Self::BuiltinInt | Self::BuiltinFloat | Self::BuiltinBool => {
                require_arg_count(self, args, 1)?;
                scalar_arg(self, &args[0])?;
                let typ = match self {
                    Self::BuiltinInt => ScalarType::Int64,
                    Self::BuiltinFloat => ScalarType::Float64,
                    Self::BuiltinBool => ScalarType::Bool,
                    _ => unreachable!(),
                };
                Ok(RumbaType::Scalar(typ))
            }
            Self::MathSqrt
            | Self::MathSin
            | Self::MathCos
            | Self::MathTan
            | Self::MathExp
            | Self::MathLog
            | Self::MathFloor
            | Self::MathCeil => {
                require_arg_count(self, args, 1)?;
                let typ = scalar_arg(self, &args[0])?;
                if typ != ScalarType::Float64 {
                    return Err(unsupported(format!(
                        "{} requires float64 argument, got {}; use an explicit float(...) cast",
                        self.name(),
                        typ.name()
                    )));
                }
                Ok(RumbaType::Scalar(ScalarType::Float64))
            }
            Self::NumpyMax | Self::NumpyMin | Self::NumpySum => {
                require_arg_count(self, args, 1)?;
                match args[0].typ {
                    RumbaType::Array1D(element_type) => Ok(RumbaType::Scalar(element_type)),
                    RumbaType::Array1DStruct(_) => Err(unsupported(format!(
                        "{} does not support structured arrays; read a field first",
                        self.name()
                    ))),
                    RumbaType::Scalar(_) | RumbaType::ByteBuffer | RumbaType::Dtype(_) => Err(unsupported(format!(
                        "{} expects a 1D numpy array",
                        self.name()
                    ))),
                }
            }
            Self::NumpyFromBuffer => {
                require_arg_count(self, args, 4)?;
                if args[0].typ != RumbaType::ByteBuffer {
                    return Err(unsupported(
                        "np.frombuffer source must be a 1D uint8 numpy array or memmap",
                    ));
                }
                let dtype = match &args[1].typ {
                    RumbaType::Dtype(dtype) => dtype,
                    _ => {
                        return Err(unsupported(
                            "np.frombuffer dtype must be a module-level numpy dtype object",
                        ));
                    }
                };
                if args[2].typ != RumbaType::Scalar(ScalarType::Int64) {
                    return Err(unsupported("np.frombuffer count must be an int64 scalar"));
                }
                if args[3].typ != RumbaType::Scalar(ScalarType::Int64) {
                    return Err(unsupported("np.frombuffer offset must be an int64 scalar"));
                }
                match dtype {
                    crate::types::DtypeSpec::Scalar(typ) => Ok(RumbaType::Array1D(*typ)),
                    crate::types::DtypeSpec::Struct(dtype) => Ok(RumbaType::Array1DStruct(dtype.clone())),
                }
            }
        }
    }
}

fn require_arg_count(intrinsic: IntrinsicId, args: &[TypedExpr], expected: usize) -> PyResult<()> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(unsupported(format!(
            "{} expects {expected} argument(s)",
            intrinsic.name()
        )))
    }
}

fn scalar_arg(intrinsic: IntrinsicId, arg: &TypedExpr) -> PyResult<ScalarType> {
    arg.typ
        .as_scalar()
        .ok_or_else(|| unsupported(format!("{} expects scalar argument(s)", intrinsic.name())))
}
