use pyo3::prelude::*;

use crate::errors::unsupported;
use crate::types::{promote_numeric, RumbaType, ScalarType};
use crate::typing::TypedExpr;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum IntrinsicId {
    BuiltinLen,
    BuiltinMin,
    BuiltinMax,
    BuiltinAbs,
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
}

impl IntrinsicId {
    pub(crate) fn from_builtin(name: &str) -> Option<Self> {
        match name {
            "len" => Some(Self::BuiltinLen),
            "min" => Some(Self::BuiltinMin),
            "max" => Some(Self::BuiltinMax),
            "abs" => Some(Self::BuiltinAbs),
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
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::BuiltinLen => "len",
            Self::BuiltinMin => "min",
            Self::BuiltinMax => "max",
            Self::BuiltinAbs => "abs",
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
        }
    }

    pub(crate) fn type_call(self, args: &[TypedExpr]) -> PyResult<RumbaType> {
        match self {
            Self::BuiltinLen => {
                require_arg_count(self, args, 1)?;
                match args[0].typ {
                    RumbaType::Array1D(_) => Ok(RumbaType::Scalar(ScalarType::Int64)),
                    RumbaType::Scalar(_) => Err(unsupported("len expects a 1D numpy array")),
                }
            }
            Self::BuiltinMin | Self::BuiltinMax => {
                if args.len() < 2 {
                    return Err(unsupported(format!(
                        "{} expects at least two scalar arguments",
                        self.name()
                    )));
                }
                let mut out = scalar_arg(self, &args[0])?;
                for arg in &args[1..] {
                    out = promote_numeric(out, scalar_arg(self, arg)?, self.name());
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
            Self::MathSqrt
            | Self::MathSin
            | Self::MathCos
            | Self::MathTan
            | Self::MathExp
            | Self::MathLog
            | Self::MathFloor
            | Self::MathCeil => {
                require_arg_count(self, args, 1)?;
                scalar_arg(self, &args[0])?;
                Ok(RumbaType::Scalar(ScalarType::Float64))
            }
            Self::NumpyMax | Self::NumpyMin | Self::NumpySum => {
                require_arg_count(self, args, 1)?;
                match args[0].typ {
                    RumbaType::Array1D(element_type) => Ok(RumbaType::Scalar(element_type)),
                    RumbaType::Scalar(_) => Err(unsupported(format!(
                        "{} expects a 1D numpy array",
                        self.name()
                    ))),
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
