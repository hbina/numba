use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;

use crate::intrinsics::IntrinsicId;
use crate::types::RumbaType;
use crate::types::ScalarType;
use crate::typing::{helper_key, TypedExpr, TypedExprKind, TypedFunction, TypedStmt};

struct CExpr {
    code: String,
}

pub(crate) struct Emitter {
    function: TypedFunction,
    declared: HashSet<String>,
    lines: Vec<String>,
    helper_sources: Vec<String>,
    helper_cache: HashMap<String, String>,
    intrinsic_cache: HashMap<String, String>,
}

impl Emitter {
    pub(crate) fn new(function: TypedFunction) -> Self {
        let declared = function.args.iter().cloned().collect::<HashSet<_>>();
        Self {
            function,
            declared,
            lines: Vec::new(),
            helper_sources: Vec::new(),
            helper_cache: HashMap::new(),
            intrinsic_cache: HashMap::new(),
        }
    }

    pub(crate) fn emit(&mut self) -> PyResult<String> {
        let entry_source = self.emit_function("rumba_entry", true)?;
        let mut source = vec![
            "#include <stdbool.h>".to_string(),
            "#include <stdint.h>".to_string(),
            "#include <math.h>".to_string(),
            String::new(),
            "typedef struct { int64_t *data; int64_t len; } rumba_array_i64;".to_string(),
            "typedef struct { double *data; int64_t len; } rumba_array_f64;".to_string(),
            String::new(),
        ];
        source.append(&mut self.helper_sources);
        source.push(entry_source);
        Ok(format!("{}\n", source.join("\n")))
    }

    fn emit_function(&mut self, c_name: &str, exported: bool) -> PyResult<String> {
        let params = self
            .function
            .args
            .iter()
            .zip(self.function.signature.iter())
            .map(|(name, typ)| format!("{} {}", typ.c_type(), name))
            .collect::<Vec<_>>()
            .join(", ");

        self.lines.clear();
        let visibility = if exported {
            "__attribute__((visibility(\"default\"))) "
        } else {
            "static "
        };
        self.lines.push(format!(
            "{visibility}{} {c_name}({params}) {{",
            self.function.return_type.c_type()
        ));

        let body = self.function.body.clone();
        for stmt in &body {
            self.stmt(stmt, 1)?;
        }

        self.lines.push("}".to_string());
        Ok(format!("{}\n", self.lines.join("\n")))
    }

    fn stmt(&mut self, node: &TypedStmt, level: usize) -> PyResult<()> {
        match node {
            TypedStmt::Return(value) => {
                let expr = self.expr(value)?;
                self.lines
                    .push(format!("{}return {};", indent(level), expr.code));
            }
            TypedStmt::Assign { name, value } => {
                let expr = self.expr(value)?;
                let prefix = if self.declared.contains(name) {
                    String::new()
                } else {
                    self.declared.insert(name.clone());
                    format!("{} ", value.typ.c_type())
                };
                self.lines
                    .push(format!("{}{prefix}{name} = {};", indent(level), expr.code));
            }
            TypedStmt::AugAssign {
                name, op, value, ..
            } => {
                let expr = self.expr(value)?;
                self.lines.push(format!(
                    "{}{name} {}= {};",
                    indent(level),
                    op.symbol(),
                    expr.code
                ));
            }
            TypedStmt::StoreIndex {
                target,
                index,
                value,
                element_type,
            } => {
                let target = self.expr(target)?;
                let index = self.expr(index)?;
                let value_type = value
                    .typ
                    .as_scalar()
                    .expect("typed scalar array assignment");
                let value = self.expr(value)?;
                let rhs = if value_type == *element_type {
                    value.code
                } else {
                    format!("({})({})", element_type.c_type(), value.code)
                };
                self.lines.push(format!(
                    "{}{}.data[{}] = {};",
                    indent(level),
                    target.code,
                    index.code,
                    rhs
                ));
            }
            TypedStmt::If { test, body, orelse } => {
                let test = self.expr(test)?;
                let declared_before = self.declared.clone();
                self.lines
                    .push(format!("{}if ({}) {{", indent(level), test.code));
                for child in body {
                    self.stmt(child, level + 1)?;
                }
                let body_declared = self.declared.clone();
                if !orelse.is_empty() {
                    self.declared = declared_before;
                    self.lines.push(format!("{}}} else {{", indent(level)));
                    for child in orelse {
                        self.stmt(child, level + 1)?;
                    }
                }
                let else_declared = self.declared.clone();
                self.declared = body_declared;
                self.declared.extend(else_declared);
                self.lines.push(format!("{}}}", indent(level)));
            }
            TypedStmt::ForRange {
                target,
                start,
                stop,
                step,
                body,
            } => self.for_range(target, start, stop, step, body, level)?,
        }
        Ok(())
    }

    fn for_range(
        &mut self,
        name: &str,
        start: &TypedExpr,
        stop: &TypedExpr,
        step: &TypedExpr,
        body: &[TypedStmt],
        level: usize,
    ) -> PyResult<()> {
        let start = self.expr(start)?.code;
        let stop = self.expr(stop)?.code;
        let step = self.expr(step)?.code;
        let decl = if self.declared.contains(name) {
            String::new()
        } else {
            self.declared.insert(name.to_string());
            "int64_t ".to_string()
        };
        let cmp = if step.starts_with('-') { ">" } else { "<" };
        self.lines.push(format!(
            "{}for ({decl}{name} = {start}; {name} {cmp} {stop}; {name} += {step}) {{",
            indent(level)
        ));
        for child in body {
            self.stmt(child, level + 1)?;
        }
        self.lines.push(format!("{}}}", indent(level)));
        Ok(())
    }

    fn expr(&mut self, node: &TypedExpr) -> PyResult<CExpr> {
        match &node.kind {
            TypedExprKind::Constant(value) => Ok(CExpr {
                code: match value {
                    crate::ir::ConstantValue::Bool(value) => {
                        if *value { "true" } else { "false" }.to_string()
                    }
                    crate::ir::ConstantValue::Int(value) => value.to_string(),
                    crate::ir::ConstantValue::Float(value) => value.to_string(),
                },
            }),
            TypedExprKind::Name(name) => Ok(CExpr { code: name.clone() }),
            TypedExprKind::Call { function, args } => {
                let key = helper_key(function);
                let c_name = if let Some(c_name) = self.helper_cache.get(&key) {
                    c_name.clone()
                } else {
                    let c_name = format!("rumba_helper_{}", self.helper_cache.len());
                    let mut helper = Emitter::new((**function).clone());
                    let source = helper.emit_function(&c_name, false)?;
                    self.helper_sources.append(&mut helper.helper_sources);
                    self.helper_sources.push(source);
                    self.helper_cache.insert(key, c_name.clone());
                    c_name
                };
                let code = args
                    .iter()
                    .map(|arg| self.expr(arg).map(|expr| expr.code))
                    .collect::<PyResult<Vec<_>>>()?
                    .join(", ");
                Ok(CExpr {
                    code: format!("{c_name}({code})"),
                })
            }
            TypedExprKind::IntrinsicCall { intrinsic, args } => {
                self.intrinsic_expr(*intrinsic, args, node.typ)
            }
            TypedExprKind::Index { target, index } => {
                let target = self.expr(target)?;
                let index = self.expr(index)?;
                Ok(CExpr {
                    code: format!("{}.data[{}]", target.code, index.code),
                })
            }
            TypedExprKind::BinOp { left, op, right } => {
                let left = self.expr(left)?;
                let right = self.expr(right)?;
                Ok(CExpr {
                    code: format!("({} {} {})", left.code, op.symbol(), right.code),
                })
            }
            TypedExprKind::UnaryOp { op, value } => {
                let value = self.expr(value)?;
                Ok(CExpr {
                    code: format!("({}{})", op.symbol(), value.code),
                })
            }
            TypedExprKind::Compare { left, op, right } => {
                let target_type = compare_operand_type(left, right);
                let left = self.expr(left)?;
                let right = self.expr(right)?;
                let left = cast_compare_operand(&left.code, target_type);
                let right = cast_compare_operand(&right.code, target_type);
                Ok(CExpr {
                    code: format!("({left} {} {right})", op.symbol()),
                })
            }
        }
    }

    fn intrinsic_expr(
        &mut self,
        intrinsic: IntrinsicId,
        args: &[TypedExpr],
        return_type: RumbaType,
    ) -> PyResult<CExpr> {
        let arg_codes = args
            .iter()
            .map(|arg| self.expr(arg).map(|expr| expr.code))
            .collect::<PyResult<Vec<_>>>()?;
        let code = match intrinsic {
            IntrinsicId::BuiltinLen => format!("{}.len", arg_codes[0]),
            IntrinsicId::BuiltinAbs => {
                let scalar_type = args[0].typ.as_scalar().expect("typed scalar abs");
                match scalar_type {
                    ScalarType::Float64 => format!("fabs({})", arg_codes[0]),
                    ScalarType::Int64 => {
                        let helper = self.ensure_intrinsic_helper("rumba_abs_i64", abs_i64_source);
                        format!("{helper}({})", arg_codes[0])
                    }
                    ScalarType::Bool => unreachable!("bool abs rejected by typing"),
                }
            }
            IntrinsicId::BuiltinMin | IntrinsicId::BuiltinMax => {
                let scalar_type = return_type.as_scalar().expect("typed scalar min/max");
                let helper = self.ensure_minmax_helper(intrinsic, scalar_type, args.len());
                format!("{}({})", helper, arg_codes.join(", "))
            }
            IntrinsicId::MathSqrt
            | IntrinsicId::MathSin
            | IntrinsicId::MathCos
            | IntrinsicId::MathTan
            | IntrinsicId::MathExp
            | IntrinsicId::MathLog
            | IntrinsicId::MathFloor
            | IntrinsicId::MathCeil => {
                let c_name = match intrinsic {
                    IntrinsicId::MathSqrt => "sqrt",
                    IntrinsicId::MathSin => "sin",
                    IntrinsicId::MathCos => "cos",
                    IntrinsicId::MathTan => "tan",
                    IntrinsicId::MathExp => "exp",
                    IntrinsicId::MathLog => "log",
                    IntrinsicId::MathFloor => "floor",
                    IntrinsicId::MathCeil => "ceil",
                    _ => unreachable!(),
                };
                format!("{c_name}((double)({}))", arg_codes[0])
            }
            IntrinsicId::NumpyMax | IntrinsicId::NumpyMin | IntrinsicId::NumpySum => {
                let array_type = match args[0].typ {
                    RumbaType::Array1D(element_type) => element_type,
                    RumbaType::Scalar(_) => unreachable!("numpy reductions require array"),
                };
                let helper = self.ensure_numpy_reduction_helper(intrinsic, array_type);
                format!("{helper}({})", arg_codes[0])
            }
        };
        Ok(CExpr { code })
    }

    fn ensure_intrinsic_helper(&mut self, key: &str, source_fn: impl FnOnce() -> String) -> String {
        if let Some(c_name) = self.intrinsic_cache.get(key) {
            return c_name.clone();
        }
        let c_name = key.to_string();
        self.helper_sources.push(source_fn());
        self.intrinsic_cache.insert(key.to_string(), c_name.clone());
        c_name
    }

    fn ensure_minmax_helper(
        &mut self,
        intrinsic: IntrinsicId,
        scalar_type: ScalarType,
        argc: usize,
    ) -> String {
        let op = if intrinsic == IntrinsicId::BuiltinMin {
            "min"
        } else {
            "max"
        };
        let typ = scalar_type.c_type();
        let key = format!("rumba_{op}_{}_{}", scalar_type.name(), argc);
        if let Some(c_name) = self.intrinsic_cache.get(&key) {
            return c_name.clone();
        }
        let params = (0..argc)
            .map(|index| format!("{typ} a{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let cmp = if intrinsic == IntrinsicId::BuiltinMin {
            "<"
        } else {
            ">"
        };
        let mut lines = vec![
            format!("static {typ} {key}({params}) {{"),
            "    ".to_string() + typ + " out = a0;",
        ];
        for index in 1..argc {
            lines.push(format!("    if (a{index} {cmp} out) {{ out = a{index}; }}"));
        }
        lines.push("    return out;".to_string());
        lines.push("}".to_string());
        self.helper_sources.push(format!("{}\n", lines.join("\n")));
        self.intrinsic_cache.insert(key.clone(), key.clone());
        key
    }

    fn ensure_numpy_reduction_helper(
        &mut self,
        intrinsic: IntrinsicId,
        element_type: ScalarType,
    ) -> String {
        let key = format!(
            "rumba_{}_{}",
            intrinsic.name().replace('.', "_"),
            element_type.name()
        );
        if let Some(c_name) = self.intrinsic_cache.get(&key) {
            return c_name.clone();
        }
        let source = numpy_reduction_source(&key, intrinsic, element_type);
        self.helper_sources.push(source);
        self.intrinsic_cache.insert(key.clone(), key.clone());
        key
    }
}

fn indent(level: usize) -> String {
    "    ".repeat(level)
}

fn compare_operand_type(left: &TypedExpr, right: &TypedExpr) -> ScalarType {
    let left_type = left.typ.as_scalar().expect("typed scalar comparison");
    let right_type = right.typ.as_scalar().expect("typed scalar comparison");
    if left_type == ScalarType::Float64 || right_type == ScalarType::Float64 {
        ScalarType::Float64
    } else {
        ScalarType::Int64
    }
}

fn cast_compare_operand(code: &str, target_type: ScalarType) -> String {
    match target_type {
        ScalarType::Float64 => format!("(double)({code})"),
        ScalarType::Int64 => format!("(int64_t)({code})"),
        ScalarType::Bool => unreachable!("comparison operands are normalized to numeric types"),
    }
}

fn abs_i64_source() -> String {
    [
        "static int64_t rumba_abs_i64(int64_t value) {",
        "    return value < 0 ? -value : value;",
        "}",
        "",
    ]
    .join("\n")
}

fn numpy_reduction_source(name: &str, intrinsic: IntrinsicId, element_type: ScalarType) -> String {
    let c_type = element_type.c_type();
    let array_type = match element_type {
        ScalarType::Int64 => "rumba_array_i64",
        ScalarType::Float64 => "rumba_array_f64",
        ScalarType::Bool => unreachable!("bool arrays are not supported"),
    };
    let mut lines = vec![
        format!("static {c_type} {name}({array_type} a) {{"),
        format!(
            "    {c_type} out = {};",
            if intrinsic == IntrinsicId::NumpySum {
                "0"
            } else {
                "a.data[0]"
            }
        ),
    ];
    match intrinsic {
        IntrinsicId::NumpySum => {
            lines.push("    for (int64_t i = 0; i < a.len; i++) {".to_string());
            lines.push("        out += a.data[i];".to_string());
        }
        IntrinsicId::NumpyMax => {
            lines.push("    for (int64_t i = 1; i < a.len; i++) {".to_string());
            lines.push("        if (a.data[i] > out) { out = a.data[i]; }".to_string());
        }
        IntrinsicId::NumpyMin => {
            lines.push("    for (int64_t i = 1; i < a.len; i++) {".to_string());
            lines.push("        if (a.data[i] < out) { out = a.data[i]; }".to_string());
        }
        _ => unreachable!("not a numpy reduction"),
    }
    lines.push("    }".to_string());
    lines.push("    return out;".to_string());
    lines.push("}".to_string());
    format!("{}\n", lines.join("\n"))
}

trait UnarySymbol {
    fn symbol(&self) -> &'static str;
}

impl UnarySymbol for crate::ir::UnaryOp {
    fn symbol(&self) -> &'static str {
        match self {
            Self::Not => "!",
            Self::USub => "-",
            Self::UAdd => "+",
        }
    }
}
