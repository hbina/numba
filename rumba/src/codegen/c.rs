use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;

use crate::errors::unsupported;
use crate::intrinsics::IntrinsicId;
use crate::types::{DtypeSpec, RumbaType, ScalarType, StructDtype};
use crate::typing::{
    helper_key, TypedExpr, TypedExprKind, TypedFunction, TypedPrintArg, TypedStmt,
};

struct CExpr {
    code: String,
}

struct GeneratorInline<'a> {
    names: &'a HashMap<String, String>,
    consumer_target: &'a str,
    consumer_body: &'a [TypedStmt],
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
        let struct_dtypes = collect_struct_dtypes(&self.function);
        let entry_source = self.emit_function("rumba_entry", true)?;
        let call_wrapper = self.emit_call_wrapper();
        let mut source = vec![
            "#include <stdbool.h>".to_string(),
            "#include <stdint.h>".to_string(),
            "#include <stddef.h>".to_string(),
            "#include <stdio.h>".to_string(),
            "#include <math.h>".to_string(),
            String::new(),
            "static int rumba_runtime_error = 0;".to_string(),
            String::new(),
            "typedef struct { uint8_t *data; int64_t len; } rumba_byte_buffer;".to_string(),
            "typedef struct { int64_t *data; int64_t len; } rumba_array_i64;".to_string(),
            "typedef struct { double *data; int64_t len; } rumba_array_f64;".to_string(),
            String::new(),
        ];
        source.extend(struct_dtypes.iter().map(|dtype| emit_struct_dtype(dtype)));
        if !struct_dtypes.is_empty() {
            source.push(String::new());
        }
        source.append(&mut self.helper_sources);
        source.push(entry_source);
        source.push(call_wrapper);
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
        let mut locals = self.function.locals.iter().collect::<Vec<_>>();
        locals.sort_by(|(left, _), (right, _)| left.cmp(right));
        for (name, typ) in locals {
            self.declared.insert(name.clone());
            self.lines
                .push(format!("{}{} {name};", indent(1), typ.c_type()));
        }

        let body = self.function.body.clone();
        for stmt in &body {
            self.stmt(stmt, 1)?;
        }

        self.lines.push("}".to_string());
        Ok(format!("{}\n", self.lines.join("\n")))
    }

    fn emit_call_wrapper(&self) -> String {
        let args = self
            .function
            .signature
            .iter()
            .enumerate()
            .map(|(index, typ)| format!("*({} *)args[{index}]", typ.c_type()))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "__attribute__((visibility(\"default\"))) void rumba_call(void **args, void *ret) {{\n    rumba_runtime_error = 0;\n    *({} *)ret = rumba_entry({args});\n}}\n\n__attribute__((visibility(\"default\"))) int rumba_error_status(void) {{\n    return rumba_runtime_error;\n}}\n",
            self.function.return_type.c_type()
        )
    }

    fn stmt(&mut self, node: &TypedStmt, level: usize) -> PyResult<()> {
        match node {
            TypedStmt::Return(value) => {
                let expr = self.expr(value)?;
                self.lines
                    .push(format!("{}return {};", indent(level), expr.code));
            }
            TypedStmt::Yield(_) => {
                return Err(unsupported(
                    "yield is only supported inside generator helper loops",
                ));
            }
            TypedStmt::Print(args) => self.print_stmt(args, level, None)?,
            TypedStmt::Break => {
                self.lines.push(format!("{}break;", indent(level)));
            }
            TypedStmt::Continue => {
                self.lines.push(format!("{}continue;", indent(level)));
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
                if is_frombuffer_expr(value) {
                    self.lines.push(format!(
                        "{}if (rumba_runtime_error != 0) {{ return {}; }}",
                        indent(level),
                        self.default_return_literal()
                    ));
                }
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
            TypedStmt::StoreIndexField {
                target,
                index,
                field,
                value,
                field_type,
                ..
            } => {
                let target = self.expr(target)?;
                let index = self.expr(index)?;
                let value_type = value
                    .typ
                    .as_scalar()
                    .expect("typed scalar struct field assignment");
                let value = self.expr(value)?;
                let rhs = if value_type.c_type() == field_type.c_type() {
                    value.code
                } else {
                    format!("({})({})", field_type.c_type(), value.code)
                };
                self.lines.push(format!(
                    "{}{}.data[{}].{} = {};",
                    indent(level),
                    target.code,
                    index.code,
                    field,
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
            TypedStmt::While { test, body } => {
                let test = self.expr(test)?;
                self.lines
                    .push(format!("{}while ({}) {{", indent(level), test.code));
                for child in body {
                    self.stmt(child, level + 1)?;
                }
                self.lines.push(format!("{}}}", indent(level)));
            }
            TypedStmt::ForRange {
                target,
                start,
                stop,
                step,
                body,
            } => self.for_range(target, start, stop, step, body, level)?,
            TypedStmt::ForGenerator {
                target,
                function,
                args,
                body,
            } => {
                let prefix = format!("__rumba_gen{}_", self.helper_cache.len());
                let mut names = HashMap::new();
                for name in function.args.iter().chain(function.locals.keys()) {
                    names.insert(name.clone(), format!("{prefix}{name}"));
                }
                self.lines.push(format!("{}{{", indent(level)));
                for (name, arg) in function.args.iter().zip(args.iter()) {
                    let c_name = names.get(name).expect("generator arg has renamed local");
                    let expr = self.expr(arg)?;
                    self.lines.push(format!(
                        "{}{} {c_name} = {};",
                        indent(level + 1),
                        arg.typ.c_type(),
                        expr.code
                    ));
                    self.declared.insert(c_name.clone());
                }
                let inline = GeneratorInline {
                    names: &names,
                    consumer_target: target,
                    consumer_body: body,
                };
                for stmt in &function.body {
                    self.generator_stmt(stmt, level + 1, &inline)?;
                }
                self.lines.push(format!("{}}}", indent(level)));
            }
        }
        Ok(())
    }

    fn generator_stmt(
        &mut self,
        node: &TypedStmt,
        level: usize,
        inline: &GeneratorInline<'_>,
    ) -> PyResult<()> {
        match node {
            TypedStmt::Yield(value) => {
                let expr = self.expr_renamed(value, inline.names)?;
                let prefix = if self.declared.contains(inline.consumer_target) {
                    String::new()
                } else {
                    self.declared.insert(inline.consumer_target.to_string());
                    format!("{} ", value.typ.c_type())
                };
                self.lines.push(format!(
                    "{}{prefix}{} = {};",
                    indent(level),
                    inline.consumer_target,
                    expr.code
                ));
                if is_frombuffer_expr(value) {
                    self.lines.push(format!(
                        "{}if (rumba_runtime_error != 0) {{ return {}; }}",
                        indent(level),
                        self.default_return_literal()
                    ));
                }
                for stmt in inline.consumer_body {
                    self.stmt(stmt, level)?;
                }
            }
            TypedStmt::Return(_) => {
                return Err(unsupported(
                    "return values from generator helpers are not supported",
                ));
            }
            TypedStmt::Print(args) => self.print_stmt(args, level, Some(inline.names))?,
            TypedStmt::Break => self.lines.push(format!("{}break;", indent(level))),
            TypedStmt::Continue => self.lines.push(format!("{}continue;", indent(level))),
            TypedStmt::Assign { name, value } => {
                let expr = self.expr_renamed(value, inline.names)?;
                let name = inline.names.get(name).map(String::as_str).unwrap_or(name);
                let prefix = if self.declared.contains(name) {
                    String::new()
                } else {
                    self.declared.insert(name.to_string());
                    format!("{} ", value.typ.c_type())
                };
                self.lines
                    .push(format!("{}{prefix}{name} = {};", indent(level), expr.code));
                if is_frombuffer_expr(value) {
                    self.lines.push(format!(
                        "{}if (rumba_runtime_error != 0) {{ return {}; }}",
                        indent(level),
                        self.default_return_literal()
                    ));
                }
            }
            TypedStmt::AugAssign {
                name, op, value, ..
            } => {
                let expr = self.expr_renamed(value, inline.names)?;
                let name = inline.names.get(name).map(String::as_str).unwrap_or(name);
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
                let target = self.expr_renamed(target, inline.names)?;
                let index = self.expr_renamed(index, inline.names)?;
                let value_type = value
                    .typ
                    .as_scalar()
                    .expect("typed scalar array assignment");
                let value = self.expr_renamed(value, inline.names)?;
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
            TypedStmt::StoreIndexField {
                target,
                index,
                field,
                value,
                field_type,
                ..
            } => {
                let target = self.expr_renamed(target, inline.names)?;
                let index = self.expr_renamed(index, inline.names)?;
                let value_type = value
                    .typ
                    .as_scalar()
                    .expect("typed scalar struct field assignment");
                let value = self.expr_renamed(value, inline.names)?;
                let rhs = if value_type.c_type() == field_type.c_type() {
                    value.code
                } else {
                    format!("({})({})", field_type.c_type(), value.code)
                };
                self.lines.push(format!(
                    "{}{}.data[{}].{} = {};",
                    indent(level),
                    target.code,
                    index.code,
                    field,
                    rhs
                ));
            }
            TypedStmt::If { test, body, orelse } => {
                let test = self.expr_renamed(test, inline.names)?;
                self.lines
                    .push(format!("{}if ({}) {{", indent(level), test.code));
                for child in body {
                    self.generator_stmt(child, level + 1, inline)?;
                }
                if !orelse.is_empty() {
                    self.lines.push(format!("{}}} else {{", indent(level)));
                    for child in orelse {
                        self.generator_stmt(child, level + 1, inline)?;
                    }
                }
                self.lines.push(format!("{}}}", indent(level)));
            }
            TypedStmt::While { test, body } => {
                let test = self.expr_renamed(test, inline.names)?;
                self.lines
                    .push(format!("{}while ({}) {{", indent(level), test.code));
                for child in body {
                    self.generator_stmt(child, level + 1, inline)?;
                }
                self.lines.push(format!("{}}}", indent(level)));
            }
            TypedStmt::ForRange {
                target,
                start,
                stop,
                step,
                body,
            } => {
                let start = self.expr_renamed(start, inline.names)?;
                let stop = self.expr_renamed(stop, inline.names)?;
                let step = self.expr_renamed(step, inline.names)?;
                let name = inline
                    .names
                    .get(target)
                    .map(String::as_str)
                    .unwrap_or(target);
                let decl = if self.declared.contains(name) {
                    String::new()
                } else {
                    self.declared.insert(name.to_string());
                    "int64_t ".to_string()
                };
                let cmp = format!(
                    "(({}) > 0 ? {name} < {} : {name} > {})",
                    step.code, stop.code, stop.code
                );
                self.lines.push(format!(
                    "{}for ({decl}{name} = {}; {cmp}; {name} += {}) {{",
                    indent(level),
                    start.code,
                    step.code
                ));
                for child in body {
                    self.generator_stmt(child, level + 1, inline)?;
                }
                self.lines.push(format!("{}}}", indent(level)));
            }
            TypedStmt::ForGenerator { .. } => {
                return Err(unsupported(
                    "nested generator helper loops are not supported",
                ));
            }
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
        let cmp = format!("(({step}) > 0 ? {name} < {stop} : {name} > {stop})");
        self.lines.push(format!(
            "{}for ({decl}{name} = {start}; {cmp}; {name} += {step}) {{",
            indent(level)
        ));
        for child in body {
            self.stmt(child, level + 1)?;
        }
        self.lines.push(format!("{}}}", indent(level)));
        Ok(())
    }

    fn print_stmt(
        &mut self,
        args: &[TypedPrintArg],
        level: usize,
        names: Option<&HashMap<String, String>>,
    ) -> PyResult<()> {
        for (index, arg) in args.iter().enumerate() {
            if index > 0 {
                self.lines
                    .push(format!("{}fputc(' ', stdout);", indent(level)));
            }
            match arg {
                TypedPrintArg::StaticStr(value) => {
                    self.lines.push(format!(
                        "{}fputs({}, stdout);",
                        indent(level),
                        c_string_literal(value)
                    ));
                }
                TypedPrintArg::Expr(expr) => {
                    let code = if let Some(names) = names {
                        self.expr_renamed(expr, names)?.code
                    } else {
                        self.expr(expr)?.code
                    };
                    let scalar_type = expr.typ.as_scalar().expect("typed print scalar argument");
                    let line = match scalar_type {
                        ScalarType::Int64 => format!("printf(\"%lld\", (long long)({code}));"),
                        ScalarType::Float64 => format!("printf(\"%.17g\", (double)({code}));"),
                        ScalarType::Bool => {
                            format!("fputs(({code}) ? \"True\" : \"False\", stdout);")
                        }
                    };
                    self.lines.push(format!("{}{}", indent(level), line));
                }
            }
        }
        self.lines
            .push(format!("{}fputc('\\n', stdout);", indent(level)));
        self.lines.push(format!("{}fflush(stdout);", indent(level)));
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
                    crate::ir::ConstantValue::Str(_) => {
                        return Err(unsupported(
                            "string constants are only valid as struct field keys (a[i]['field'])",
                        ));
                    }
                },
            }),
            TypedExprKind::Dtype(_) => Err(unsupported(
                "numpy dtype objects are only supported as np.frombuffer dtype arguments",
            )),
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
                self.intrinsic_expr(*intrinsic, args, node.typ.clone())
            }
            TypedExprKind::Index { target, index } => {
                let target = self.expr(target)?;
                let index = self.expr(index)?;
                Ok(CExpr {
                    code: format!("{}.data[{}]", target.code, index.code),
                })
            }
            TypedExprKind::IndexField {
                target,
                index,
                field,
                ..
            } => {
                let target = self.expr(target)?;
                let index = self.expr(index)?;
                Ok(CExpr {
                    code: format!("{}.data[{}].{}", target.code, index.code, field),
                })
            }
            TypedExprKind::BinOp { left, op, right } => {
                let left_type = left.typ.as_scalar();
                let right_type = right.typ.as_scalar();
                let left = self.expr(left)?;
                let right = self.expr(right)?;
                if *op == crate::ir::BinOp::Div
                    && left_type == Some(ScalarType::Int64)
                    && right_type == Some(ScalarType::Int64)
                {
                    return Ok(CExpr {
                        code: format!("((double)({}) / (double)({}))", left.code, right.code),
                    });
                }
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
                let left = self.expr(left)?;
                let right = self.expr(right)?;
                Ok(CExpr {
                    code: format!("({} {} {})", left.code, op.symbol(), right.code),
                })
            }
        }
    }

    fn expr_renamed(
        &mut self,
        node: &TypedExpr,
        names: &HashMap<String, String>,
    ) -> PyResult<CExpr> {
        self.expr(&rename_expr(node, names))
    }

    fn intrinsic_expr(
        &mut self,
        intrinsic: IntrinsicId,
        args: &[TypedExpr],
        return_type: RumbaType,
    ) -> PyResult<CExpr> {
        if intrinsic == IntrinsicId::NumpyFromBuffer {
            let buffer = self.expr(&args[0])?.code;
            let count = self.expr(&args[2])?.code;
            let offset = self.expr(&args[3])?.code;
            let dtype = match &args[1].kind {
                TypedExprKind::Dtype(dtype) => dtype,
                _ => {
                    return Err(unsupported(
                        "np.frombuffer dtype must be a module-level numpy dtype object",
                    ));
                }
            };
            let helper = self.ensure_frombuffer_helper(dtype);
            return Ok(CExpr {
                code: format!("{helper}({buffer}, {count}, {offset})"),
            });
        }
        let arg_codes = args
            .iter()
            .map(|arg| self.expr(arg).map(|expr| expr.code))
            .collect::<PyResult<Vec<_>>>()?;
        let code = match intrinsic {
            IntrinsicId::BuiltinPrint => {
                unreachable!("print is emitted as a statement")
            }
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
            IntrinsicId::BuiltinInt | IntrinsicId::BuiltinFloat | IntrinsicId::BuiltinBool => {
                match intrinsic {
                    IntrinsicId::BuiltinInt => format!("(int64_t)({})", arg_codes[0]),
                    IntrinsicId::BuiltinFloat => format!("(double)({})", arg_codes[0]),
                    IntrinsicId::BuiltinBool => format!("(({}) != 0)", arg_codes[0]),
                    _ => unreachable!(),
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
                format!("{c_name}({})", arg_codes[0])
            }
            IntrinsicId::NumpyMax | IntrinsicId::NumpyMin | IntrinsicId::NumpySum => {
                let array_type = match args[0].typ {
                    RumbaType::Array1D(element_type) => element_type,
                    RumbaType::Array1DStruct(_) => {
                        unreachable!("numpy reductions reject structured arrays during typing")
                    }
                    RumbaType::Scalar(_) | RumbaType::ByteBuffer | RumbaType::Dtype(_) => {
                        unreachable!("numpy reductions require array")
                    }
                };
                let helper = self.ensure_numpy_reduction_helper(intrinsic, array_type);
                format!("{helper}({})", arg_codes[0])
            }
            IntrinsicId::NumpyFromBuffer => {
                unreachable!("np.frombuffer is handled before generic intrinsic argument emission")
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

    fn ensure_frombuffer_helper(&mut self, dtype: &DtypeSpec) -> String {
        let (array_type, element_type, itemsize, key_part) = match dtype {
            DtypeSpec::Scalar(ScalarType::Int64) => (
                "rumba_array_i64".to_string(),
                "int64_t".to_string(),
                8,
                "i64".to_string(),
            ),
            DtypeSpec::Scalar(ScalarType::Float64) => (
                "rumba_array_f64".to_string(),
                "double".to_string(),
                8,
                "f64".to_string(),
            ),
            DtypeSpec::Scalar(ScalarType::Bool) => {
                unreachable!("bool dtype rejected by np.frombuffer typing")
            }
            DtypeSpec::Struct(dtype) => (
                dtype.c_array_name.clone(),
                dtype.c_struct_name.clone(),
                dtype.itemsize,
                dtype.c_struct_name.clone(),
            ),
        };
        let key = format!("rumba_numpy_frombuffer_{key_part}");
        if let Some(c_name) = self.intrinsic_cache.get(&key) {
            return c_name.clone();
        }
        let source = frombuffer_source(&key, &array_type, &element_type, itemsize);
        self.helper_sources.push(source);
        self.intrinsic_cache.insert(key.clone(), key.clone());
        key
    }

    fn default_return_literal(&self) -> &'static str {
        match self.function.return_type {
            ScalarType::Int64 => "0",
            ScalarType::Float64 => "0.0",
            ScalarType::Bool => "false",
        }
    }
}

fn is_frombuffer_expr(expr: &TypedExpr) -> bool {
    matches!(
        expr.kind,
        TypedExprKind::IntrinsicCall {
            intrinsic: IntrinsicId::NumpyFromBuffer,
            ..
        }
    )
}

fn collect_struct_dtypes(function: &TypedFunction) -> Vec<StructDtype> {
    let mut out = Vec::new();
    collect_struct_dtypes_from_function(function, &mut out);
    out
}

fn rename_expr(expr: &TypedExpr, names: &HashMap<String, String>) -> TypedExpr {
    let kind = match &expr.kind {
        TypedExprKind::Name(name) => {
            TypedExprKind::Name(names.get(name).cloned().unwrap_or_else(|| name.clone()))
        }
        TypedExprKind::Dtype(dtype) => TypedExprKind::Dtype(dtype.clone()),
        TypedExprKind::Call { function, args } => TypedExprKind::Call {
            function: function.clone(),
            args: args.iter().map(|arg| rename_expr(arg, names)).collect(),
        },
        TypedExprKind::IntrinsicCall { intrinsic, args } => TypedExprKind::IntrinsicCall {
            intrinsic: *intrinsic,
            args: args.iter().map(|arg| rename_expr(arg, names)).collect(),
        },
        TypedExprKind::Index { target, index } => TypedExprKind::Index {
            target: Box::new(rename_expr(target, names)),
            index: Box::new(rename_expr(index, names)),
        },
        TypedExprKind::IndexField {
            target,
            index,
            field,
            field_type,
        } => TypedExprKind::IndexField {
            target: Box::new(rename_expr(target, names)),
            index: Box::new(rename_expr(index, names)),
            field: field.clone(),
            field_type: *field_type,
        },
        TypedExprKind::BinOp { left, op, right } => TypedExprKind::BinOp {
            left: Box::new(rename_expr(left, names)),
            op: *op,
            right: Box::new(rename_expr(right, names)),
        },
        TypedExprKind::UnaryOp { op, value } => TypedExprKind::UnaryOp {
            op: op.clone(),
            value: Box::new(rename_expr(value, names)),
        },
        TypedExprKind::Compare { left, op, right } => TypedExprKind::Compare {
            left: Box::new(rename_expr(left, names)),
            op: op.clone(),
            right: Box::new(rename_expr(right, names)),
        },
        TypedExprKind::Constant(value) => TypedExprKind::Constant(value.clone()),
    };
    TypedExpr {
        kind,
        typ: expr.typ.clone(),
    }
}

fn collect_struct_dtypes_from_function(function: &TypedFunction, out: &mut Vec<StructDtype>) {
    for typ in function.signature.iter().chain(function.locals.values()) {
        collect_struct_dtype_from_type(typ, out);
    }
    for stmt in &function.body {
        collect_struct_dtypes_from_stmt(stmt, out);
    }
}

fn collect_struct_dtypes_from_stmt(stmt: &TypedStmt, out: &mut Vec<StructDtype>) {
    match stmt {
        TypedStmt::Return(value) | TypedStmt::Yield(value) | TypedStmt::Assign { value, .. } => {
            collect_struct_dtypes_from_expr(value, out);
        }
        TypedStmt::Print(args) => {
            for arg in args {
                if let TypedPrintArg::Expr(expr) = arg {
                    collect_struct_dtypes_from_expr(expr, out);
                }
            }
        }
        TypedStmt::Break | TypedStmt::Continue => {}
        TypedStmt::AugAssign {
            value, target_type, ..
        } => {
            collect_struct_dtype_from_type(target_type, out);
            collect_struct_dtypes_from_expr(value, out);
        }
        TypedStmt::StoreIndex {
            target,
            index,
            value,
            ..
        } => {
            collect_struct_dtypes_from_expr(target, out);
            collect_struct_dtypes_from_expr(index, out);
            collect_struct_dtypes_from_expr(value, out);
        }
        TypedStmt::StoreIndexField {
            target,
            index,
            value,
            dtype,
            ..
        } => {
            collect_struct_dtype(dtype, out);
            collect_struct_dtypes_from_expr(target, out);
            collect_struct_dtypes_from_expr(index, out);
            collect_struct_dtypes_from_expr(value, out);
        }
        TypedStmt::If { test, body, orelse } => {
            collect_struct_dtypes_from_expr(test, out);
            for stmt in body.iter().chain(orelse.iter()) {
                collect_struct_dtypes_from_stmt(stmt, out);
            }
        }
        TypedStmt::While { test, body } => {
            collect_struct_dtypes_from_expr(test, out);
            for stmt in body {
                collect_struct_dtypes_from_stmt(stmt, out);
            }
        }
        TypedStmt::ForRange {
            start,
            stop,
            step,
            body,
            ..
        } => {
            collect_struct_dtypes_from_expr(start, out);
            collect_struct_dtypes_from_expr(stop, out);
            collect_struct_dtypes_from_expr(step, out);
            for stmt in body {
                collect_struct_dtypes_from_stmt(stmt, out);
            }
        }
        TypedStmt::ForGenerator {
            function,
            args,
            body,
            ..
        } => {
            for typ in function.signature.iter().chain(function.locals.values()) {
                collect_struct_dtype_from_type(typ, out);
            }
            for stmt in &function.body {
                collect_struct_dtypes_from_stmt(stmt, out);
            }
            for arg in args {
                collect_struct_dtypes_from_expr(arg, out);
            }
            for stmt in body {
                collect_struct_dtypes_from_stmt(stmt, out);
            }
        }
    }
}

fn collect_struct_dtypes_from_expr(expr: &TypedExpr, out: &mut Vec<StructDtype>) {
    collect_struct_dtype_from_type(&expr.typ, out);
    match &expr.kind {
        TypedExprKind::Call { function, args } => {
            collect_struct_dtypes_from_function(function, out);
            for arg in args {
                collect_struct_dtypes_from_expr(arg, out);
            }
        }
        TypedExprKind::IntrinsicCall { args, .. } => {
            for arg in args {
                collect_struct_dtypes_from_expr(arg, out);
            }
        }
        TypedExprKind::Index { target, index }
        | TypedExprKind::IndexField { target, index, .. } => {
            collect_struct_dtypes_from_expr(target, out);
            collect_struct_dtypes_from_expr(index, out);
        }
        TypedExprKind::BinOp { left, right, .. } | TypedExprKind::Compare { left, right, .. } => {
            collect_struct_dtypes_from_expr(left, out);
            collect_struct_dtypes_from_expr(right, out);
        }
        TypedExprKind::UnaryOp { value, .. } => collect_struct_dtypes_from_expr(value, out),
        TypedExprKind::Constant(_) | TypedExprKind::Dtype(_) | TypedExprKind::Name(_) => {}
    }
}

fn collect_struct_dtype_from_type(typ: &RumbaType, out: &mut Vec<StructDtype>) {
    if let RumbaType::Array1DStruct(dtype) = typ {
        collect_struct_dtype(dtype, out);
    }
}

fn collect_struct_dtype(dtype: &StructDtype, out: &mut Vec<StructDtype>) {
    if !out.iter().any(|existing| existing == dtype) {
        out.push(dtype.clone());
    }
}

fn emit_struct_dtype(dtype: &StructDtype) -> String {
    let mut lines = vec![format!("typedef struct {} {{", dtype.c_struct_name)];
    let mut cursor = 0_usize;
    let mut pad_index = 0_usize;
    for ((name, field_type), offset) in dtype.fields.iter().zip(dtype.offsets.iter()) {
        if *offset > cursor {
            lines.push(format!("    uint8_t _pad{pad_index}[{}];", offset - cursor));
            pad_index += 1;
            cursor = *offset;
        }
        lines.push(format!("    {} {};", field_type.c_type(), name));
        cursor += field_type.byte_size();
    }
    if dtype.itemsize > cursor {
        lines.push(format!(
            "    uint8_t _pad{pad_index}[{}];",
            dtype.itemsize - cursor
        ));
    }
    lines.push(format!("}} {};", dtype.c_struct_name));
    lines.push(format!(
        "typedef struct {{ {} *data; int64_t len; }} {};",
        dtype.c_struct_name, dtype.c_array_name
    ));
    lines.push(format!(
        "_Static_assert(sizeof({}) == {}, \"{} size mismatch\");",
        dtype.c_struct_name, dtype.itemsize, dtype.c_struct_name
    ));
    for ((name, _), offset) in dtype.fields.iter().zip(dtype.offsets.iter()) {
        lines.push(format!(
            "_Static_assert(offsetof({}, {}) == {}, \"{} offset mismatch\");",
            dtype.c_struct_name, name, offset, name
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}

fn indent(level: usize) -> String {
    "    ".repeat(level)
}

fn c_string_literal(value: &str) -> String {
    let mut out = String::from("\"");
    for byte in value.bytes() {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7e => out.push(byte as char),
            byte => out.push_str(&format!("\\{byte:03o}")),
        }
    }
    out.push('"');
    out
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

fn frombuffer_source(name: &str, array_type: &str, element_type: &str, itemsize: usize) -> String {
    [
        format!(
            "static {array_type} {name}(rumba_byte_buffer buf, int64_t count, int64_t offset) {{"
        ),
        format!("    {array_type} out;"),
        "    out.data = 0;".to_string(),
        "    out.len = 0;".to_string(),
        "    if (offset < 0 || count < 0) {".to_string(),
        "        rumba_runtime_error = 1;".to_string(),
        "        return out;".to_string(),
        "    }".to_string(),
        "    if (offset > buf.len || count > ((buf.len - offset) / ".to_string()
            + &itemsize.to_string()
            + ")) {",
        "        rumba_runtime_error = 1;".to_string(),
        "        return out;".to_string(),
        "    }".to_string(),
        format!("    out.data = ({element_type} *)(buf.data + offset);"),
        "    out.len = count;".to_string(),
        "    return out;".to_string(),
        "}".to_string(),
    ]
    .join("\n")
        + "\n"
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
