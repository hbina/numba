use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::prelude::*;

use crate::errors::unsupported;
use crate::intrinsics::IntrinsicId;
use crate::ir::{BinOp, CallTarget, ConstantValue, ExprNode, ParsedFunction, StmtNode, UnaryOp};
use crate::types::{format_signature, DtypeSpec, FieldType, RumbaType, ScalarType, StructDtype};

#[derive(Clone, Debug)]
pub(crate) struct TypedFunction {
    pub(crate) name: String,
    pub(crate) args: Vec<String>,
    pub(crate) signature: Vec<RumbaType>,
    pub(crate) body: Vec<TypedStmt>,
    pub(crate) return_type: ScalarType,
    pub(crate) locals: HashMap<String, RumbaType>,
}

#[derive(Clone, Debug)]
pub(crate) struct TypedGeneratorFunction {
    pub(crate) name: String,
    pub(crate) args: Vec<String>,
    pub(crate) signature: Vec<RumbaType>,
    pub(crate) body: Vec<TypedStmt>,
    pub(crate) yield_type: RumbaType,
    pub(crate) locals: HashMap<String, RumbaType>,
}

#[derive(Clone, Debug)]
pub(crate) enum TypedStmt {
    Return(TypedExpr),
    Yield(TypedExpr),
    Print(Vec<TypedPrintArg>),
    Break,
    Continue,
    Assign {
        name: String,
        value: TypedExpr,
    },
    AugAssign {
        name: String,
        op: BinOp,
        value: TypedExpr,
        target_type: RumbaType,
    },
    StoreIndex {
        target: TypedExpr,
        index: TypedExpr,
        value: TypedExpr,
        element_type: ScalarType,
    },
    StoreIndexField {
        target: TypedExpr,
        index: TypedExpr,
        field: String,
        value: TypedExpr,
        field_type: FieldType,
        dtype: Arc<StructDtype>,
    },
    If {
        test: TypedExpr,
        body: Vec<TypedStmt>,
        orelse: Vec<TypedStmt>,
    },
    While {
        test: TypedExpr,
        body: Vec<TypedStmt>,
    },
    ForRange {
        target: String,
        start: TypedExpr,
        stop: TypedExpr,
        step: TypedExpr,
        body: Vec<TypedStmt>,
    },
    ForGenerator {
        target: String,
        function: Box<TypedGeneratorFunction>,
        args: Vec<TypedExpr>,
        body: Vec<TypedStmt>,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum TypedPrintArg {
    StaticStr(String),
    Expr(TypedExpr),
}

#[derive(Clone, Debug)]
pub(crate) struct TypedExpr {
    pub(crate) kind: TypedExprKind,
    pub(crate) typ: RumbaType,
}

#[derive(Clone, Debug)]
pub(crate) enum TypedExprKind {
    Constant(ConstantValue),
    Dtype(DtypeSpec),
    Name(String),
    Call {
        function: Box<TypedFunction>,
        args: Vec<TypedExpr>,
    },
    IntrinsicCall {
        intrinsic: IntrinsicId,
        args: Vec<TypedExpr>,
    },
    Index {
        target: Box<TypedExpr>,
        index: Box<TypedExpr>,
    },
    IndexField {
        target: Box<TypedExpr>,
        index: Box<TypedExpr>,
        field: String,
        field_type: FieldType,
    },
    BinOp {
        left: Box<TypedExpr>,
        op: BinOp,
        right: Box<TypedExpr>,
    },
    UnaryOp {
        op: UnaryOp,
        value: Box<TypedExpr>,
    },
    Compare {
        left: Box<TypedExpr>,
        op: crate::ir::CmpOp,
        right: Box<TypedExpr>,
    },
}

pub(crate) fn type_function(
    function_ir: ParsedFunction,
    signature: Vec<RumbaType>,
) -> PyResult<TypedFunction> {
    if function_ir.args.len() != signature.len() {
        return Err(unsupported("argument count does not match signature"));
    }

    let ParsedFunction { name, args, body } = function_ir;
    let env = args
        .iter()
        .cloned()
        .zip(signature.iter().cloned())
        .collect::<HashMap<_, _>>();
    let mut pass = TypePass {
        env,
        return_type: None,
        helper_cache: HashMap::new(),
        branch_only: HashSet::new(),
        loop_depth: 0,
        in_generator: false,
        yield_type: None,
        generator_cache: HashMap::new(),
        frombuffer_locals: HashSet::new(),
    };

    let typed_body = body
        .iter()
        .map(|stmt| pass.stmt(stmt))
        .collect::<PyResult<Vec<_>>>()?;

    let return_type = pass
        .return_type
        .ok_or_else(|| unsupported("function must return a scalar value"))?;

    Ok(TypedFunction {
        name,
        locals: local_types(&args, &pass.env),
        args,
        signature,
        body: typed_body,
        return_type,
    })
}

fn type_generator_function(
    function_ir: ParsedFunction,
    signature: Vec<RumbaType>,
) -> PyResult<TypedGeneratorFunction> {
    if function_ir.args.len() != signature.len() {
        return Err(unsupported("argument count does not match signature"));
    }

    let ParsedFunction { name, args, body } = function_ir;
    if !contains_yield(&body) {
        return Err(unsupported("generator helper loops require yield"));
    }
    let env = args
        .iter()
        .cloned()
        .zip(signature.iter().cloned())
        .collect::<HashMap<_, _>>();
    let mut pass = TypePass {
        env,
        return_type: None,
        helper_cache: HashMap::new(),
        branch_only: HashSet::new(),
        loop_depth: 0,
        in_generator: true,
        yield_type: None,
        generator_cache: HashMap::new(),
        frombuffer_locals: HashSet::new(),
    };

    let typed_body = body
        .iter()
        .map(|stmt| pass.stmt(stmt))
        .collect::<PyResult<Vec<_>>>()?;
    let yield_type = pass
        .yield_type
        .ok_or_else(|| unsupported("generator helper must yield a value"))?;

    Ok(TypedGeneratorFunction {
        name,
        locals: local_types(&args, &pass.env),
        args,
        signature,
        body: typed_body,
        yield_type,
    })
}

fn local_types(args: &[String], env: &HashMap<String, RumbaType>) -> HashMap<String, RumbaType> {
    env.iter()
        .filter(|(name, _)| !args.contains(name))
        .map(|(name, typ)| (name.clone(), typ.clone()))
        .collect()
}

struct TypePass {
    env: HashMap<String, RumbaType>,
    return_type: Option<ScalarType>,
    helper_cache: HashMap<String, TypedFunction>,
    generator_cache: HashMap<String, TypedGeneratorFunction>,
    branch_only: HashSet<String>,
    frombuffer_locals: HashSet<String>,
    loop_depth: usize,
    in_generator: bool,
    yield_type: Option<RumbaType>,
}

impl TypePass {
    fn stmt(&mut self, node: &StmtNode) -> PyResult<TypedStmt> {
        match node {
            StmtNode::Return(value) => {
                if self.in_generator {
                    return Err(unsupported(
                        "return values from generator helpers are not supported",
                    ));
                }
                let expr = self.expr(value)?;
                let expr_typ = expr
                    .typ
                    .as_scalar()
                    .ok_or_else(|| unsupported("array return values are not supported"))?;
                self.return_type = Some(match self.return_type {
                    None => expr_typ,
                    Some(current) if current == expr_typ => current,
                    Some(current) => {
                        return Err(exact_type_mismatch("return values", current, expr_typ));
                    }
                });
                Ok(TypedStmt::Return(expr))
            }
            StmtNode::Yield(value) => {
                if !self.in_generator {
                    return Err(unsupported("yield is only supported in generator helpers consumed directly by for loops"));
                }
                let expr = self.expr(value)?;
                if expr.typ.as_scalar().is_none() && !self.is_frombuffer_view(&expr) {
                    return Err(unsupported(
                        "generator array yields are only supported for np.frombuffer views",
                    ));
                }
                self.yield_type = Some(match &self.yield_type {
                    None => expr.typ.clone(),
                    Some(current) if current == &expr.typ => current.clone(),
                    Some(current) => {
                        return Err(exact_rumba_type_mismatch(
                            "generator yield values",
                            current,
                            &expr.typ,
                        ));
                    }
                });
                Ok(TypedStmt::Yield(expr))
            }
            StmtNode::Print(args) => args
                .iter()
                .map(|arg| self.print_arg(arg))
                .collect::<PyResult<Vec<_>>>()
                .map(TypedStmt::Print),
            StmtNode::Break => {
                if self.loop_depth == 0 {
                    return Err(unsupported("break is only supported inside loops"));
                }
                Ok(TypedStmt::Break)
            }
            StmtNode::Continue => {
                if self.loop_depth == 0 {
                    return Err(unsupported("continue is only supported inside loops"));
                }
                Ok(TypedStmt::Continue)
            }
            StmtNode::Assign { name, value } => {
                let expr = self.expr(value)?;
                if is_frombuffer_expr(&expr) {
                    self.frombuffer_locals.insert(name.clone());
                } else {
                    self.frombuffer_locals.remove(name);
                }
                self.env.insert(name.clone(), expr.typ.clone());
                Ok(TypedStmt::Assign {
                    name: name.clone(),
                    value: expr,
                })
            }
            StmtNode::AugAssign { name, op, value } => {
                let target_type = self.env.get(name).cloned().ok_or_else(|| {
                    unsupported(format!("local variable {name:?} is used before assignment"))
                })?;
                if target_type.as_scalar().is_none() {
                    return Err(unsupported("augmented assignment requires a scalar target"));
                }
                let expr = self.expr(value)?;
                if expr.typ.as_scalar().is_none() {
                    return Err(unsupported("augmented assignment requires a scalar value"));
                }
                let target_scalar = target_type
                    .as_scalar()
                    .expect("augmented assignment target checked as scalar");
                let value_scalar = expr
                    .typ
                    .as_scalar()
                    .expect("augmented assignment value checked as scalar");
                let result_type = type_binary_op(target_scalar, *op, value_scalar)?;
                if result_type != target_scalar {
                    return Err(unsupported(format!(
                        "augmented assignment for local {name:?} produces {} but target is {}; use an explicit cast",
                        result_type.name(),
                        target_scalar.name()
                    )));
                }
                Ok(TypedStmt::AugAssign {
                    name: name.clone(),
                    op: *op,
                    value: expr,
                    target_type,
                })
            }
            StmtNode::StoreIndex {
                target,
                index,
                value,
            } => {
                let target = self.expr(target)?;
                let index = self.expr(index)?;
                let value = self.expr(value)?;
                let element_type = match &target.typ {
                    RumbaType::Array1D(typ) => *typ,
                    RumbaType::Array1DStruct(_) => {
                        return Err(unsupported(
                            "use a[i]['field'] syntax to assign struct array fields",
                        ));
                    }
                    RumbaType::Scalar(_) => {
                        return Err(unsupported("indexed assignment requires an array target"));
                    }
                    RumbaType::ByteBuffer | RumbaType::Dtype(_) => {
                        return Err(unsupported("indexed assignment requires an array target"));
                    }
                };
                if self.is_frombuffer_view(&target) {
                    return Err(unsupported(
                        "assigning through np.frombuffer views is not supported",
                    ));
                }
                if index.typ != RumbaType::Scalar(ScalarType::Int64) {
                    return Err(unsupported("array index must be an int64 scalar"));
                }
                if value.typ.as_scalar().is_none() {
                    return Err(unsupported("array assignment value must be scalar"));
                }
                Ok(TypedStmt::StoreIndex {
                    target,
                    index,
                    value,
                    element_type,
                })
            }
            StmtNode::StoreIndexField {
                target,
                index,
                field,
                value,
            } => {
                let target = self.expr(target)?;
                let index = self.expr(index)?;
                let value = self.expr(value)?;
                let dtype = match &target.typ {
                    RumbaType::Array1DStruct(dtype) => dtype.clone(),
                    _ => {
                        return Err(unsupported(
                            "field assignment requires a structured array target",
                        ));
                    }
                };
                if self.is_frombuffer_view(&target) {
                    return Err(unsupported(
                        "assigning through np.frombuffer views is not supported",
                    ));
                }
                if index.typ != RumbaType::Scalar(ScalarType::Int64) {
                    return Err(unsupported("struct array index must be an int64 scalar"));
                }
                let (field_type, _) = dtype.field(field).ok_or_else(|| {
                    unsupported(format!(
                        "field {field:?} does not exist in structured dtype {}",
                        dtype.c_struct_name
                    ))
                })?;
                if value.typ.as_scalar().is_none() {
                    return Err(unsupported("struct field assignment value must be scalar"));
                }
                Ok(TypedStmt::StoreIndexField {
                    target,
                    index,
                    field: field.clone(),
                    value,
                    field_type,
                    dtype,
                })
            }
            StmtNode::If { test, body, orelse } => {
                let test = self.expr(test)?;
                if test.typ != RumbaType::Scalar(ScalarType::Bool) {
                    return Err(unsupported("if condition must be boolean"));
                }

                let env_before = self.env.clone();
                let branch_only_before = self.branch_only.clone();
                let return_type_before = self.return_type;
                let body_continues = stmts_may_continue(body);
                let else_continues = if orelse.is_empty() {
                    true
                } else {
                    stmts_may_continue(orelse)
                };

                self.env = env_before.clone();
                self.branch_only = branch_only_before.clone();
                self.return_type = return_type_before;
                let yield_type_before = self.yield_type.clone();
                self.yield_type = yield_type_before.clone();
                let typed_body = body
                    .iter()
                    .map(|stmt| self.stmt(stmt))
                    .collect::<PyResult<Vec<_>>>()?;
                let body_env = self.env.clone();
                let body_return_type = self.return_type;
                let body_yield_type = self.yield_type.clone();
                let body_helper_cache = self.helper_cache.clone();
                let body_generator_cache = self.generator_cache.clone();
                let body_frombuffer_locals = self.frombuffer_locals.clone();

                self.env = env_before.clone();
                self.branch_only = branch_only_before.clone();
                self.return_type = return_type_before;
                self.yield_type = yield_type_before.clone();
                let typed_orelse = orelse
                    .iter()
                    .map(|stmt| self.stmt(stmt))
                    .collect::<PyResult<Vec<_>>>()?;
                let else_env = self.env.clone();
                let else_return_type = self.return_type;
                let else_yield_type = self.yield_type.clone();
                let else_helper_cache = self.helper_cache.clone();
                let else_generator_cache = self.generator_cache.clone();
                let else_frombuffer_locals = self.frombuffer_locals.clone();

                self.env = merge_branch_envs(
                    &env_before,
                    &body_env,
                    body_continues,
                    &else_env,
                    else_continues,
                )?;
                self.branch_only = branch_only_before;
                if body_continues && else_continues {
                    mark_branch_only_locals(
                        &mut self.branch_only,
                        &env_before,
                        &body_env,
                        &else_env,
                    );
                }
                self.return_type = merge_return_types(
                    return_type_before,
                    merge_return_types(body_return_type, else_return_type)?,
                )?;
                self.yield_type = merge_yield_types(
                    yield_type_before,
                    merge_yield_types(body_yield_type, else_yield_type)?,
                )?;
                self.helper_cache = body_helper_cache;
                self.helper_cache.extend(else_helper_cache);
                self.generator_cache = body_generator_cache;
                self.generator_cache.extend(else_generator_cache);
                self.frombuffer_locals = body_frombuffer_locals
                    .intersection(&else_frombuffer_locals)
                    .cloned()
                    .collect();

                Ok(TypedStmt::If {
                    test,
                    body: typed_body,
                    orelse: typed_orelse,
                })
            }
            StmtNode::While { test, body } => {
                let test = self.expr(test)?;
                if test.typ != RumbaType::Scalar(ScalarType::Bool) {
                    return Err(unsupported("while condition must be boolean"));
                }
                self.loop_depth += 1;
                let body = body
                    .iter()
                    .map(|stmt| self.stmt(stmt))
                    .collect::<PyResult<Vec<_>>>()?;
                self.loop_depth -= 1;
                Ok(TypedStmt::While { test, body })
            }
            StmtNode::ForRange {
                target,
                start,
                stop,
                step,
                body,
            } => {
                let start = self.expr(start)?;
                let stop = self.expr(stop)?;
                let step = self.expr(step)?;
                require_int64(&start, "range start")?;
                require_int64(&stop, "range stop")?;
                require_int64(&step, "range step")?;
                if matches!(step.kind, TypedExprKind::Constant(ConstantValue::Int(0))) {
                    return Err(unsupported("range step cannot be zero"));
                }
                self.env
                    .insert(target.to_string(), RumbaType::Scalar(ScalarType::Int64));
                self.loop_depth += 1;
                let body = body
                    .iter()
                    .map(|stmt| self.stmt(stmt))
                    .collect::<PyResult<Vec<_>>>()?;
                self.loop_depth -= 1;
                Ok(TypedStmt::ForRange {
                    target: target.clone(),
                    start,
                    stop,
                    step,
                    body,
                })
            }
            StmtNode::ForGenerator {
                target,
                function,
                args,
                body,
            } => {
                if !contains_yield(&function.body) {
                    return Err(unsupported(
                        "for loops over helpers require a generator helper with yield",
                    ));
                }
                let args = args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<PyResult<Vec<_>>>()?;
                let signature = args.iter().map(|arg| arg.typ.clone()).collect::<Vec<_>>();
                let key = parsed_helper_key(function, &signature);
                let function = if let Some(function) = self.generator_cache.get(&key) {
                    function.clone()
                } else {
                    let function = type_generator_function((**function).clone(), signature)?;
                    self.generator_cache.insert(key, function.clone());
                    function
                };
                self.env
                    .insert(target.to_string(), function.yield_type.clone());
                self.loop_depth += 1;
                let body = body
                    .iter()
                    .map(|stmt| self.stmt(stmt))
                    .collect::<PyResult<Vec<_>>>()?;
                self.loop_depth -= 1;
                Ok(TypedStmt::ForGenerator {
                    target: target.clone(),
                    function: Box::new(function),
                    args,
                    body,
                })
            }
        }
    }

    fn expr(&mut self, node: &ExprNode) -> PyResult<TypedExpr> {
        match node {
            ExprNode::Constant(value) => Ok(TypedExpr {
                kind: TypedExprKind::Constant(value.clone()),
                typ: match value {
                    ConstantValue::Bool(_) => RumbaType::Scalar(ScalarType::Bool),
                    ConstantValue::Int(_) => RumbaType::Scalar(ScalarType::Int64),
                    ConstantValue::Float(_) => RumbaType::Scalar(ScalarType::Float64),
                    ConstantValue::Str(_) => {
                        return Err(unsupported(
                            "string constants are only valid as struct field keys (a[i]['field'])",
                        ));
                    }
                },
            }),
            ExprNode::Dtype(dtype) => Ok(TypedExpr {
                kind: TypedExprKind::Dtype(dtype.clone()),
                typ: RumbaType::Dtype(dtype.clone()),
            }),
            ExprNode::Name(name) => {
                let typ = self
                    .env
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        if self.branch_only.contains(name) {
                            unsupported(format!(
                                "local variable {name:?} is assigned in only one branch and is used after the branch"
                            ))
                        } else {
                            unsupported(format!("unknown name {name:?}"))
                        }
                    })?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Name(name.clone()),
                    typ,
                })
            }
            ExprNode::Call { target, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<PyResult<Vec<_>>>()?;
                match target {
                    CallTarget::Helper {
                        function,
                        explicit_signature,
                    } => {
                        if contains_yield(&function.body) {
                            return Err(unsupported(
                                "generator helpers are only supported when consumed directly by a for loop",
                            ));
                        }
                        let signature = args.iter().map(|arg| arg.typ.clone()).collect::<Vec<_>>();
                        if let Some(explicit_signature) = explicit_signature {
                            if explicit_signature != &signature {
                                return Err(unsupported(format!(
                                    "helper call signature [{}] does not match explicit helper signature [{}]",
                                    format_signature(&signature),
                                    format_signature(explicit_signature)
                                )));
                            }
                        }
                        let key = parsed_helper_key(function, &signature);
                        let function = if let Some(function) = self.helper_cache.get(&key) {
                            function.clone()
                        } else {
                            let function = type_function((**function).clone(), signature)?;
                            self.helper_cache.insert(key, function.clone());
                            function
                        };
                        Ok(TypedExpr {
                            typ: RumbaType::Scalar(function.return_type),
                            kind: TypedExprKind::Call {
                                function: Box::new(function),
                                args,
                            },
                        })
                    }
                    CallTarget::Intrinsic(intrinsic) => {
                        let typ = intrinsic.type_call(&args)?;
                        Ok(TypedExpr {
                            typ,
                            kind: TypedExprKind::IntrinsicCall {
                                intrinsic: *intrinsic,
                                args,
                            },
                        })
                    }
                }
            }
            ExprNode::Index { target, index } => {
                let target = self.expr(target)?;
                let index = self.expr(index)?;
                let element_type = match &target.typ {
                    RumbaType::Array1D(typ) => *typ,
                    RumbaType::Array1DStruct(_) => {
                        return Err(unsupported(
                            "use a[i]['field'] syntax to read struct array fields",
                        ));
                    }
                    RumbaType::Scalar(_) | RumbaType::ByteBuffer | RumbaType::Dtype(_) => {
                        return Err(unsupported("indexing requires an array"));
                    }
                };
                if index.typ != RumbaType::Scalar(ScalarType::Int64) {
                    return Err(unsupported("array index must be an int64 scalar"));
                }
                Ok(TypedExpr {
                    kind: TypedExprKind::Index {
                        target: Box::new(target),
                        index: Box::new(index),
                    },
                    typ: RumbaType::Scalar(element_type),
                })
            }
            ExprNode::IndexField {
                target,
                index,
                field,
            } => {
                let target = self.expr(target)?;
                let index = self.expr(index)?;
                let dtype = match &target.typ {
                    RumbaType::Array1DStruct(dtype) => dtype.clone(),
                    _ => {
                        return Err(unsupported(
                            "field indexing requires a structured array; use a[i]['field']",
                        ));
                    }
                };
                if index.typ != RumbaType::Scalar(ScalarType::Int64) {
                    return Err(unsupported("struct array index must be an int64 scalar"));
                }
                let (field_type, _) = dtype.field(field).ok_or_else(|| {
                    unsupported(format!(
                        "field {field:?} does not exist in structured dtype {}",
                        dtype.c_struct_name
                    ))
                })?;
                let scalar_type = field_type.to_scalar_type();
                Ok(TypedExpr {
                    kind: TypedExprKind::IndexField {
                        target: Box::new(target),
                        index: Box::new(index),
                        field: field.clone(),
                        field_type,
                    },
                    typ: RumbaType::Scalar(scalar_type),
                })
            }
            ExprNode::BinOp { left, op, right } => {
                let left = self.expr(left)?;
                let right = self.expr(right)?;
                let left_type = left
                    .typ
                    .as_scalar()
                    .ok_or_else(|| unsupported("binary operators require scalar operands"))?;
                let right_type = right
                    .typ
                    .as_scalar()
                    .ok_or_else(|| unsupported("binary operators require scalar operands"))?;
                Ok(TypedExpr {
                    typ: RumbaType::Scalar(type_binary_op(left_type, *op, right_type)?),
                    kind: TypedExprKind::BinOp {
                        left: Box::new(left),
                        op: *op,
                        right: Box::new(right),
                    },
                })
            }
            ExprNode::UnaryOp { op, value } => {
                let value = self.expr(value)?;
                let value_type = value
                    .typ
                    .as_scalar()
                    .ok_or_else(|| unsupported("unary operators require scalar operands"))?;
                let typ = match op {
                    UnaryOp::Not => {
                        if value_type != ScalarType::Bool {
                            return Err(unsupported(format!(
                                "not requires bool operand, got {}; use an explicit bool(...) cast",
                                value_type.name()
                            )));
                        }
                        RumbaType::Scalar(ScalarType::Bool)
                    }
                    UnaryOp::USub | UnaryOp::UAdd => {
                        if !value_type.is_numeric() {
                            return Err(unsupported(format!(
                                "unary {} requires int64 or float64 operand, got {}",
                                op.name(),
                                value_type.name()
                            )));
                        }
                        value.typ.clone()
                    }
                };
                Ok(TypedExpr {
                    typ,
                    kind: TypedExprKind::UnaryOp {
                        op: op.clone(),
                        value: Box::new(value),
                    },
                })
            }
            ExprNode::Compare { left, op, right } => {
                let left = self.expr(left)?;
                let right = self.expr(right)?;
                type_compare(
                    left.typ
                        .as_scalar()
                        .ok_or_else(|| unsupported("comparisons require scalar operands"))?,
                    op,
                    right
                        .typ
                        .as_scalar()
                        .ok_or_else(|| unsupported("comparisons require scalar operands"))?,
                )?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Compare {
                        left: Box::new(left),
                        op: op.clone(),
                        right: Box::new(right),
                    },
                    typ: RumbaType::Scalar(ScalarType::Bool),
                })
            }
        }
    }

    fn print_arg(&mut self, node: &ExprNode) -> PyResult<TypedPrintArg> {
        if let ExprNode::Constant(ConstantValue::Str(value)) = node {
            return Ok(TypedPrintArg::StaticStr(value.clone()));
        }
        let expr = self.expr(node)?;
        if expr.typ.as_scalar().is_none() {
            return Err(unsupported(
                "print arguments must be scalar values or string literals",
            ));
        }
        Ok(TypedPrintArg::Expr(expr))
    }

    fn is_frombuffer_view(&self, expr: &TypedExpr) -> bool {
        if is_frombuffer_expr(expr) {
            return true;
        }
        matches!(&expr.kind, TypedExprKind::Name(name) if self.frombuffer_locals.contains(name))
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

fn merge_branch_envs(
    before: &HashMap<String, RumbaType>,
    body: &HashMap<String, RumbaType>,
    body_continues: bool,
    orelse: &HashMap<String, RumbaType>,
    else_continues: bool,
) -> PyResult<HashMap<String, RumbaType>> {
    reject_incompatible_common_branch_locals(before, body, orelse)?;
    match (body_continues, else_continues) {
        (true, true) => {}
        (true, false) => return Ok(body.clone()),
        (false, true) => return Ok(orelse.clone()),
        (false, false) => return Ok(before.clone()),
    }

    let mut names = before.keys().cloned().collect::<HashSet<_>>();
    names.extend(body.keys().cloned());
    names.extend(orelse.keys().cloned());

    let mut merged = HashMap::new();
    for name in names {
        let before_type = before.get(&name).cloned();
        let body_type = body.get(&name).cloned();
        let else_type = orelse.get(&name).cloned();
        match (before_type, body_type, else_type) {
            (Some(_), Some(ref body_type), Some(ref else_type)) if body_type != else_type => {
                return Err(incompatible_branch_type(
                    &name,
                    body_type.clone(),
                    else_type.clone(),
                ));
            }
            (Some(ref before_type), Some(ref body_type), None) if body_type != before_type => {
                return Err(incompatible_branch_type(
                    &name,
                    body_type.clone(),
                    before_type.clone(),
                ));
            }
            (Some(ref before_type), None, Some(ref else_type)) if else_type != before_type => {
                return Err(incompatible_branch_type(
                    &name,
                    before_type.clone(),
                    else_type.clone(),
                ));
            }
            (Some(_), Some(body_type), Some(_)) => {
                merged.insert(name, body_type);
            }
            (Some(before_type), Some(_), None)
            | (Some(before_type), None, Some(_))
            | (Some(before_type), None, None) => {
                merged.insert(name, before_type);
            }
            (None, Some(body_type), Some(ref else_type)) if &body_type == else_type => {
                merged.insert(name, body_type);
            }
            (None, Some(body_type), Some(else_type)) => {
                return Err(incompatible_branch_type(&name, body_type, else_type));
            }
            (None, Some(_), None) | (None, None, Some(_)) | (None, None, None) => {}
        }
    }
    Ok(merged)
}

fn reject_incompatible_common_branch_locals(
    before: &HashMap<String, RumbaType>,
    body: &HashMap<String, RumbaType>,
    orelse: &HashMap<String, RumbaType>,
) -> PyResult<()> {
    for (name, body_type) in body {
        if before.get(name).is_some() {
            continue;
        }
        if let Some(else_type) = orelse.get(name) {
            if body_type != else_type {
                return Err(incompatible_branch_type(
                    name,
                    body_type.clone(),
                    else_type.clone(),
                ));
            }
        }
    }
    Ok(())
}

fn mark_branch_only_locals(
    branch_only: &mut HashSet<String>,
    before: &HashMap<String, RumbaType>,
    body: &HashMap<String, RumbaType>,
    orelse: &HashMap<String, RumbaType>,
) {
    for name in body.keys().chain(orelse.keys()) {
        if !before.contains_key(name) && (body.contains_key(name) != orelse.contains_key(name)) {
            branch_only.insert(name.clone());
        }
    }
}

fn incompatible_branch_type(name: &str, left: RumbaType, right: RumbaType) -> pyo3::PyErr {
    unsupported(format!(
        "incompatible branch assignment types for local {name:?}: {} vs {}",
        left.name(),
        right.name()
    ))
}

fn merge_return_types(
    left: Option<ScalarType>,
    right: Option<ScalarType>,
) -> PyResult<Option<ScalarType>> {
    match (left, right) {
        (None, None) => Ok(None),
        (Some(typ), None) | (None, Some(typ)) => Ok(Some(typ)),
        (Some(left), Some(right)) if left == right => Ok(Some(left)),
        (Some(left), Some(right)) => Err(exact_type_mismatch("return values", left, right)),
    }
}

fn merge_yield_types(
    left: Option<RumbaType>,
    right: Option<RumbaType>,
) -> PyResult<Option<RumbaType>> {
    match (left, right) {
        (None, None) => Ok(None),
        (Some(typ), None) | (None, Some(typ)) => Ok(Some(typ)),
        (Some(left), Some(right)) if left == right => Ok(Some(left)),
        (Some(left), Some(right)) => Err(exact_rumba_type_mismatch(
            "generator yield values",
            &left,
            &right,
        )),
    }
}

fn type_binary_op(left: ScalarType, op: BinOp, right: ScalarType) -> PyResult<ScalarType> {
    if left != right {
        return Err(exact_type_mismatch(op.type_name(), left, right));
    }
    if !left.is_numeric() {
        return Err(unsupported(format!(
            "{} requires int64 or float64 operands, got {}; use an explicit cast",
            op.type_name(),
            left.name()
        )));
    }
    match op {
        BinOp::Div => Ok(ScalarType::Float64),
        BinOp::Add | BinOp::Sub | BinOp::Mult | BinOp::FloorDiv | BinOp::Mod => Ok(left),
    }
}

fn type_compare(left: ScalarType, op: &crate::ir::CmpOp, right: ScalarType) -> PyResult<()> {
    if left != right {
        return Err(exact_type_mismatch("comparison", left, right));
    }
    match op {
        crate::ir::CmpOp::Eq | crate::ir::CmpOp::NotEq => Ok(()),
        crate::ir::CmpOp::Lt
        | crate::ir::CmpOp::LtE
        | crate::ir::CmpOp::Gt
        | crate::ir::CmpOp::GtE => {
            if left.is_numeric() {
                Ok(())
            } else {
                Err(unsupported(
                    "ordering comparisons require int64 or float64 operands, got bool",
                ))
            }
        }
    }
}

fn exact_type_mismatch(context: &str, left: ScalarType, right: ScalarType) -> pyo3::PyErr {
    unsupported(format!(
        "{context} require exact matching scalar types, got {} and {}; use an explicit cast",
        left.name(),
        right.name()
    ))
}

fn exact_rumba_type_mismatch(context: &str, left: &RumbaType, right: &RumbaType) -> pyo3::PyErr {
    unsupported(format!(
        "{context} require exact matching types, got {} and {}; use an explicit cast",
        left.name(),
        right.name()
    ))
}

trait ScalarTypeExt {
    fn is_numeric(self) -> bool;
}

impl ScalarTypeExt for ScalarType {
    fn is_numeric(self) -> bool {
        matches!(self, ScalarType::Int64 | ScalarType::Float64)
    }
}

trait UnaryOpExt {
    fn name(&self) -> &'static str;
}

impl UnaryOpExt for UnaryOp {
    fn name(&self) -> &'static str {
        match self {
            UnaryOp::Not => "not",
            UnaryOp::USub => "-",
            UnaryOp::UAdd => "+",
        }
    }
}

fn require_int64(expr: &TypedExpr, label: &str) -> PyResult<()> {
    if expr.typ != RumbaType::Scalar(ScalarType::Int64) {
        return Err(unsupported(format!("{label} must be an int64 scalar")));
    }
    Ok(())
}

fn stmts_may_continue(stmts: &[StmtNode]) -> bool {
    for stmt in stmts {
        if !stmt_may_continue(stmt) {
            return false;
        }
    }
    true
}

fn stmt_may_continue(stmt: &StmtNode) -> bool {
    match stmt {
        StmtNode::Return(_) | StmtNode::Break | StmtNode::Continue => false,
        StmtNode::If { body, orelse, .. } => {
            orelse.is_empty() || stmts_may_continue(body) || stmts_may_continue(orelse)
        }
        StmtNode::While { .. } | StmtNode::ForGenerator { .. } => true,
        _ => true,
    }
}

fn contains_yield(stmts: &[StmtNode]) -> bool {
    stmts.iter().any(|stmt| match stmt {
        StmtNode::Yield(_) => true,
        StmtNode::If { body, orelse, .. } => contains_yield(body) || contains_yield(orelse),
        StmtNode::While { body, .. } | StmtNode::ForRange { body, .. } => contains_yield(body),
        StmtNode::ForGenerator { body, function, .. } => {
            contains_yield(body) || contains_yield(&function.body)
        }
        _ => false,
    })
}

pub(crate) fn helper_key(function: &TypedFunction) -> String {
    let mut key = function.name.clone();
    key.push('(');
    for typ in &function.signature {
        key.push_str(&typ.name());
        key.push(',');
    }
    key.push(')');
    key
}

fn parsed_helper_key(function: &ParsedFunction, signature: &[RumbaType]) -> String {
    let mut key = function.name.clone();
    key.push('(');
    for typ in signature {
        key.push_str(&typ.name());
        key.push(',');
    }
    key.push(')');
    key
}
