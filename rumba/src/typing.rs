use std::collections::HashMap;

use pyo3::prelude::*;

use crate::errors::unsupported;
use crate::ir::{BinOp, ConstantValue, ExprNode, ParsedFunction, StmtNode, UnaryOp};
use crate::types::{promote_numeric, RumbaType, ScalarType};

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
pub(crate) enum TypedStmt {
    Return(TypedExpr),
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
    If {
        test: TypedExpr,
        body: Vec<TypedStmt>,
        orelse: Vec<TypedStmt>,
    },
    ForRange {
        target: String,
        start: TypedExpr,
        stop: TypedExpr,
        step: TypedExpr,
        body: Vec<TypedStmt>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct TypedExpr {
    pub(crate) kind: TypedExprKind,
    pub(crate) typ: RumbaType,
}

#[derive(Clone, Debug)]
pub(crate) enum TypedExprKind {
    Constant(ConstantValue),
    Name(String),
    Call {
        function: Box<TypedFunction>,
        args: Vec<TypedExpr>,
    },
    Len(Box<TypedExpr>),
    Index {
        target: Box<TypedExpr>,
        index: Box<TypedExpr>,
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
        .zip(signature.iter().copied())
        .collect::<HashMap<_, _>>();
    let mut pass = TypePass {
        env,
        return_type: None,
        helper_cache: HashMap::new(),
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

fn local_types(args: &[String], env: &HashMap<String, RumbaType>) -> HashMap<String, RumbaType> {
    env.iter()
        .filter(|(name, _)| !args.contains(name))
        .map(|(name, typ)| (name.clone(), *typ))
        .collect()
}

struct TypePass {
    env: HashMap<String, RumbaType>,
    return_type: Option<ScalarType>,
    helper_cache: HashMap<String, TypedFunction>,
}

impl TypePass {
    fn stmt(&mut self, node: &StmtNode) -> PyResult<TypedStmt> {
        match node {
            StmtNode::Return(value) => {
                let expr = self.expr(value)?;
                let expr_typ = expr
                    .typ
                    .as_scalar()
                    .ok_or_else(|| unsupported("array return values are not supported"))?;
                self.return_type = Some(match self.return_type {
                    None => expr_typ,
                    Some(current) if current == expr_typ => current,
                    Some(current) => promote_numeric(current, expr_typ, "Add"),
                });
                Ok(TypedStmt::Return(expr))
            }
            StmtNode::Assign { name, value } => {
                let expr = self.expr(value)?;
                self.env.insert(name.clone(), expr.typ);
                Ok(TypedStmt::Assign {
                    name: name.clone(),
                    value: expr,
                })
            }
            StmtNode::AugAssign { name, op, value } => {
                let target_type = self.env.get(name).copied().ok_or_else(|| {
                    unsupported(format!("local variable {name:?} is used before assignment"))
                })?;
                if target_type.as_scalar().is_none() {
                    return Err(unsupported("augmented assignment requires a scalar target"));
                }
                let expr = self.expr(value)?;
                if expr.typ.as_scalar().is_none() {
                    return Err(unsupported("augmented assignment requires a scalar value"));
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
                let element_type = match target.typ {
                    RumbaType::Array1D(typ) => typ,
                    RumbaType::Scalar(_) => {
                        return Err(unsupported("indexed assignment requires an array target"));
                    }
                };
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
            StmtNode::If { test, body, orelse } => {
                let test = self.expr(test)?;
                if test.typ != RumbaType::Scalar(ScalarType::Bool) {
                    return Err(unsupported("if condition must be boolean"));
                }

                let env_before = self.env.clone();
                let typed_body = body
                    .iter()
                    .map(|stmt| self.stmt(stmt))
                    .collect::<PyResult<Vec<_>>>()?;
                let body_env = self.env.clone();
                let typed_orelse = if orelse.is_empty() {
                    Vec::new()
                } else {
                    self.env = env_before;
                    orelse
                        .iter()
                        .map(|stmt| self.stmt(stmt))
                        .collect::<PyResult<Vec<_>>>()?
                };
                let else_env = self.env.clone();
                self.env = body_env;
                self.env.extend(else_env);

                Ok(TypedStmt::If {
                    test,
                    body: typed_body,
                    orelse: typed_orelse,
                })
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
                self.env
                    .insert(target.to_string(), RumbaType::Scalar(ScalarType::Int64));
                let body = body
                    .iter()
                    .map(|stmt| self.stmt(stmt))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(TypedStmt::ForRange {
                    target: target.clone(),
                    start,
                    stop,
                    step,
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
                },
            }),
            ExprNode::Name(name) => {
                let typ = self
                    .env
                    .get(name)
                    .copied()
                    .ok_or_else(|| unsupported(format!("unknown name {name:?}")))?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Name(name.clone()),
                    typ,
                })
            }
            ExprNode::Call { function, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<PyResult<Vec<_>>>()?;
                let signature = args.iter().map(|arg| arg.typ).collect::<Vec<_>>();
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
            ExprNode::Len(value) => {
                let value = self.expr(value)?;
                match value.typ {
                    RumbaType::Array1D(_) => Ok(TypedExpr {
                        kind: TypedExprKind::Len(Box::new(value)),
                        typ: RumbaType::Scalar(ScalarType::Int64),
                    }),
                    RumbaType::Scalar(_) => Err(unsupported("len expects a 1D numpy array")),
                }
            }
            ExprNode::Index { target, index } => {
                let target = self.expr(target)?;
                let index = self.expr(index)?;
                let element_type = match target.typ {
                    RumbaType::Array1D(typ) => typ,
                    RumbaType::Scalar(_) => return Err(unsupported("indexing requires an array")),
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
                    typ: RumbaType::Scalar(promote_numeric(left_type, right_type, op.type_name())),
                    kind: TypedExprKind::BinOp {
                        left: Box::new(left),
                        op: *op,
                        right: Box::new(right),
                    },
                })
            }
            ExprNode::UnaryOp { op, value } => {
                let value = self.expr(value)?;
                let typ = match op {
                    UnaryOp::Not => RumbaType::Scalar(ScalarType::Bool),
                    UnaryOp::USub | UnaryOp::UAdd => value.typ,
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
                if left.typ.as_scalar().is_none() || right.typ.as_scalar().is_none() {
                    return Err(unsupported("comparisons require scalar operands"));
                }
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
