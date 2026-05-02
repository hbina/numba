use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::ir::{BinOp, CmpOp, ConstantValue, UnaryOp};
use crate::types::{FieldType, RumbaType, ScalarType};
use crate::typing::{TypedExpr, TypedExprKind, TypedFunction, TypedStmt};

pub(crate) fn typed_function_to_py(py: Python<'_>, function: &TypedFunction) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("name", &function.name)?;
    out.set_item("signature", type_names(&function.signature))?;
    out.set_item("return_type", function.return_type.name())?;
    out.set_item("args", args_to_py(py, function)?)?;
    out.set_item("body", stmts_to_py(py, &function.body)?)?;
    out.set_item("locals", locals_to_py(py, function)?)?;
    Ok(out.into())
}

fn args_to_py(py: Python<'_>, function: &TypedFunction) -> PyResult<PyObject> {
    let out = PyList::empty_bound(py);
    for (name, typ) in function.args.iter().zip(function.signature.iter()) {
        let item = PyDict::new_bound(py);
        item.set_item("name", name)?;
        item.set_item("type", typ.name())?;
        out.append(item)?;
    }
    Ok(out.into())
}

fn locals_to_py(py: Python<'_>, function: &TypedFunction) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    let mut names = function.locals.keys().collect::<Vec<_>>();
    names.sort();
    for name in names {
        out.set_item(name, function.locals[name].name())?;
    }
    Ok(out.into())
}

fn stmts_to_py(py: Python<'_>, stmts: &[TypedStmt]) -> PyResult<PyObject> {
    let out = PyList::empty_bound(py);
    for stmt in stmts {
        out.append(stmt_to_py(py, stmt)?)?;
    }
    Ok(out.into())
}

fn stmt_to_py(py: Python<'_>, stmt: &TypedStmt) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    match stmt {
        TypedStmt::Return(value) => {
            out.set_item("kind", "Return")?;
            out.set_item("value_type", value.typ.name())?;
            out.set_item("value", expr_to_py(py, value)?)?;
        }
        TypedStmt::Break => {
            out.set_item("kind", "Break")?;
        }
        TypedStmt::Continue => {
            out.set_item("kind", "Continue")?;
        }
        TypedStmt::Assign { name, value } => {
            out.set_item("kind", "Assign")?;
            out.set_item("target", name)?;
            out.set_item("target_type", value.typ.name())?;
            out.set_item("value", expr_to_py(py, value)?)?;
        }
        TypedStmt::AugAssign {
            name,
            op,
            value,
            target_type,
        } => {
            out.set_item("kind", "AugAssign")?;
            out.set_item("target", name)?;
            out.set_item("target_type", target_type.name())?;
            out.set_item("op", bin_op_name(*op))?;
            out.set_item("value", expr_to_py(py, value)?)?;
        }
        TypedStmt::StoreIndex {
            target,
            index,
            value,
            element_type,
        } => {
            out.set_item("kind", "StoreIndex")?;
            out.set_item("target_type", target.typ.name())?;
            out.set_item("element_type", element_type.name())?;
            out.set_item("value_type", value.typ.name())?;
            out.set_item("target", expr_to_py(py, target)?)?;
            out.set_item("index", expr_to_py(py, index)?)?;
            out.set_item("value", expr_to_py(py, value)?)?;
        }
        TypedStmt::StoreIndexField {
            target,
            index,
            field,
            value,
            field_type,
            dtype,
        } => {
            out.set_item("kind", "StoreIndexField")?;
            out.set_item("target_type", target.typ.name())?;
            out.set_item("dtype", &dtype.c_struct_name)?;
            out.set_item("field", field)?;
            out.set_item("field_type", field_type_name(*field_type))?;
            out.set_item("value_type", value.typ.name())?;
            out.set_item("target", expr_to_py(py, target)?)?;
            out.set_item("index", expr_to_py(py, index)?)?;
            out.set_item("value", expr_to_py(py, value)?)?;
        }
        TypedStmt::If { test, body, orelse } => {
            out.set_item("kind", "If")?;
            out.set_item("test_type", test.typ.name())?;
            out.set_item("test", expr_to_py(py, test)?)?;
            out.set_item("body", stmts_to_py(py, body)?)?;
            out.set_item("orelse", stmts_to_py(py, orelse)?)?;
        }
        TypedStmt::While { test, body } => {
            out.set_item("kind", "While")?;
            out.set_item("test_type", test.typ.name())?;
            out.set_item("test", expr_to_py(py, test)?)?;
            out.set_item("body", stmts_to_py(py, body)?)?;
        }
        TypedStmt::ForRange {
            target,
            start,
            stop,
            step,
            body,
        } => {
            out.set_item("kind", "ForRange")?;
            out.set_item("target", target)?;
            out.set_item("target_type", ScalarType::Int64.name())?;
            out.set_item("reason", "range_index")?;
            out.set_item("start", expr_to_py(py, start)?)?;
            out.set_item("stop", expr_to_py(py, stop)?)?;
            out.set_item("step", expr_to_py(py, step)?)?;
            out.set_item("body", stmts_to_py(py, body)?)?;
        }
    }
    Ok(out.into())
}

fn expr_to_py(py: Python<'_>, expr: &TypedExpr) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("type", expr.typ.name())?;
    match &expr.kind {
        TypedExprKind::Constant(value) => {
            out.set_item("kind", "Constant")?;
            out.set_item("reason", "literal")?;
            match value {
                ConstantValue::Int(value) => out.set_item("value", value)?,
                ConstantValue::Float(value) => out.set_item("value", value)?,
                ConstantValue::Bool(value) => out.set_item("value", value)?,
                ConstantValue::Str(value) => out.set_item("value", value)?,
            }
        }
        TypedExprKind::Name(name) => {
            out.set_item("kind", "Name")?;
            out.set_item("reason", "environment")?;
            out.set_item("name", name)?;
        }
        TypedExprKind::Call { function, args } => {
            out.set_item("kind", "Call")?;
            out.set_item("reason", "helper_return")?;
            out.set_item("helper", typed_function_to_py(py, function)?)?;
            out.set_item("args", exprs_to_py(py, args)?)?;
        }
        TypedExprKind::IntrinsicCall { intrinsic, args } => {
            out.set_item("kind", "IntrinsicCall")?;
            out.set_item("reason", "intrinsic")?;
            out.set_item("intrinsic", intrinsic.name())?;
            out.set_item("args", exprs_to_py(py, args)?)?;
        }
        TypedExprKind::Index { target, index } => {
            out.set_item("kind", "Index")?;
            out.set_item("reason", "array_element")?;
            out.set_item("target", expr_to_py(py, target)?)?;
            out.set_item("index", expr_to_py(py, index)?)?;
        }
        TypedExprKind::IndexField {
            target,
            index,
            field,
            field_type,
        } => {
            out.set_item("kind", "IndexField")?;
            out.set_item("reason", "struct_array_field")?;
            out.set_item("field", field)?;
            out.set_item("field_type", field_type_name(*field_type))?;
            out.set_item("target", expr_to_py(py, target)?)?;
            out.set_item("index", expr_to_py(py, index)?)?;
        }
        TypedExprKind::BinOp { left, op, right } => {
            out.set_item("kind", "BinOp")?;
            out.set_item("reason", "promote_numeric")?;
            out.set_item("op", bin_op_name(*op))?;
            out.set_item("left", expr_to_py(py, left)?)?;
            out.set_item("right", expr_to_py(py, right)?)?;
        }
        TypedExprKind::UnaryOp { op, value } => {
            out.set_item("kind", "UnaryOp")?;
            out.set_item("op", unary_op_name(op))?;
            out.set_item("value", expr_to_py(py, value)?)?;
        }
        TypedExprKind::Compare { left, op, right } => {
            out.set_item("kind", "Compare")?;
            out.set_item("reason", "comparison")?;
            out.set_item("op", cmp_op_name(op))?;
            out.set_item("left", expr_to_py(py, left)?)?;
            out.set_item("right", expr_to_py(py, right)?)?;
        }
    }
    Ok(out.into())
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

fn exprs_to_py(py: Python<'_>, exprs: &[TypedExpr]) -> PyResult<PyObject> {
    let out = PyList::empty_bound(py);
    for expr in exprs {
        out.append(expr_to_py(py, expr)?)?;
    }
    Ok(out.into())
}

fn type_names(types: &[RumbaType]) -> Vec<String> {
    types.iter().map(|typ| typ.name()).collect()
}

fn bin_op_name(op: BinOp) -> &'static str {
    op.type_name()
}

fn cmp_op_name(op: &CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "Eq",
        CmpOp::NotEq => "NotEq",
        CmpOp::Lt => "Lt",
        CmpOp::LtE => "LtE",
        CmpOp::Gt => "Gt",
        CmpOp::GtE => "GtE",
    }
}

fn unary_op_name(op: &UnaryOp) -> &'static str {
    match op {
        UnaryOp::Not => "Not",
        UnaryOp::USub => "USub",
        UnaryOp::UAdd => "UAdd",
    }
}
