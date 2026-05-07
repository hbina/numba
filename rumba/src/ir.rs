use crate::intrinsics::IntrinsicId;
use crate::types::{DtypeSpec, RumbaType};

#[derive(Clone, Debug)]
pub(crate) struct ParsedFunction {
    pub(crate) name: String,
    pub(crate) args: Vec<String>,
    pub(crate) body: Vec<StmtNode>,
}

#[derive(Clone, Debug)]
pub(crate) enum StmtNode {
    Return(ExprNode),
    Yield(ExprNode),
    Break,
    Continue,
    Assign {
        name: String,
        value: ExprNode,
    },
    AugAssign {
        name: String,
        op: BinOp,
        value: ExprNode,
    },
    StoreIndex {
        target: ExprNode,
        index: ExprNode,
        value: ExprNode,
    },
    StoreIndexField {
        target: ExprNode,
        index: ExprNode,
        field: String,
        value: ExprNode,
    },
    If {
        test: ExprNode,
        body: Vec<StmtNode>,
        orelse: Vec<StmtNode>,
    },
    While {
        test: ExprNode,
        body: Vec<StmtNode>,
    },
    ForRange {
        target: String,
        start: ExprNode,
        stop: ExprNode,
        step: ExprNode,
        body: Vec<StmtNode>,
    },
    ForGenerator {
        target: String,
        function: Box<ParsedFunction>,
        args: Vec<ExprNode>,
        body: Vec<StmtNode>,
    },
}

impl StmtNode {
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Return(_) => "Return",
            Self::Yield(_) => "Yield",
            Self::Break => "Break",
            Self::Continue => "Continue",
            Self::Assign { .. } => "Assign",
            Self::AugAssign { .. } => "AugAssign",
            Self::StoreIndex { .. } => "StoreIndex",
            Self::StoreIndexField { .. } => "StoreIndexField",
            Self::If { .. } => "If",
            Self::While { .. } => "While",
            Self::ForRange { .. } => "For",
            Self::ForGenerator { .. } => "ForGenerator",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ExprNode {
    Constant(ConstantValue),
    Dtype(DtypeSpec),
    Name(String),
    Call {
        target: CallTarget,
        args: Vec<ExprNode>,
    },
    Index {
        target: Box<ExprNode>,
        index: Box<ExprNode>,
    },
    IndexField {
        target: Box<ExprNode>,
        index: Box<ExprNode>,
        field: String,
    },
    BinOp {
        left: Box<ExprNode>,
        op: BinOp,
        right: Box<ExprNode>,
    },
    UnaryOp {
        op: UnaryOp,
        value: Box<ExprNode>,
    },
    Compare {
        left: Box<ExprNode>,
        op: CmpOp,
        right: Box<ExprNode>,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum CallTarget {
    Helper {
        function: Box<ParsedFunction>,
        explicit_signature: Option<Vec<RumbaType>>,
    },
    Intrinsic(IntrinsicId),
}

#[derive(Clone, Debug)]
pub(crate) enum ConstantValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mult,
    Div,
    FloorDiv,
    Mod,
}

impl BinOp {
    pub(crate) fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mult => "*",
            Self::Div => "/",
            Self::FloorDiv => "/",
            Self::Mod => "%",
        }
    }

    pub(crate) fn type_name(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mult => "Mult",
            Self::Div => "Div",
            Self::FloorDiv => "FloorDiv",
            Self::Mod => "Mod",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum UnaryOp {
    Not,
    USub,
    UAdd,
}

#[derive(Clone, Debug)]
pub(crate) enum CmpOp {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
}

impl CmpOp {
    pub(crate) fn symbol(&self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::NotEq => "!=",
            Self::Lt => "<",
            Self::LtE => "<=",
            Self::Gt => ">",
            Self::GtE => ">=",
        }
    }
}
