use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::artifact::{CompiledArtifact, PyCompiledArtifact};
use crate::compile::compile_parsed_function;
use crate::errors::unsupported;
use crate::frontend::bytecode::{build_rumba_ast, inspect_bytecode};
use crate::frontend::parse_function_input;
use crate::inspect::typed_function_to_py;
use crate::runtime::call_native;
use crate::types::{format_signature, parse_signature_tuple, signature_tuple, RumbaType};

#[pyclass]
pub(crate) struct Dispatcher {
    py_func: Py<PyAny>,
    cache: bool,
    debug: bool,
    explicit_signature: Option<Vec<RumbaType>>,
    compiled: Vec<CompiledArtifact>,
}

#[pymethods]
impl Dispatcher {
    #[getter]
    fn py_func(&self, py: Python<'_>) -> PyObject {
        self.py_func.clone_ref(py).into()
    }

    #[getter]
    fn cache(&self) -> bool {
        self.cache
    }

    #[getter]
    fn debug(&self) -> bool {
        self.debug
    }

    #[getter]
    fn signatures(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyList::empty_bound(py);
        for artifact in &self.compiled {
            out.append(signature_tuple(py, &artifact.signature)?)?;
        }
        Ok(out.into())
    }

    #[getter]
    fn _compiled(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new_bound(py);
        for artifact in &self.compiled {
            out.set_item(
                signature_tuple(py, &artifact.signature)?,
                Py::new(
                    py,
                    PyCompiledArtifact {
                        inner: artifact.clone(),
                    },
                )?,
            )?;
        }
        Ok(out.into())
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        if kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
            return Err(unsupported("keyword arguments are not supported"));
        }
        let signature = match &self.explicit_signature {
            Some(signature) => signature.clone(),
            None => args
                .iter()
                .map(|arg| RumbaType::from_arg(&arg))
                .collect::<PyResult<Vec<_>>>()?,
        };
        if self.debug {
            eprintln!("[rumba-debug] call: function: {}", self.function_name(py));
            eprintln!(
                "[rumba-debug] call: positional argument count: {}",
                args.len()
            );
            eprintln!(
                "[rumba-debug] call: selected signature: [{}]",
                format_signature(&signature)
            );
            eprintln!(
                "[rumba-debug] call: explicit signature: {}",
                self.explicit_signature.is_some()
            );
        }
        if signature.len() != args.len() {
            return Err(unsupported(
                "argument count does not match explicit signature",
            ));
        }
        let artifact_index = self.compile_if_needed(py, &signature)?;
        let artifact = &self.compiled[artifact_index];
        call_native(py, artifact, args, self.debug)
    }

    fn inspect_bytecode(&self, py: Python<'_>) -> PyResult<PyObject> {
        inspect_bytecode(py, self.py_func.bind(py))
    }

    fn inspect_rumba_ast(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ir = build_rumba_ast(py, self.py_func.bind(py))?;
        let out = PyDict::new_bound(py);
        out.set_item("name", ir.name)?;
        out.set_item("args", ir.args)?;
        let body = PyList::empty_bound(py);
        for stmt in ir.body.iter() {
            body.append(stmt.kind())?;
        }
        out.set_item("body", body)?;
        Ok(out.into())
    }

    #[pyo3(signature = (signature=None))]
    fn inspect_c(&self, signature: Option<&Bound<'_, PyTuple>>) -> PyResult<String> {
        Ok(self.select_artifact(signature)?.source.clone())
    }

    #[pyo3(signature = (signature=None))]
    fn inspect_compile_command(
        &self,
        signature: Option<&Bound<'_, PyTuple>>,
    ) -> PyResult<Vec<String>> {
        Ok(self.select_artifact(signature)?.compile_command.clone())
    }

    #[pyo3(signature = (signature=None))]
    fn inspect_cache_path(&self, signature: Option<&Bound<'_, PyTuple>>) -> PyResult<String> {
        Ok(self
            .select_artifact(signature)?
            .cache_path
            .display()
            .to_string())
    }

    #[pyo3(signature = (signature=None))]
    fn inspect_typed_ast(
        &self,
        py: Python<'_>,
        signature: Option<&Bound<'_, PyTuple>>,
    ) -> PyResult<PyObject> {
        typed_function_to_py(py, &self.select_artifact(signature)?.typed_function)
    }
}

impl Dispatcher {
    fn select_artifact(
        &self,
        signature: Option<&Bound<'_, PyTuple>>,
    ) -> PyResult<&CompiledArtifact> {
        if let Some(signature) = signature {
            let signature = parse_signature_tuple(signature)?;
            return self
                .compiled
                .iter()
                .find(|artifact| artifact.signature == signature)
                .ok_or_else(|| unsupported("signature has not been compiled"));
        }
        match self.compiled.len() {
            1 => Ok(&self.compiled[0]),
            0 => Err(unsupported(
                "inspection requires a signature before compilation",
            )),
            _ => Err(unsupported(
                "inspection requires a signature after multiple signatures have been compiled",
            )),
        }
    }
}

impl Dispatcher {
    pub(crate) fn original_py_func(&self, py: Python<'_>) -> Py<PyAny> {
        self.py_func.clone_ref(py)
    }

    pub(crate) fn explicit_signature(&self) -> Option<Vec<RumbaType>> {
        self.explicit_signature.clone()
    }

    pub(crate) fn new(
        py_func: Py<PyAny>,
        cache: bool,
        debug: bool,
        explicit_signature: Option<Vec<RumbaType>>,
    ) -> Self {
        Self {
            py_func,
            cache,
            debug,
            explicit_signature,
            compiled: Vec::new(),
        }
    }

    fn compile_if_needed(&mut self, py: Python<'_>, signature: &[RumbaType]) -> PyResult<usize> {
        if let Some(index) = self
            .compiled
            .iter()
            .position(|artifact| artifact.signature == signature)
        {
            if self.debug {
                eprintln!(
                    "[rumba-debug] dispatcher: cache hit for signature [{}] at artifact index {index}",
                    format_signature(signature)
                );
            }
            return Ok(index);
        }

        if self.debug {
            eprintln!(
                "[rumba-debug] dispatcher: cache miss for signature [{}]; compiling",
                format_signature(signature)
            );
        }
        let parsed = parse_function_input(py, self.py_func.bind(py))?;
        let artifact = compile_parsed_function(parsed, signature.to_vec(), self.debug)?;
        if self.debug {
            eprintln!(
                "[rumba-debug] dispatcher: compiled artifact index {}",
                self.compiled.len()
            );
            eprintln!("[rumba-debug] dispatcher: artifact key: {}", artifact.key);
            eprintln!(
                "[rumba-debug] dispatcher: artifact return type: {}",
                artifact.return_type.name()
            );
            eprintln!(
                "[rumba-debug] dispatcher: artifact library path: {}",
                artifact.library_path.display()
            );
        }
        self.compiled.push(artifact);
        Ok(self.compiled.len() - 1)
    }

    fn function_name(&self, py: Python<'_>) -> String {
        self.py_func
            .bind(py)
            .getattr("__qualname__")
            .and_then(|name| name.extract())
            .or_else(|_| {
                self.py_func
                    .bind(py)
                    .getattr("__name__")
                    .and_then(|name| name.extract())
            })
            .unwrap_or_else(|_| "<unknown>".to_string())
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Dispatcher>()?;
    Ok(())
}
