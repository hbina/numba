use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use libloading::Library;
use pyo3::prelude::*;

use crate::artifact::CompiledArtifact;
use crate::cache::cache_key;
use crate::codegen::c::Emitter;
use crate::errors::{compilation, unsupported};
use crate::frontend::ParsedInput;
use crate::ir::StmtNode;
use crate::types::RumbaType;
use crate::typing::type_function;

pub(crate) fn compile_parsed_function(
    parsed: ParsedInput,
    signature: Vec<RumbaType>,
    debug: bool,
) -> PyResult<CompiledArtifact> {
    if debug {
        eprintln!("[rumba-debug] compile: parsed input: {parsed:#?}");
        eprintln!(
            "[rumba-debug] compile: requested signature: [{}]",
            crate::types::format_signature(&signature)
        );
    }
    let requires_writable_arrays = has_store_index(&parsed.function.body);
    let typed = type_function(parsed.function, signature.clone())?;
    if debug {
        eprintln!("[rumba-debug] compile: requires_writable_arrays: {requires_writable_arrays}");
        eprintln!("[rumba-debug] compile: typed function: {typed:#?}");
        eprintln!(
            "[rumba-debug] compile: inferred return type: {}",
            typed.return_type.name()
        );
    }
    let typed_function = typed.clone();
    let return_type = typed.return_type;
    let mut emitter = Emitter::new(typed);
    let source = emitter.emit()?;
    if debug {
        eprintln!("[rumba-debug] compile: generated C source follows");
        eprintln!("----- rumba generated C begin -----");
        eprintln!("{source}");
        eprintln!("----- rumba generated C end -----");
    }
    let key = cache_key(&parsed.metadata, &signature, &source);
    let cache_dir = env::var("RUMBA_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::temp_dir().join("rumba-cache"));
    let build_dir = cache_dir.join(&key);
    fs::create_dir_all(&build_dir)
        .map_err(|err| compilation(format!("failed to create cache directory: {err}")))?;
    let source_path = build_dir.join("module.c");
    let library_path = build_dir.join(format!("module{}", shared_suffix()?));
    if debug {
        eprintln!("[rumba-debug] compile: cache key: {key}");
        eprintln!(
            "[rumba-debug] compile: cache directory: {}",
            build_dir.display()
        );
        eprintln!(
            "[rumba-debug] compile: source path: {}",
            source_path.display()
        );
        eprintln!(
            "[rumba-debug] compile: library path: {}",
            library_path.display()
        );
    }
    fs::write(&source_path, &source)
        .map_err(|err| compilation(format!("failed to write generated C source: {err}")))?;

    let cc = env::var("CC")
        .ok()
        .or_else(|| find_executable("cc"))
        .or_else(|| find_executable("clang"))
        .or_else(|| find_executable("gcc"))
        .ok_or_else(|| compilation("no C compiler found; set CC to a working compiler"))?;

    let command = vec![
        cc,
        "-shared".to_string(),
        "-fPIC".to_string(),
        "-O2".to_string(),
        source_path.display().to_string(),
        "-o".to_string(),
        library_path.display().to_string(),
        "-lm".to_string(),
    ];
    if debug {
        eprintln!("[rumba-debug] compile: command: {}", command.join(" "));
    }

    if !library_path.exists() {
        if debug {
            eprintln!("[rumba-debug] compile: shared library missing; invoking C compiler");
        }
        let output = Command::new(&command[0])
            .args(&command[1..])
            .output()
            .map_err(|err| compilation(format!("failed to run C compiler: {err}")))?;
        if debug {
            eprintln!("[rumba-debug] compile: compiler status: {}", output.status);
            eprintln!(
                "[rumba-debug] compile: compiler stdout:\n{}",
                String::from_utf8_lossy(&output.stdout)
            );
            eprintln!(
                "[rumba-debug] compile: compiler stderr:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        if !output.status.success() {
            return Err(compilation(format!(
                "C compilation failed:\ncommand: {}\nstdout: {}\nstderr: {}",
                command.join(" "),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )));
        }
    } else if debug {
        eprintln!("[rumba-debug] compile: reusing existing shared library");
    }

    let library = unsafe { Library::new(&library_path) }
        .map_err(|err| compilation(format!("failed to load shared library: {err}")))?;
    if debug {
        eprintln!(
            "[rumba-debug] compile: loaded shared library: {}",
            library_path.display()
        );
    }

    Ok(CompiledArtifact {
        key,
        signature,
        return_type,
        typed_function,
        requires_writable_arrays,
        source,
        cache_path: build_dir,
        library_path,
        compile_command: command,
        library: Arc::new(library),
    })
}

fn has_store_index(body: &[StmtNode]) -> bool {
    body.iter().any(|stmt| match stmt {
        StmtNode::StoreIndex { .. } | StmtNode::StoreIndexField { .. } => true,
        StmtNode::If { body, orelse, .. } => has_store_index(body) || has_store_index(orelse),
        StmtNode::While { body, .. } => has_store_index(body),
        StmtNode::ForRange { body, .. } => has_store_index(body),
        StmtNode::ForGenerator { function, body, .. } => {
            has_store_index(&function.body) || has_store_index(body)
        }
        StmtNode::Return(_)
        | StmtNode::Yield(_)
        | StmtNode::Break
        | StmtNode::Continue
        | StmtNode::Assign { .. }
        | StmtNode::AugAssign { .. } => false,
    })
}

fn shared_suffix() -> PyResult<&'static str> {
    if cfg!(target_os = "macos") {
        Ok(".dylib")
    } else if cfg!(target_os = "linux") {
        Ok(".so")
    } else {
        Err(unsupported(
            "native compilation currently supports Linux and macOS",
        ))
    }
}

fn find_executable(name: &str) -> Option<String> {
    let paths = env::var_os("PATH")?;
    env::split_paths(&paths)
        .map(|path| path.join(name))
        .find(|candidate| candidate.is_file())
        .map(|candidate| candidate.display().to_string())
}
