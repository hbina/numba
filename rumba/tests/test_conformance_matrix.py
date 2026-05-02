import importlib.util
import sys
from pathlib import Path


def _load_conformance_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "compare_python_numba_rumba_2.py"
    spec = importlib.util.spec_from_file_location("rumba_conformance_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_python_numba_rumba_conformance_matrix():
    conformance = _load_conformance_example()

    results = conformance.run_conformance_matrix(debug=False, emit_details=False)
    failures = conformance.failure_results(results)

    assert not failures, conformance.format_failures(failures)
