# Agent Notes

- Use `uv` to manage Python environments and dependencies for this repository.
- Prefer `uv run ...` for Python commands when possible so tools run in the managed environment.
- Do not manually edit virtualenv contents; update project dependency metadata and let `uv` sync the environment.
