# Agent Notes

- Use `uv` to manage Python environments and dependencies for this repository.
- Prefer `uv run ...` for Python commands when possible so tools run in the managed environment.
- Do not manually edit virtualenv contents; update project dependency metadata and let `uv` sync the environment.
- Use `RUMBA_MIGRATION_PLAN.md` as the source of truth for Rumba migration status, milestones, and scope.
- Run the Rumba verification flow with:

```bash
uv run maturin develop -m rumba/Cargo.toml && uv run pytest -- rumba/tests
```
