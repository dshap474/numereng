All platform and submission modules must be wired up to src/numereng/api/.

Update src/numereng/cli/ when you are finished updating API code after adding new source code.

API and CLI must use folder-based domain modules (not monolithic single files):
- API: src/numereng/api/
- CLI: src/numereng/cli/

No source file in these folders may exceed 500 lines. Split by command/domain before crossing the limit.
