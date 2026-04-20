# Serve Release Gate

This runbook is for the internal model-upload and cloud packaging surface. It is not part of the public repo-clone OSS readiness contract.

Use this runbook before calling the `serve` / model-upload surface release-ready.

## Local Gates

1. Run the serving unit and integration coverage:
   - `uv run pytest tests/unit/numereng/features/serving tests/unit/numereng/test_api_serving.py tests/unit/numereng/test_cli_serving.py tests/integration/test_serving_smoke.py -q`
2. Run the internal wheel/install smoke:
   - `uv run pytest tests/e2e/test_serving_wheel_install.py -q`
3. Confirm the CLI help surface is present from the installed internal wheel:
   - `numereng serve --help`
4. Confirm a package inspection report is written under:
   - `experiments/<experiment_id>/submission_packages/<package_id>/artifacts/preflight/report.json`
5. Confirm the inspection classification is release-grade before upload:
   - `preflight_deployment_classification=pickle_upload_ready`
   - `preflight_artifact_ready=true`

## Real Numerai Smoke

This is the release blocker for public model-upload support.

1. Export Numerai credentials into the current shell.
2. Choose a spare Classic model slot that is not handling production submissions.
3. Use a run-backed package whose components come from persisted `full_history_refit` model artifacts.
4. Build a known-good pickle package:
   - `uv run numereng serve pickle build --experiment-id <id> --package-id <id> --docker-image "Python 3.12"`
   - confirm the build recorded `pickle_smoke_verified=true` for that same docker image
5. Upload it:
   - `uv run numereng serve pickle upload --experiment-id <id> --package-id <id> --model-name <spare_model_name> --docker-image "Python 3.12"`
6. Confirm:
   - upload succeeds locally
   - Numerai accepts the uploaded pickle
   - the hosted run reaches a successful state for the current round

Do not mark the surface release-ready until one real authenticated upload passes end to end.
