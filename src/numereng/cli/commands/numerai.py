"""Numerai command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from numereng import api
from numereng.cli.common import _parse_int_value, _parse_tournament_value
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def _parse_datasets_list_request(argv: Sequence[str]) -> tuple[api.NumeraiDatasetListRequest | None, str | None]:
    tournament: api.NumeraiTournament = "classic"
    round_num: int | None = None
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return None, "__help__"
        if arg in {"--round", "--tournament"}:
            if idx + 1 >= len(argv):
                return None, f"missing value for {arg}"
            if arg == "--round":
                parsed_round, parse_error = _parse_int_value(argv[idx + 1], flag="--round")
                if parse_error is not None:
                    return None, parse_error
                round_num = parsed_round
            else:
                parsed_tournament, parse_error = _parse_tournament_value(argv[idx + 1])
                if parse_error is not None:
                    return None, parse_error
                if parsed_tournament is None:  # pragma: no cover - parse_error branch guards this
                    return None, "invalid value for --tournament"
                tournament = parsed_tournament
            idx += 2
            continue
        return None, f"unknown arguments: {arg}"
    return api.NumeraiDatasetListRequest(tournament=tournament, round_num=round_num), None


def _parse_datasets_download_request(
    argv: Sequence[str],
) -> tuple[api.NumeraiDatasetDownloadRequest | None, str | None]:
    filename: str | None = None
    dest_path: str | None = None
    tournament: api.NumeraiTournament = "classic"
    round_num: int | None = None

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return None, "__help__"
        if arg in {"--filename", "--dest-path", "--round", "--tournament"}:
            if idx + 1 >= len(argv):
                return None, f"missing value for {arg}"
            value = argv[idx + 1]
            if arg == "--filename":
                filename = value
            elif arg == "--dest-path":
                dest_path = value
            elif arg == "--round":
                parsed_round, parse_error = _parse_int_value(value, flag="--round")
                if parse_error is not None:
                    return None, parse_error
                round_num = parsed_round
            else:
                parsed_tournament, parse_error = _parse_tournament_value(value)
                if parse_error is not None:
                    return None, parse_error
                if parsed_tournament is None:  # pragma: no cover - parse_error branch guards this
                    return None, "invalid value for --tournament"
                tournament = parsed_tournament
            idx += 2
            continue
        return None, f"unknown arguments: {arg}"

    if filename is None:
        return None, "missing required argument: --filename"

    return (
        api.NumeraiDatasetDownloadRequest(
            filename=filename,
            tournament=tournament,
            dest_path=dest_path,
            round_num=round_num,
        ),
        None,
    )


def _parse_models_request(argv: Sequence[str]) -> tuple[api.NumeraiModelsRequest | None, str | None]:
    tournament: api.NumeraiTournament = "classic"
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return None, "__help__"
        if arg == "list":
            idx += 1
            continue
        if arg == "--tournament":
            if idx + 1 >= len(argv):
                return None, "missing value for --tournament"
            parsed_tournament, parse_error = _parse_tournament_value(argv[idx + 1])
            if parse_error is not None:
                return None, parse_error
            if parsed_tournament is None:  # pragma: no cover - parse_error branch guards this
                return None, "invalid value for --tournament"
            tournament = parsed_tournament
            idx += 2
            continue
        return None, f"unknown arguments: {arg}"
    return api.NumeraiModelsRequest(tournament=tournament), None


def _parse_current_round_request(
    argv: Sequence[str],
) -> tuple[api.NumeraiCurrentRoundRequest | None, str | None]:
    tournament: api.NumeraiTournament = "classic"
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return None, "__help__"
        if arg == "--tournament":
            if idx + 1 >= len(argv):
                return None, "missing value for --tournament"
            parsed_tournament, parse_error = _parse_tournament_value(argv[idx + 1])
            if parse_error is not None:
                return None, parse_error
            if parsed_tournament is None:  # pragma: no cover - parse_error branch guards this
                return None, "invalid value for --tournament"
            tournament = parsed_tournament
            idx += 2
            continue
        return None, f"unknown arguments: {arg}"
    return api.NumeraiCurrentRoundRequest(tournament=tournament), None


def _parse_forum_scrape_options(argv: Sequence[str]) -> tuple[dict[str, object] | None, str | None]:
    output_dir = "docs/numerai/forum"
    state_path: str | None = None
    full_refresh = False

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            return None, "__help__"
        if arg == "--full":
            full_refresh = True
            idx += 1
            continue
        if arg in {"--output-dir", "--state-path"}:
            if idx + 1 >= len(argv):
                return None, f"missing value for {arg}"
            value = argv[idx + 1]
            if arg == "--output-dir":
                output_dir = value
            else:
                state_path = value
            idx += 2
            continue
        return None, f"unknown arguments: {arg}"

    return {
        "output_dir": output_dir,
        "state_path": state_path,
        "full_refresh": full_refresh,
    }, None


def _handle_numerai_datasets_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] == "list":
        list_request, parse_error = _parse_datasets_list_request(args[1:])
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if list_request is None:  # pragma: no cover - parse_error branch guards this
            print("datasets_list_request_invalid", file=sys.stderr)
            return 2

        try:
            list_payload = api.list_numerai_datasets(list_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(list_payload.model_dump_json())
        return 0

    if args[0] == "download":
        download_request, parse_error = _parse_datasets_download_request(args[1:])
        if parse_error == "__help__":
            print(USAGE)
            return 0
        if parse_error is not None:
            print(parse_error, file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2
        if download_request is None:  # pragma: no cover - parse_error branch guards this
            print("datasets_download_request_invalid", file=sys.stderr)
            return 2

        try:
            download_payload = api.download_numerai_dataset(download_request)
        except PackageError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(download_payload.model_dump_json())
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    print(f"unknown arguments: numerai datasets {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _handle_numerai_models_command(args: Sequence[str]) -> int:
    models_request, parse_error = _parse_models_request(args)
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if models_request is None:  # pragma: no cover - parse_error branch guards this
        print("models_request_invalid", file=sys.stderr)
        return 2

    try:
        payload = api.list_numerai_models(models_request)
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(payload.model_dump_json())
    return 0


def _handle_numerai_round_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] != "current":
        print(f"unknown arguments: numerai round {' '.join(args)}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    round_request, parse_error = _parse_current_round_request(args[1:])
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if round_request is None:  # pragma: no cover - parse_error branch guards this
        print("current_round_request_invalid", file=sys.stderr)
        return 2

    try:
        payload = api.get_numerai_current_round(round_request)
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(payload.model_dump_json())
    return 0


def _handle_numerai_forum_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] != "scrape":
        print(f"unknown arguments: numerai forum {' '.join(args)}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    options, parse_error = _parse_forum_scrape_options(args[1:])
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if options is None:  # pragma: no cover - parse_error branch guards this
        print("forum_scrape_options_invalid", file=sys.stderr)
        return 2

    output_dir = str(options.get("output_dir", "docs/numerai/forum"))
    state_path_value = options.get("state_path")
    state_path = str(state_path_value) if isinstance(state_path_value, str) else None
    full_refresh = bool(options.get("full_refresh", False))

    try:
        payload = api.scrape_numerai_forum(
            output_dir=output_dir,
            state_path=state_path,
            full_refresh=full_refresh,
        )
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(payload.model_dump_json())
    return 0


def handle_numerai_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0

    if args[0] == "datasets":
        return _handle_numerai_datasets_command(args[1:])
    if args[0] == "models":
        return _handle_numerai_models_command(args[1:])
    if args[0] == "round":
        return _handle_numerai_round_command(args[1:])
    if args[0] == "forum":
        return _handle_numerai_forum_command(args[1:])
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    print(f"unknown arguments: numerai {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_numerai_command"]
