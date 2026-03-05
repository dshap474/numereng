"""Deterministic Discourse forum scraper for Numerai markdown exports."""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from numereng.platform.errors import ForumScraperError

_POSTS_ENDPOINT = "/posts.json"
_POSTS_DIRNAME = "posts"
_INDEX_FILENAME = "INDEX.md"
_MANIFEST_FILENAME = ".forum_scraper_manifest.json"
_STATE_FILENAME = ".forum_scraper_state.json"
_DEFAULT_BASE_URL = "https://forum.numer.ai"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_SLEEP_SECONDS = 0.02
_DEFAULT_USER_AGENT = "numereng-forum-scraper/1.0"
_MAX_RETRIES = 4

_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_CLOSE_P_RE = re.compile(r"</p\s*>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

PostPayload = Mapping[str, Any]
FetchPostsPageFn = Callable[[int | None], Sequence[PostPayload]]


def scrape_forum_posts(
    *,
    output_dir: str | Path = "docs/numerai/forum",
    state_path: str | Path | None = None,
    full_refresh: bool = False,
    base_url: str = _DEFAULT_BASE_URL,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    _fetch_posts_page_fn: FetchPostsPageFn | None = None,
) -> dict[str, object]:
    """Scrape forum posts and materialize markdown artifacts."""

    started_at = _utc_now_iso()
    try:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ForumScraperError("forum_scraper_output_dir_unavailable") from exc

    try:
        resolved_state_path = (
            Path(state_path).expanduser().resolve()
            if state_path is not None
            else (resolved_output_dir / _STATE_FILENAME)
        )
    except OSError as exc:
        raise ForumScraperError("forum_scraper_state_path_unavailable") from exc

    posts_dir = resolved_output_dir / _POSTS_DIRNAME
    index_path = resolved_output_dir / _INDEX_FILENAME
    manifest_path = resolved_output_dir / _MANIFEST_FILENAME

    existing_entries: dict[int, dict[str, Any]] = {}
    latest_checkpoint: int | None = None
    incremental_mode = not full_refresh
    if incremental_mode:
        existing_entries = _load_manifest_entries(manifest_path)
        state_payload = _read_json_dict(resolved_state_path)
        latest_checkpoint = _coerce_int(state_payload.get("latest_post_id"))
        if latest_checkpoint is None or not existing_entries:
            incremental_mode = False

    mode = "incremental" if incremental_mode else "full"
    write_output_dir = resolved_output_dir
    write_index_path = index_path
    write_manifest_path = manifest_path
    staging_dir: Path | None = None

    if mode == "full":
        staging_dir = _create_full_refresh_staging_dir(output_dir=resolved_output_dir)
        write_output_dir = staging_dir
        write_index_path = staging_dir / _INDEX_FILENAME
        write_manifest_path = staging_dir / _MANIFEST_FILENAME
        existing_entries = {}
        latest_checkpoint = None

    fetch_posts_page = (
        _fetch_posts_page_fn
        if _fetch_posts_page_fn is not None
        else (lambda before: _fetch_posts_page(base_url=base_url, before=before, timeout_seconds=timeout_seconds))
    )

    pages_fetched = 0
    next_before: int | None = None
    seen_payload_ids: set[int] = set()
    collected_payloads: list[PostPayload] = []

    while True:
        payloads = list(fetch_posts_page(next_before))
        pages_fetched += 1
        if not payloads:
            break

        min_page_id: int | None = None
        reached_checkpoint = False
        for payload in payloads:
            post_id = _coerce_int(payload.get("id"))
            if post_id is None:
                continue
            if min_page_id is None or post_id < min_page_id:
                min_page_id = post_id
            if post_id in seen_payload_ids:
                continue
            seen_payload_ids.add(post_id)
            if latest_checkpoint is not None and post_id <= latest_checkpoint:
                reached_checkpoint = True
                continue
            collected_payloads.append(payload)

        if latest_checkpoint is not None and reached_checkpoint:
            break
        if min_page_id is None:
            break
        if next_before is not None and min_page_id >= next_before:
            break
        next_before = min_page_id
        if _fetch_posts_page_fn is None and _DEFAULT_SLEEP_SECONDS > 0:
            time.sleep(_DEFAULT_SLEEP_SECONDS)

    new_payloads_by_id: dict[int, PostPayload] = {}
    for payload in collected_payloads:
        post_id = _coerce_int(payload.get("id"))
        if post_id is None:
            continue
        new_payloads_by_id[post_id] = payload

    try:
        new_entries: list[dict[str, Any]] = []
        ordered_new_payloads = sorted(new_payloads_by_id.values(), key=_payload_sort_key)
        for payload in ordered_new_payloads:
            entry = _build_manifest_entry(payload=payload, base_url=base_url)
            body = _extract_markdown_body(payload)
            relative_path = _write_post_markdown(
                output_dir=write_output_dir,
                entry=entry,
                body=body,
            )
            entry["path"] = relative_path
            post_id = _coerce_int(entry.get("post_id"))
            if post_id is None:
                raise ForumScraperError("forum_scraper_post_id_missing_after_write")
            existing_entries[post_id] = entry
            new_entries.append(entry)

        all_entries = sorted(existing_entries.values(), key=_entry_sort_key)
        completed_at = _utc_now_iso()
        _write_index(
            output_dir=write_output_dir,
            index_path=write_index_path,
            entries=all_entries,
            generated_at=completed_at,
            mode=mode,
            new_posts=len(new_entries),
        )
        _write_json_dict(
            write_manifest_path,
            {
                "source": f"{base_url.rstrip('/')}{_POSTS_ENDPOINT}",
                "generated_at": completed_at,
                "mode": mode,
                "full_refresh": mode == "full",
                "pages_fetched": pages_fetched,
                "new_posts": len(new_entries),
                "total_posts": len(all_entries),
                "posts": all_entries,
            },
        )

        if mode == "full":
            if staging_dir is None:
                raise ForumScraperError("forum_scraper_staging_not_initialized")
            _promote_staged_outputs(
                output_dir=resolved_output_dir,
                staging_dir=staging_dir,
                posts_dir=posts_dir,
                index_path=index_path,
                manifest_path=manifest_path,
            )
            staging_dir = None

        latest_post_id = max(existing_entries) if existing_entries else None
        oldest_post_id = min(existing_entries) if existing_entries else None
        _write_json_dict(
            resolved_state_path,
            {
                "updated_at": completed_at,
                "latest_post_id": latest_post_id,
                "oldest_post_id": oldest_post_id,
                "total_posts": len(all_entries),
                "manifest_path": str(manifest_path),
                "index_path": str(index_path),
            },
        )

        return {
            "output_dir": str(resolved_output_dir),
            "posts_dir": str(posts_dir),
            "index_path": str(index_path),
            "manifest_path": str(manifest_path),
            "state_path": str(resolved_state_path),
            "mode": mode,
            "pages_fetched": pages_fetched,
            "fetched_posts": len(ordered_new_payloads),
            "new_posts": len(new_entries),
            "total_posts": len(all_entries),
            "latest_post_id": latest_post_id,
            "oldest_post_id": oldest_post_id,
            "started_at": started_at,
            "completed_at": completed_at,
        }
    finally:
        if staging_dir is not None:
            _remove_tree_if_exists(staging_dir)


def _fetch_posts_page(*, base_url: str, before: int | None, timeout_seconds: float) -> list[PostPayload]:
    """Fetch one Discourse posts page with retry/backoff."""

    if before is None:
        target = f"{base_url.rstrip('/')}{_POSTS_ENDPOINT}"
    else:
        target = f"{base_url.rstrip('/')}{_POSTS_ENDPOINT}?before={before}"

    attempt = 0
    while attempt < _MAX_RETRIES:
        request = Request(target, headers={"User-Agent": _DEFAULT_USER_AGENT})  # noqa: S310
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                raw_payload = response.read()
            try:
                decoded_payload = raw_payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ForumScraperError("forum_scraper_invalid_utf8_response") from exc
            payload = json.loads(decoded_payload)
        except HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt + 1 < _MAX_RETRIES:
                time.sleep(_retry_delay(exc, attempt))
                attempt += 1
                continue
            raise ForumScraperError(f"forum_scraper_http_error:{exc.code}") from exc
        except URLError as exc:
            if attempt + 1 < _MAX_RETRIES:
                time.sleep(0.5 * (2**attempt))
                attempt += 1
                continue
            raise ForumScraperError("forum_scraper_network_error") from exc
        except json.JSONDecodeError as exc:
            raise ForumScraperError("forum_scraper_invalid_json_response") from exc

        posts = payload.get("latest_posts")
        if not isinstance(posts, list):
            raise ForumScraperError("forum_scraper_response_missing_latest_posts")
        records: list[PostPayload] = []
        for item in posts:
            if isinstance(item, Mapping):
                records.append(item)
        return records

    raise ForumScraperError("forum_scraper_fetch_exhausted_retries")


def _retry_delay(error: HTTPError, attempt: int) -> float:
    retry_after_value = error.headers.get("Retry-After")
    retry_after = str(retry_after_value) if retry_after_value is not None else None
    if retry_after is not None:
        try:
            parsed = float(retry_after)
        except ValueError:
            parsed = 0.0
        if parsed > 0:
            return 10.0 if parsed > 10.0 else parsed
    fallback_delay = 0.5 * (2**attempt)
    return 10.0 if fallback_delay > 10.0 else fallback_delay


def _load_manifest_entries(manifest_path: Path) -> dict[int, dict[str, Any]]:
    payload = _read_json_dict(manifest_path)
    posts = payload.get("posts")
    if not isinstance(posts, list):
        return {}

    loaded: dict[int, dict[str, Any]] = {}
    for item in posts:
        if not isinstance(item, dict):
            continue
        post_id = _coerce_int(item.get("post_id"))
        if post_id is None:
            continue
        normalized = _normalize_manifest_entry(item)
        loaded[post_id] = normalized
    return loaded


def _normalize_manifest_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "post_id": _coerce_int(raw.get("post_id")) or 0,
        "topic_id": _coerce_int(raw.get("topic_id")) or 0,
        "post_number": _coerce_int(raw.get("post_number")) or 0,
        "topic_slug": _coerce_str(raw.get("topic_slug")) or "",
        "topic_title": _coerce_str(raw.get("topic_title")) or "",
        "username": _coerce_str(raw.get("username")) or "unknown",
        "created_at": _coerce_str(raw.get("created_at")) or "",
        "updated_at": _coerce_str(raw.get("updated_at")),
        "url": _coerce_str(raw.get("url")) or "",
        "path": _coerce_str(raw.get("path")) or "",
    }


def _build_manifest_entry(*, payload: PostPayload, base_url: str) -> dict[str, Any]:
    post_id = _coerce_int(payload.get("id"))
    if post_id is None:
        raise ForumScraperError("forum_scraper_post_missing_id")

    topic_id = _coerce_int(payload.get("topic_id")) or 0
    topic_slug = _coerce_str(payload.get("topic_slug")) or f"topic-{topic_id}"
    topic_title = _coerce_str(payload.get("topic_title")) or topic_slug
    post_number = _coerce_int(payload.get("post_number")) or 0

    return {
        "post_id": post_id,
        "topic_id": topic_id,
        "post_number": post_number,
        "topic_slug": topic_slug,
        "topic_title": topic_title,
        "username": _coerce_str(payload.get("username")) or "unknown",
        "created_at": _coerce_str(payload.get("created_at")) or "",
        "updated_at": _coerce_str(payload.get("updated_at")),
        "url": _normalize_post_url(
            base_url=base_url,
            payload=payload,
            topic_slug=topic_slug,
            topic_id=topic_id,
            post_number=post_number,
        ),
    }


def _normalize_post_url(
    *,
    base_url: str,
    payload: Mapping[str, Any],
    topic_slug: str,
    topic_id: int,
    post_number: int,
) -> str:
    raw_url = _coerce_str(payload.get("post_url"))
    if raw_url is not None:
        parsed_url = urlparse(raw_url)
        if parsed_url.scheme in {"http", "https"} and parsed_url.netloc:
            return raw_url
        if parsed_url.scheme or parsed_url.netloc:
            return _default_post_url(
                base_url=base_url,
                topic_slug=topic_slug,
                topic_id=topic_id,
                post_number=post_number,
            )
        normalized_path = parsed_url.path
        if normalized_path:
            if not normalized_path.startswith("/"):
                normalized_path = f"/{normalized_path}"
            normalized_url = normalized_path
            if parsed_url.query:
                normalized_url = f"{normalized_url}?{parsed_url.query}"
            if parsed_url.fragment:
                normalized_url = f"{normalized_url}#{parsed_url.fragment}"
            return urljoin(f"{base_url.rstrip('/')}/", normalized_url)

    return _default_post_url(
        base_url=base_url,
        topic_slug=topic_slug,
        topic_id=topic_id,
        post_number=post_number,
    )


def _default_post_url(*, base_url: str, topic_slug: str, topic_id: int, post_number: int) -> str:
    if post_number > 0:
        return f"{base_url.rstrip('/')}/t/{topic_slug}/{topic_id}/{post_number}"
    return f"{base_url.rstrip('/')}/t/{topic_slug}/{topic_id}"


def _extract_markdown_body(payload: Mapping[str, Any]) -> str:
    raw_body = _coerce_str(payload.get("raw"))
    if raw_body:
        return _clean_markdown(raw_body)

    excerpt = _coerce_str(payload.get("excerpt"))
    if excerpt:
        return _clean_markdown(excerpt)

    cooked = _coerce_str(payload.get("cooked"))
    if cooked:
        return _clean_markdown(_strip_html(cooked))

    return "_(empty post body)_"


def _strip_html(value: str) -> str:
    with_breaks = _HTML_BREAK_RE.sub("\n", value)
    with_paragraphs = _HTML_CLOSE_P_RE.sub("\n\n", with_breaks)
    no_tags = _HTML_TAG_RE.sub("", with_paragraphs)
    return unescape(no_tags)


def _clean_markdown(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.rstrip() for line in normalized.split("\n")]

    collapsed: list[str] = []
    blank_streak = 0
    for line in lines:
        if line == "":
            blank_streak += 1
            if blank_streak > 2:
                continue
        else:
            blank_streak = 0
        collapsed.append(line)

    cleaned = "\n".join(collapsed).strip()
    return cleaned if cleaned else "_(empty post body)_"


def _write_post_markdown(*, output_dir: Path, entry: Mapping[str, Any], body: str) -> str:
    post_id = _coerce_int(entry.get("post_id"))
    if post_id is None:
        raise ForumScraperError("forum_scraper_post_id_missing")

    created_at = _coerce_str(entry.get("created_at")) or ""
    created_timestamp = _parse_timestamp(created_at)
    relative_path = (
        f"{_POSTS_DIRNAME}/{created_timestamp.year:04d}/{created_timestamp.month:02d}/{post_id}.md"
    )
    post_path = output_dir / relative_path

    lines = [
        f"# {entry.get('topic_title', '')}",
        "",
        f"- Post ID: `{post_id}`",
        f"- Topic ID: `{entry.get('topic_id', 0)}`",
        f"- Post Number: `{entry.get('post_number', 0)}`",
        f"- Author: `@{entry.get('username', 'unknown')}`",
        f"- Created: `{entry.get('created_at', '')}`",
    ]
    updated_at = _coerce_str(entry.get("updated_at"))
    if updated_at:
        lines.append(f"- Updated: `{updated_at}`")
    lines.extend(
        [
            f"- URL: <{entry.get('url', '')}>",
            "",
            "---",
            "",
            body,
            "",
        ]
    )
    _write_text(post_path, "\n".join(lines))
    return relative_path


def _write_index(
    *,
    output_dir: Path,
    index_path: Path,
    entries: list[dict[str, Any]],
    generated_at: str,
    mode: str,
    new_posts: int,
) -> None:
    grouped_entries = _group_entries_by_year_month(entries)

    lines = [
        "# Numerai Forum Archive",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Mode: `{mode}`",
        f"- Total posts: `{len(entries)}`",
        f"- New posts this run: `{new_posts}`",
        "",
        "## Browse by Year",
        "",
    ]

    for year, month_map in grouped_entries.items():
        year_post_count = sum(len(month_entries) for month_entries in month_map.values())
        lines.append(
            f"- [{year}]({_POSTS_DIRNAME}/{year}/{_INDEX_FILENAME}) — `{year_post_count}` posts"
        )

    lines.append("")
    _write_text(index_path, "\n".join(lines))
    _write_year_month_indexes(
        output_dir=output_dir,
        grouped_entries=grouped_entries,
        generated_at=generated_at,
    )


def _group_entries_by_year_month(entries: Sequence[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for entry in entries:
        created_at = _coerce_str(entry.get("created_at")) or ""
        created_timestamp = _parse_timestamp(created_at)
        year = f"{created_timestamp.year:04d}"
        month = f"{created_timestamp.month:02d}"
        grouped.setdefault(year, {}).setdefault(month, []).append(entry)

    return {
        year: {month: grouped[year][month] for month in sorted(grouped[year])}
        for year in sorted(grouped)
    }


def _write_year_month_indexes(
    *,
    output_dir: Path,
    grouped_entries: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    generated_at: str,
) -> None:
    for year, month_map in grouped_entries.items():
        year_lines = [
            f"# Numerai Forum Archive — {year}",
            "",
            f"- Generated at: `{generated_at}`",
            f"- Total posts: `{sum(len(entries) for entries in month_map.values())}`",
            "",
            "## Months",
            "",
        ]

        for month, month_entries in month_map.items():
            year_lines.append(
                f"- [{year}/{month}]({month}/{_INDEX_FILENAME}) — `{len(month_entries)}` posts"
            )
            _write_month_index(
                output_dir=output_dir,
                year=year,
                month=month,
                entries=month_entries,
                generated_at=generated_at,
            )

        year_lines.extend(
            [
                "",
                f"- [Back to archive index](../../{_INDEX_FILENAME})",
                "",
            ]
        )
        year_index_path = output_dir / _POSTS_DIRNAME / year / _INDEX_FILENAME
        _write_text(year_index_path, "\n".join(year_lines))


def _write_month_index(
    *,
    output_dir: Path,
    year: str,
    month: str,
    entries: Sequence[dict[str, Any]],
    generated_at: str,
) -> None:
    lines = [
        f"# Numerai Forum Archive — {year}/{month}",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Total posts: `{len(entries)}`",
        "",
        "## Posts (oldest first)",
        "",
    ]

    for entry in entries:
        topic_title = _coerce_str(entry.get("topic_title")) or "(untitled)"
        username = _coerce_str(entry.get("username")) or "unknown"
        post_id = _coerce_int(entry.get("post_id")) or 0
        created_at = _coerce_str(entry.get("created_at")) or ""
        lines.append(
            f"- {created_at} — [{topic_title}]({post_id}.md) — `@{username}` — post `{post_id}`"
        )

    lines.extend(
        [
            "",
            "- [Back to year index](../INDEX.md)",
            f"- [Back to archive index](../../../{_INDEX_FILENAME})",
            "",
        ]
    )
    month_index_path = output_dir / _POSTS_DIRNAME / year / month / _INDEX_FILENAME
    _write_text(month_index_path, "\n".join(lines))


def _payload_sort_key(payload: Mapping[str, Any]) -> tuple[datetime, int]:
    created_at = _coerce_str(payload.get("created_at")) or ""
    post_id = _coerce_int(payload.get("id")) or 0
    return (_parse_timestamp(created_at), post_id)


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[datetime, int]:
    created_at = _coerce_str(entry.get("created_at")) or ""
    post_id = _coerce_int(entry.get("post_id")) or 0
    return (_parse_timestamp(created_at), post_id)


def _parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        normalized = value[:-1] + "+00:00"
    else:
        normalized = value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _create_full_refresh_staging_dir(*, output_dir: Path) -> Path:
    staging_dir = output_dir.parent / f".{output_dir.name}.forum_scraper_staging_{time.time_ns()}"
    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ForumScraperError("forum_scraper_staging_dir_create_failed") from exc
    return staging_dir


def _promote_staged_outputs(
    *,
    output_dir: Path,
    staging_dir: Path,
    posts_dir: Path,
    index_path: Path,
    manifest_path: Path,
) -> None:
    staged_posts_dir = staging_dir / _POSTS_DIRNAME
    staged_index_path = staging_dir / _INDEX_FILENAME
    staged_manifest_path = staging_dir / _MANIFEST_FILENAME
    if not staged_posts_dir.exists() or not staged_index_path.exists() or not staged_manifest_path.exists():
        raise ForumScraperError("forum_scraper_staging_missing_outputs")

    backup_root = output_dir.parent / f".{output_dir.name}.forum_scraper_backup_{time.time_ns()}"
    moved_backups: list[tuple[Path, Path]] = []
    promoted_targets: list[Path] = []
    promoted_successfully = False
    targets = (
        (posts_dir, staged_posts_dir, backup_root / _POSTS_DIRNAME),
        (index_path, staged_index_path, backup_root / _INDEX_FILENAME),
        (manifest_path, staged_manifest_path, backup_root / _MANIFEST_FILENAME),
    )

    try:
        backup_root.mkdir(parents=True, exist_ok=False)
        for target_path, _, backup_path in targets:
            if target_path.exists():
                target_path.replace(backup_path)
                moved_backups.append((target_path, backup_path))
        for target_path, staged_path, _ in targets:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(target_path)
            promoted_targets.append(target_path)
        promoted_successfully = True
    except OSError as exc:
        _rollback_promote(moved_backups=moved_backups, promoted_targets=promoted_targets)
        raise ForumScraperError("forum_scraper_promote_failed") from exc
    finally:
        _remove_tree_if_exists(staging_dir)
        if promoted_successfully:
            _remove_tree_if_exists(backup_root)


def _rollback_promote(*, moved_backups: Sequence[tuple[Path, Path]], promoted_targets: Sequence[Path]) -> None:
    for target_path in promoted_targets:
        _remove_path_if_exists(target_path)

    for target_path, backup_path in reversed(tuple(moved_backups)):
        try:
            if backup_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.replace(target_path)
        except OSError:
            continue


def _remove_tree_if_exists(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        pass


def _remove_path_if_exists(path: Path) -> None:
    try:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
            return
        path.unlink()
    except OSError:
        pass


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_dict(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_text(path, content)


def _write_text(path: Path, content: str) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        raise ForumScraperError("forum_scraper_write_failed") from exc


def _coerce_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["scrape_forum_posts"]
