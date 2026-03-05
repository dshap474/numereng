from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from numereng.platform import forum_scraper
from numereng.platform.errors import ForumScraperError
from numereng.platform.forum_scraper import scrape_forum_posts


def _post_payload(
    *,
    post_id: int,
    topic_id: int,
    post_number: int,
    topic_slug: str,
    topic_title: str,
    username: str,
    created_at: str,
    raw: str,
) -> dict[str, Any]:
    return {
        "id": post_id,
        "topic_id": topic_id,
        "post_number": post_number,
        "topic_slug": topic_slug,
        "topic_title": topic_title,
        "username": username,
        "created_at": created_at,
        "updated_at": created_at,
        "post_url": f"/t/{topic_slug}/{topic_id}/{post_number}",
        "raw": raw,
    }


def _extract_index_post_ids(index_text: str) -> list[int]:
    return [int(match) for match in re.findall(r"post `(\d+)`", index_text)]


def _post_path(output_dir: Path, *, year: int, month: int, post_id: int) -> Path:
    return output_dir / "posts" / f"{year:04d}" / f"{month:02d}" / f"{post_id}.md"


def _month_index_path(output_dir: Path, *, year: int, month: int) -> Path:
    return output_dir / "posts" / f"{year:04d}" / f"{month:02d}" / "INDEX.md"


def _year_index_path(output_dir: Path, *, year: int) -> Path:
    return output_dir / "posts" / f"{year:04d}" / "INDEX.md"


def test_scrape_forum_posts_full_refresh_writes_sorted_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "forum"
    pages: dict[int | None, list[dict[str, Any]]] = {
        None: [
            _post_payload(
                post_id=3,
                topic_id=10,
                post_number=3,
                topic_slug="alpha",
                topic_title="Alpha",
                username="charlie",
                created_at="2020-03-01T00:00:00.000Z",
                raw="third post",
            ),
            _post_payload(
                post_id=2,
                topic_id=10,
                post_number=2,
                topic_slug="alpha",
                topic_title="Alpha",
                username="bravo",
                created_at="2020-02-01T00:00:00.000Z",
                raw="second post",
            ),
        ],
        2: [
            _post_payload(
                post_id=1,
                topic_id=10,
                post_number=1,
                topic_slug="alpha",
                topic_title="Alpha",
                username="alpha",
                created_at="2020-01-01T00:00:00.000Z",
                raw="first line  \n\n\nsecond line",
            )
        ],
        1: [],
    }

    result = scrape_forum_posts(
        output_dir=output_dir,
        full_refresh=True,
        _fetch_posts_page_fn=lambda before: pages.get(before, []),
    )

    assert result["mode"] == "full"
    assert result["total_posts"] == 3
    assert result["new_posts"] == 3
    assert result["latest_post_id"] == 3
    assert result["oldest_post_id"] == 1

    archive_index_text = (output_dir / "INDEX.md").read_text(encoding="utf-8")
    assert "- [2020](posts/2020/INDEX.md) — `3` posts" in archive_index_text
    year_index_text = _year_index_path(output_dir, year=2020).read_text(encoding="utf-8")
    assert "- [2020/01](01/INDEX.md) — `1` posts" in year_index_text
    assert "- [2020/02](02/INDEX.md) — `1` posts" in year_index_text
    assert "- [2020/03](03/INDEX.md) — `1` posts" in year_index_text
    month_index_text = _month_index_path(output_dir, year=2020, month=1).read_text(encoding="utf-8")
    assert _extract_index_post_ids(month_index_text) == [1]

    post_body = _post_path(output_dir, year=2020, month=1, post_id=1).read_text(encoding="utf-8")
    assert "first line\n\n\nsecond line" in post_body

    state_payload = json.loads((output_dir / ".forum_scraper_state.json").read_text(encoding="utf-8"))
    assert state_payload["latest_post_id"] == 3
    assert state_payload["total_posts"] == 3


def test_scrape_forum_posts_incremental_appends_new_posts(tmp_path: Path) -> None:
    output_dir = tmp_path / "forum"
    initial_pages: dict[int | None, list[dict[str, Any]]] = {
        None: [
            _post_payload(
                post_id=2,
                topic_id=20,
                post_number=2,
                topic_slug="beta",
                topic_title="Beta",
                username="beta2",
                created_at="2020-02-01T00:00:00.000Z",
                raw="second",
            ),
            _post_payload(
                post_id=1,
                topic_id=20,
                post_number=1,
                topic_slug="beta",
                topic_title="Beta",
                username="beta1",
                created_at="2020-01-01T00:00:00.000Z",
                raw="first",
            ),
        ],
        1: [],
    }
    scrape_forum_posts(
        output_dir=output_dir,
        full_refresh=True,
        _fetch_posts_page_fn=lambda before: initial_pages.get(before, []),
    )

    incremental_pages: dict[int | None, list[dict[str, Any]]] = {
        None: [
            _post_payload(
                post_id=4,
                topic_id=20,
                post_number=4,
                topic_slug="beta",
                topic_title="Beta",
                username="beta4",
                created_at="2020-04-01T00:00:00.000Z",
                raw="fourth",
            ),
            _post_payload(
                post_id=3,
                topic_id=20,
                post_number=3,
                topic_slug="beta",
                topic_title="Beta",
                username="beta3",
                created_at="2020-03-01T00:00:00.000Z",
                raw="third",
            ),
            _post_payload(
                post_id=2,
                topic_id=20,
                post_number=2,
                topic_slug="beta",
                topic_title="Beta",
                username="beta2",
                created_at="2020-02-01T00:00:00.000Z",
                raw="second",
            ),
        ]
    }

    result = scrape_forum_posts(
        output_dir=output_dir,
        _fetch_posts_page_fn=lambda before: incremental_pages.get(before, []),
    )

    assert result["mode"] == "incremental"
    assert result["new_posts"] == 2
    assert result["total_posts"] == 4
    assert result["latest_post_id"] == 4
    assert _post_path(output_dir, year=2020, month=1, post_id=1).exists()
    assert _post_path(output_dir, year=2020, month=4, post_id=4).exists()
    april_index_text = _month_index_path(output_dir, year=2020, month=4).read_text(encoding="utf-8")
    assert _extract_index_post_ids(april_index_text) == [4]
    year_index_text = _year_index_path(output_dir, year=2020).read_text(encoding="utf-8")
    assert "- [2020/04](04/INDEX.md) — `1` posts" in year_index_text


def test_scrape_forum_posts_incremental_no_new_posts_is_noop_for_post_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "forum"
    full_pages: dict[int | None, list[dict[str, Any]]] = {
        None: [
            _post_payload(
                post_id=2,
                topic_id=33,
                post_number=2,
                topic_slug="gamma",
                topic_title="Gamma",
                username="gamma2",
                created_at="2020-02-01T00:00:00.000Z",
                raw="second",
            ),
            _post_payload(
                post_id=1,
                topic_id=33,
                post_number=1,
                topic_slug="gamma",
                topic_title="Gamma",
                username="gamma1",
                created_at="2020-01-01T00:00:00.000Z",
                raw="first",
            ),
        ],
        1: [],
    }
    scrape_forum_posts(
        output_dir=output_dir,
        full_refresh=True,
        _fetch_posts_page_fn=lambda before: full_pages.get(before, []),
    )
    original_post_one = _post_path(output_dir, year=2020, month=1, post_id=1).read_text(encoding="utf-8")
    original_post_two = _post_path(output_dir, year=2020, month=2, post_id=2).read_text(encoding="utf-8")
    original_january_index = _month_index_path(output_dir, year=2020, month=1).read_text(encoding="utf-8")
    original_january_index_post_ids = _extract_index_post_ids(original_january_index)

    incremental_pages: dict[int | None, list[dict[str, Any]]] = {
        None: [
            _post_payload(
                post_id=2,
                topic_id=33,
                post_number=2,
                topic_slug="gamma",
                topic_title="Gamma",
                username="gamma2",
                created_at="2020-02-01T00:00:00.000Z",
                raw="second",
            ),
            _post_payload(
                post_id=1,
                topic_id=33,
                post_number=1,
                topic_slug="gamma",
                topic_title="Gamma",
                username="gamma1",
                created_at="2020-01-01T00:00:00.000Z",
                raw="first",
            ),
        ]
    }
    result = scrape_forum_posts(
        output_dir=output_dir,
        _fetch_posts_page_fn=lambda before: incremental_pages.get(before, []),
    )

    assert result["mode"] == "incremental"
    assert result["new_posts"] == 0
    assert result["total_posts"] == 2
    assert _post_path(output_dir, year=2020, month=1, post_id=1).read_text(encoding="utf-8") == original_post_one
    assert _post_path(output_dir, year=2020, month=2, post_id=2).read_text(encoding="utf-8") == original_post_two
    updated_january_index = _month_index_path(output_dir, year=2020, month=1).read_text(encoding="utf-8")
    assert _extract_index_post_ids(updated_january_index) == original_january_index_post_ids


def test_scrape_forum_posts_full_refresh_preserves_existing_outputs_on_fetch_error(tmp_path: Path) -> None:
    output_dir = tmp_path / "forum"
    seed_pages: dict[int | None, list[dict[str, Any]]] = {
        None: [
            _post_payload(
                post_id=1,
                topic_id=44,
                post_number=1,
                topic_slug="delta",
                topic_title="Delta",
                username="delta1",
                created_at="2020-01-01T00:00:00.000Z",
                raw="seed",
            )
        ],
        1: [],
    }
    scrape_forum_posts(
        output_dir=output_dir,
        full_refresh=True,
        _fetch_posts_page_fn=lambda before: seed_pages.get(before, []),
    )
    existing_index = (output_dir / "INDEX.md").read_text(encoding="utf-8")
    existing_manifest = (output_dir / ".forum_scraper_manifest.json").read_text(encoding="utf-8")
    existing_post = _post_path(output_dir, year=2020, month=1, post_id=1).read_text(encoding="utf-8")
    existing_month_index = _month_index_path(output_dir, year=2020, month=1).read_text(encoding="utf-8")
    existing_year_index = _year_index_path(output_dir, year=2020).read_text(encoding="utf-8")

    def _failing_fetch(before: int | None) -> list[dict[str, Any]]:
        _ = before
        raise ForumScraperError("forum_scraper_network_error")

    with pytest.raises(ForumScraperError, match="forum_scraper_network_error"):
        scrape_forum_posts(
            output_dir=output_dir,
            full_refresh=True,
            _fetch_posts_page_fn=_failing_fetch,
        )

    assert (output_dir / "INDEX.md").read_text(encoding="utf-8") == existing_index
    assert (output_dir / ".forum_scraper_manifest.json").read_text(encoding="utf-8") == existing_manifest
    assert _post_path(output_dir, year=2020, month=1, post_id=1).read_text(encoding="utf-8") == existing_post
    assert _month_index_path(output_dir, year=2020, month=1).read_text(encoding="utf-8") == existing_month_index
    assert _year_index_path(output_dir, year=2020).read_text(encoding="utf-8") == existing_year_index


def test_scrape_forum_posts_wraps_output_dir_creation_error(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "blocked-parent"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ForumScraperError, match="forum_scraper_output_dir_unavailable"):
        scrape_forum_posts(output_dir=blocked_parent / "forum", _fetch_posts_page_fn=lambda before: [])


def test_fetch_posts_page_rejects_non_utf8_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            _ = (exc_type, exc, tb)

        def read(self) -> bytes:
            return b"\xff\xfe"

    monkeypatch.setattr(forum_scraper, "urlopen", lambda request, timeout: _FakeResponse())

    with pytest.raises(ForumScraperError, match="forum_scraper_invalid_utf8_response"):
        forum_scraper._fetch_posts_page(
            base_url="https://forum.numer.ai",
            before=None,
            timeout_seconds=5.0,
        )


def test_normalize_post_url_rejects_protocol_relative_url() -> None:
    normalized = forum_scraper._normalize_post_url(
        base_url="https://forum.numer.ai",
        payload={"post_url": "//evil.example/phishing"},
        topic_slug="epsilon",
        topic_id=99,
        post_number=7,
    )

    assert normalized == "https://forum.numer.ai/t/epsilon/99/7"
