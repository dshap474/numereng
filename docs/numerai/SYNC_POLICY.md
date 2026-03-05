# Numerai Docs Sync Policy

This directory is a synced copy of upstream Numerai docs.

## Source

- Upstream repository: `https://github.com/numerai/docs.git`
- Local mirror path: `docs/numerai/`
- Sync workflow: manual upstream mirror refresh (`git clone` + `rsync`)

## Policy

1. `docs/numerai/` content is synced from upstream and treated as vendor docs.
2. Local sync metadata can be recorded in `.sync-meta.json` (gitignored) with upstream commit SHA and sync timestamp.
3. This `SYNC_POLICY.md` file is maintained locally.
4. Do not edit mirrored upstream pages in place; update by re-running sync against upstream.
5. `docs/numerai/forum/` is a local generated export target used by `numereng numerai forum scrape`; it is not part of upstream mirrored docs pages.

## Sync Commands

```bash
tmp_dir="$(mktemp -d)"
git clone --depth 1 https://github.com/numerai/docs.git "$tmp_dir/docs"
rsync -a --delete \
  --exclude 'SYNC_POLICY.md' \
  --exclude '.sync-meta.json' \
  --exclude 'forum/' \
  "$tmp_dir/docs/" docs/numerai/
rm -rf "$tmp_dir"
```
