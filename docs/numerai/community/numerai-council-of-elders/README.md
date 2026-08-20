# Numerai Council of Elders: Out of Sample

Index of the [Numerai Council of Elders YouTube channel](https://www.youtube.com/@NumeraiCouncilofElders). This is independent community content, not official Numerai documentation or financial advice.

The channel listing below reflects every upload found on 2026-08-07: three full podcast episodes and four short trailers or announcements.

Full transcripts are a **local-only archive** and are deliberately not redistributed in this repository, for licensing reasons. They are regenerated locally from YouTube's English auto-captions (see [Refresh](#refresh)) into untracked `episodes/` and `shorts/` directories. Auto-captions may contain errors. Watch or read the captions on YouTube for the authoritative source.

## Episodes

Full interviews, in release order — the recommended listening order.

| # | Published | Duration | Title | YouTube |
| - | --- | ---: | --- | --- |
| 1 | 2026-07-17 | 53:59 | Building AI Quant Agents That Code While You Sleep | [LHNUDCmUrxU](https://www.youtube.com/watch?v=LHNUDCmUrxU) |
| 2 | 2026-07-22 | 45:36 | How ideas from modern AI research can be applied to quantitative finance | [F764ME9JFMk](https://www.youtube.com/watch?v=F764ME9JFMk) |
| 3 | 2026-07-24 | 01:14:49 | Building better machine learning models through continuous experimentation | [FTyq6vIfMUc](https://www.youtube.com/watch?v=FTyq6vIfMUc) |

## Shorts

Trailers and announcements (under a minute each; no standalone content).

| Published | Duration | Title | YouTube |
| --- | ---: | --- | --- |
| 2026-07-17 | 00:43 | Out of Sample // Coming Soon | [PtDzhJmLGMU](https://www.youtube.com/watch?v=PtDzhJmLGMU) |
| 2026-07-17 | 00:33 | Out of Sample Episode One // Trailer | [tI5BPWDvtro](https://www.youtube.com/watch?v=tI5BPWDvtro) |
| 2026-07-17 | 00:32 | Out of Sample Episode Two // Trailer | [yeQVC7P2ZJs](https://www.youtube.com/watch?v=yeQVC7P2ZJs) |
| 2026-07-23 | 00:42 | Out of Sample Episode Three // Trailer | [VYkCoFofd_g](https://www.youtube.com/watch?v=VYkCoFofd_g) |

## Local archive layout

All of the following are local-only and untracked:

- `episodes/` holds cleaned, timestamped reading copies of full interviews.
- `shorts/` holds trailer and announcement transcripts.
- `raw/*.en-orig.json3` contains the non-duplicated caption events used to render Markdown.
- `raw/*.en-orig.vtt` contains the original WebVTT auto-captions.
- `raw/*.info.json` contains yt-dlp video and channel metadata.
- No audio or video media is downloaded.

## Refresh

Run from the repository root:

```bash
yt-dlp --no-update --ignore-errors --skip-download --write-auto-subs --sub-langs en-orig --sub-format vtt --write-info-json --write-playlist-metafiles --output 'docs/numerai/community/numerai-council-of-elders/raw/%(id)s.%(ext)s' 'https://www.youtube.com/@NumeraiCouncilofElders/videos'
yt-dlp --no-update --ignore-errors --skip-download --write-auto-subs --sub-langs en-orig --sub-format json3 --output 'docs/numerai/community/numerai-council-of-elders/raw/%(id)s.%(ext)s' 'https://www.youtube.com/@NumeraiCouncilofElders/videos'
```

Render new episode transcripts into `episodes/` and shorts into `shorts/`, keeping the `<date>-<slug>-<youtube_id>.md` filename convention. Everything produced by this refresh stays local-only; do not commit it.
