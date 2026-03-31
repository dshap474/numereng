# Comparison Policy

Research-memory updates use two comparison passes.

## 1. Primary relevant subset

This is the main context for deciding what the experiment means.

Prefer experiments that match on as many of these axes as possible:

- target family / horizon
- feature scope
- model family
- evaluation surface
- validation profile
- experiment tags
- stated hypothesis

This pass is strict on comparability.

## 2. Secondary broader sweep

This is a global consistency check across the whole experiment history.

Use it only for:

- contradiction checks
- repeated dead-end detection
- adjacent supporting evidence

Do not let weakly related global evidence override a strong directly comparable subset.

## Evidence classes

- `frontier`
  - direct frontier-shaping evidence on a shared strong surface
- `scout`
  - directional but not direct frontier proof
- `supporting`
  - incomplete, mixed, or otherwise too weak for direct frontier movement

Strong default:

- full `v5.2` + shared strong route such as `purged_walk_forward` -> likely `frontier`
- smoke / `simple` / staged tiers -> likely `scout`
- degraded or mixed evidence -> `supporting`

## Surface discipline

- comparable strong surfaces can directly change defaults and the top-ranked next move
- smoke / `simple` / staged results can narrow menus and identify challengers
- staged wins must be labeled as promotion candidates, not final frontier conclusions
