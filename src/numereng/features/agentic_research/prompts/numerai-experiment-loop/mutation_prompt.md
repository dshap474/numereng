You are evolving one Numereng training config for the `agentic_research` strategy `numerai-experiment-loop`.

Your job is only to decide what exact setting changes should be applied to the next config.
Python will clone the parent config, name the new file, validate it, and run training.

Optimization policy:
- maximize `bmc_last_200_eras_mean`
- use `bmc_mean` as the tie-break
- use `corr_mean` as the sanity check
- prefer small, targeted mutations over wide jumps
- change 1 to 3 settings only

Output contract:
- return plain text only
- do not use code fences
- do not return JSON
- return exactly these two sections in this order:

RATIONALE:
<brief rationale>

CHANGES:
config.some.path = <valid JSON literal>
config.other.path = <valid JSON literal>

Rules:
- every change path must start with `config.`
- every value on the right side of `=` must be valid JSON
- only use these config paths:
$ALLOWED_PATHS
- do not invent filenames
- do not emit shell commands
- do not restate the parent config

Parent config filename:
`$PARENT_CONFIG_FILENAME`

Core metrics for the parent run:
$CORE_METRIC_SUMMARY

Recent lineage summary:
$RECENT_LINEAGE_SUMMARY

Parent config JSON:
```json
$PARENT_CONFIG_JSON
```

$VALIDATION_FEEDBACK_BLOCK
