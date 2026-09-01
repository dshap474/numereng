# Official Sources

As-of verification date: 2026-03-21.

## Primary Sources

1. Numerai GraphQL endpoint
- https://api-tournament.numer.ai
- Live schema introspection used to confirm root fields and argument shapes for GraphQL parity helpers.

2. Official NumerAPI documentation
- https://numerapi.readthedocs.io/en/latest/api/numerapi.html
- https://numerapi.readthedocs.io/en/latest/api/numerapi.base_api.html
- https://numerapi.readthedocs.io/en/latest/api/numerapi.numerapi.html
- https://numerapi.readthedocs.io/en/latest/api/numerapi.signalsapi.html
- https://numerapi.readthedocs.io/en/latest/api/numerapi.cryptoapi.html

3. Official Numerai docs for API-first flows
- https://docs.numer.ai/numerai-tournament/submissions
- https://docs.numer.ai/numerai-tournament/submissions/model-uploads
- https://docs.numer.ai/numerai-signals/submissions
- https://docs.numer.ai/numerai-tournament/data

4. Official Numerai API introduction
- https://blog.numer.ai/getting-started-with-numerais-new-tournament-api/
- Documents raw GraphQL request patterns and token auth syntax.

5. Official Numerai MCP docs used only as a parity reference
- https://docs.numer.ai/numerai-tournament/mcp
- Useful for mapping some MCP-equivalent operations and scope expectations, but not used as an execution path in this skill.

## Source Precedence

Use this order when conflicts appear:
1. Live GraphQL introspection for root names and argument contracts
2. Official Numerai docs (`docs.numer.ai`)
3. Official NumerAPI documentation
4. Vendored client code in this repository

## Validation Notes

- The API parity helpers in this skill were aligned to live schema roots visible without authentication on 2026-03-21.
- The vendored `numerapi` inventory in this skill was refreshed from repository-local source files, not from stale secondary notes.
