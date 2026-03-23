#!/usr/bin/env python3
"""Read-only Numerai GraphQL schema introspection helper."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import requests

GRAPHQL_ENDPOINT = "https://api-tournament.numer.ai"
INTROSPECTION_QUERY = """
query IntrospectSchema {
  __schema {
    queryType {
      fields {
        name
        args {
          name
          type {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                }
              }
            }
          }
        }
        type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
            }
          }
        }
      }
    }
    mutationType {
      fields {
        name
        args {
          name
          type {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                }
              }
            }
          }
        }
        type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
            }
          }
        }
      }
    }
    types {
      name
      kind
      fields {
        name
        type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
            }
          }
        }
      }
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=GRAPHQL_ENDPOINT)
    parser.add_argument("--auth-header", default=None)
    parser.add_argument("--section", choices=("query", "mutation", "both"), default="both")
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--type", dest="types", action="append", default=[])
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args()


def resolve_auth_token(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for env_name in ("NUMERAI_API_AUTH", "NUMERAI_MCP_AUTH"):
        value = os.getenv(env_name)
        if value:
            return value
    public_id = os.getenv("NUMERAI_PUBLIC_ID")
    secret_key = os.getenv("NUMERAI_SECRET_KEY")
    if public_id and secret_key:
        return f"Token {public_id}${secret_key}"
    return None


def flatten_type(type_ref: dict[str, Any] | None) -> str:
    parts: list[str] = []
    cursor = type_ref
    while cursor:
        parts.append(cursor.get("name") or cursor.get("kind"))
        cursor = cursor.get("ofType")
    return " ".join(parts)


def fetch_schema(endpoint: str, auth_header: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = resolve_auth_token(auth_header)
    if token:
        headers["Authorization"] = token
    response = requests.post(
        endpoint,
        headers=headers,
        json={"query": INTROSPECTION_QUERY},
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    if "errors" in payload:
        raise ValueError(payload["errors"])
    return payload["data"]["__schema"]


def render_markdown(schema: dict[str, Any], section: str, fields: list[str], types: list[str]) -> str:
    lines: list[str] = []
    if section in ("query", "both"):
        lines.append("## Query Roots")
        for field in schema["queryType"]["fields"]:
            if fields and field["name"] not in fields:
                continue
            lines.append(f"- `{field['name']}` -> `{flatten_type(field['type'])}`")
            for arg in field["args"]:
                lines.append(f"  - `{arg['name']}`: `{flatten_type(arg['type'])}`")
    if section in ("mutation", "both"):
        lines.append("## Mutation Roots")
        for field in schema["mutationType"]["fields"]:
            if fields and field["name"] not in fields:
                continue
            lines.append(f"- `{field['name']}` -> `{flatten_type(field['type'])}`")
            for arg in field["args"]:
                lines.append(f"  - `{arg['name']}`: `{flatten_type(arg['type'])}`")
    if types:
        by_name = {item["name"]: item for item in schema["types"]}
        lines.append("## Types")
        for name in types:
            item = by_name.get(name)
            if not item:
                lines.append(f"- `{name}`: missing")
                continue
            lines.append(f"- `{name}`")
            for field in item.get("fields") or []:
                lines.append(f"  - `{field['name']}`: `{flatten_type(field['type'])}`")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    schema = fetch_schema(args.endpoint, args.auth_header)
    if args.format == "json":
        print(json.dumps(schema, indent=2, sort_keys=True))
        return 0
    print(render_markdown(schema, args.section, args.field, args.types))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
