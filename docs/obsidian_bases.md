# Obsidian Bases “.base” File Format — Developer Guide (Sep 2025)

> Working notes for engineers using the new **Bases** core plugin. This guide consolidates current behavior, community conventions, and a practical schema you can adopt for validation and CI. Treat the JSON Schema below as a **strict, forward‑compatible** profile; it’s intentionally conservative so your `.base` files won’t break as Bases evolves.

---

## 0) TL;DR — What a `.base` File Is

* A `.base` file is a **declarative definition** of a database‑style view over Markdown notes in your vault.
* It references notes via **filters** on file location and **YAML frontmatter properties** (a.k.a. Properties).
* It defines **views** (e.g., `table`, `card`) with **columns** (property bindings, computed display), **sort**, and **group** instructions.
* It is UTF‑8 encoded, plain‑text, typically **JSON** or **YAML**. This guide standardizes on **JSON** for strictness and tooling, with a YAML equivalent shown where useful.

**Critical for passing validation:**

1. **Stable IDs** for the Base and for each View/Column (slug case) — must be unique within the file.
2. **View `type`** must be one of the currently supported types (e.g., `table`, `card`).
3. **Filters** must resolve to valid properties or file fields; unknown properties should be allowed but **warned**.
4. **Columns** must have a resolvable `source` (frontmatter key or built‑in file field like `file.name`, `file.path`, `file.ctime`, `file.mtime`).
5. **Sort keys** must reference defined columns or resolvable fields.
6. **Paths** and **glob patterns** must be normalized (no backslashes on macOS/Linux; use forward slashes).
7. **No circular references** in computed columns; expressions are side‑effect free.
8. **Date/time values** are ISO‑8601; booleans are `true/false`; numbers are plain JSON numbers; lists are arrays.
9. **Unknown top‑level keys** are ignored with a linter warning (to keep forward compatibility).

---

## 1) Minimal `.base` Example (JSON)

```json
{
  "$schema": "vault://schemas/obsidian/bases-2025-09.schema.json",
  "id": "reading-list",
  "name": "Reading List",
  "version": 1,
  "description": "All book notes with reading state.",
  "source": {
    "folders": ["Notes/Books"],
    "includeSubfolders": true,
    "filters": [
      { "property": "type", "op": "eq", "value": "book" },
      { "property": "status", "op": "in", "value": ["queued", "reading"] }
    ]
  },
  "views": [
    {
      "id": "table-main",
      "name": "Table",
      "type": "table",
      "columns": [
        { "id": "title", "header": "Title", "source": "title" },
        { "id": "author", "header": "Author", "source": "author" },
        { "id": "started", "header": "Started", "source": "started", "format": "date" },
        { "id": "state", "header": "Status", "source": "status" }
      ],
      "sort": [ { "by": "started", "dir": "desc", "nulls": "last" } ]
    }
  ]
}
```

---

## 2) Rich `.base` Example with Card View, Groups, and Computed Columns

```json
{
  "$schema": "vault://schemas/obsidian/bases-2025-09.schema.json",
  "id": "projects-dashboard",
  "name": "Projects Dashboard",
  "version": 3,
  "description": "Active projects with owner, health, and next actions",
  "source": {
    "folders": ["Work/Projects"],
    "includeSubfolders": true,
    "filters": [
      { "property": "type", "op": "eq", "value": "project" },
      { "property": "archived", "op": "neq", "value": true }
    ]
  },
  "computed": [
    {
      "id": "healthScore",
      "expr": "clamp((daysSince(lastUpdate) < 7 ? 1 : 0) + (status == 'blocked' ? -1 : 0) + (priority == 'high' ? 1 : 0), -1, 2)",
      "type": "number"
    },
    {
      "id": "cover",
      "expr": "coalesce(coverUrl, banner, thumbnail)",
      "type": "string"
    }
  ],
  "views": [
    {
      "id": "table-active",
      "name": "Active (Table)",
      "type": "table",
      "columns": [
        { "id": "col-title", "header": "Project", "source": "title", "linkTo": "file" },
        { "id": "col-owner", "header": "Owner", "source": "owner" },
        { "id": "col-status", "header": "Status", "source": "status" },
        { "id": "col-updated", "header": "Updated", "source": "file.mtime", "format": "datetime" },
        { "id": "col-health", "header": "Health", "source": "@healthScore", "format": "progress", "min": -1, "max": 2 }
      ],
      "group": { "by": "status", "order": ["blocked", "at-risk", "active", "done"] },
      "sort": [
        { "by": "@healthScore", "dir": "desc" },
        { "by": "file.mtime", "dir": "desc" }
      ],
      "pageSize": 100
    },
    {
      "id": "card-gallery",
      "name": "Gallery (Cards)",
      "type": "card",
      "card": {
        "title": "title",
        "subtitle": "owner",
        "image": "@cover",
        "badges": ["status", "priority"],
        "footer": "nextAction"
      },
      "sort": [ { "by": "priority", "dir": "desc" } ]
    }
  ]
}
```

> **Note**: Computed columns use a small, side‑effect‑free expression language (see §6). They are referenced with an `@` prefix inside views.

---

## 3) YAML Equivalent (if you prefer authoring in YAML)

```yaml
$schema: vault://schemas/obsidian/bases-2025-09.schema.json
id: reading-list
name: Reading List
version: 1
description: All book notes with reading state.
source:
  folders: ["Notes/Books"]
  includeSubfolders: true
  filters:
    - { property: type, op: eq, value: book }
    - { property: status, op: in, value: [queued, reading] }
views:
  - id: table-main
    name: Table
    type: table
    columns:
      - { id: title, header: Title, source: title }
      - { id: author, header: Author, source: author }
      - { id: started, header: Started, source: started, format: date }
      - { id: state, header: Status, source: status }
    sort: [ { by: started, dir: desc, nulls: last } ]
```

---

## 4) Proposed JSON Schema (Draft 2020‑12)

> Use this to **lint** `.base` files in CI. The schema is strict about enums and formats and permissive about extra keys (flagged via `unevaluatedProperties: false` in strict mode, or `true` if you want forward compatibility).

```json
{
  "$id": "vault://schemas/obsidian/bases-2025-09.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Obsidian Bases (.base) — Strict Profile",
  "type": "object",
  "additionalProperties": false,
  "required": ["id", "name", "version", "source", "views"],
  "properties": {
    "$schema": { "type": "string" },
    "id": { "type": "string", "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$" },
    "name": { "type": "string", "minLength": 1 },
    "version": { "type": "integer", "minimum": 1 },
    "description": { "type": "string" },
    "source": {
      "type": "object",
      "additionalProperties": false,
      "required": ["folders"],
      "properties": {
        "folders": {
          "type": "array",
          "minItems": 1,
          "items": { "type": "string", "pattern": "^[^\\\n]+$" }
        },
        "includeSubfolders": { "type": "boolean", "default": true },
        "filters": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["property", "op"],
            "properties": {
              "property": { "type": "string", "minLength": 1 },
              "op": {
                "type": "string",
                "enum": [
                  "eq", "neq", "gt", "gte", "lt", "lte",
                  "in", "nin", "contains", "ncontains", "exists", "nexists",
                  "regex"
                ]
              },
              "value": {}
            }
          }
        }
      }
    },
    "computed": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "expr", "type"],
        "properties": {
          "id": { "type": "string", "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$" },
          "expr": { "type": "string", "minLength": 1 },
          "type": { "type": "string", "enum": ["string", "number", "boolean", "date", "datetime"] }
        }
      }
    },
    "views": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "name", "type"],
        "properties": {
          "id": { "type": "string", "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$" },
          "name": { "type": "string", "minLength": 1 },
          "type": { "type": "string", "enum": ["table", "card"] },
          "columns": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": false,
              "required": ["id", "header", "source"],
              "properties": {
                "id": { "type": "string", "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$" },
                "header": { "type": "string" },
                "source": { "type": "string", "minLength": 1 },
                "format": { "type": "string", "enum": ["text", "number", "date", "datetime", "progress", "badge", "url"] },
                "min": { "type": "number" },
                "max": { "type": "number" },
                "linkTo": { "type": "string", "enum": ["file", "none"] }
              }
            }
          },
          "card": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "title": { "type": "string" },
              "subtitle": { "type": "string" },
              "image": { "type": "string" },
              "badges": { "type": "array", "items": { "type": "string" } },
              "footer": { "type": "string" }
            }
          },
          "group": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "by": { "type": "string" },
              "order": { "type": "array", "items": { "type": "string" } }
            }
          },
          "sort": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": false,
              "required": ["by"],
              "properties": {
                "by": { "type": "string" },
                "dir": { "type": "string", "enum": ["asc", "desc"], "default": "asc" },
                "nulls": { "type": "string", "enum": ["first", "last"], "default": "last" }
              }
            }
          },
          "pageSize": { "type": "integer", "minimum": 1, "maximum": 1000 }
        }
      }
    }
  }
}
```

> Tip: Set your linter to `additionalProperties: false` in CI for strictness, but keep your editor/UX permissive to avoid blocking authors while experimenting.

---

## 5) Validation & Formatting Rules (What Linters Should Enforce)

**Identifiers**

* `id` fields (base, views, computed, columns) must match `^[a-z0-9]+(?:-[a-z0-9]+)*$`.
* All `id` values must be **unique** within the file.

**Paths**

* Use forward slashes, e.g., `Work/Projects`.
* Disallow `..` components to prevent escaping the vault root.

**Data Types**

* Dates: ISO‑8601 `YYYY-MM-DD` or `YYYY-MM-DDTHH:mm:ssZ` (UTC) or with offset.
* Booleans: `true/false` (no `yes/no`).
* Numbers: plain JSON numbers; forbid trailing commas.

**Filters**

* `op` must be one of the allowed enums; `value` is optional for `exists/nexists`.
* `regex` values must compile; recommend RE2‑compatible dialect if you want determinism.

**Columns**

* `source` may be:

  * Frontmatter property (e.g., `status`, `owner`, `started`)
  * Built‑in file field (`file.name`, `file.basename`, `file.path`, `file.ctime`, `file.mtime`, `file.size`)
  * Computed reference starting with `@` to a defined computed id.

**Sort/Group**

* `sort[].by` must reference a valid column `id` or a resolvable field/computed.
* `group.by` should resolve to a discrete/categorical field.

**Card View**

* `image` should resolve to a URL, vault path, or frontmatter string. Prefer HTTPS URLs or relative vault paths.

**Unicode & Encoding**

* Files are UTF‑8, no BOM. Normalize to NFC where possible (macOS HFS+ caveat).

**Forward Compatibility**

* Unknown keys: warn, ignore. Don’t fail builds.
* Keep `version` integer; bump when you add/rename ids or change semantics.

---

## 6) Computed Expression Mini‑Language (Profile)

> Keep expressions **pure** and predictable. Example functions below are a minimal, portable set you can polyfill.

* **Operators**: `+ - * / %`, comparisons `== != > >= < <=`, logical `&& || !`, ternary `cond ? a : b`.
* **Functions** (suggested):

  * `coalesce(a, b, ...)`, `clamp(x, min, max)`
  * `lower(s)`, `upper(s)`, `trim(s)`, `concat(a, b, ...)`
  * `contains(haystack, needle)`, `regexMatch(s, pattern)`
  * `daysSince(date)`, `now()`, `date(format, src)`
  * `len(x)` for arrays/strings
* **Names** resolve in this order: computed ids, frontmatter properties, built‑in `file.*`.
* **No mutation**, no I/O, no vault reads beyond the current note.

**Validation**: your linter should parse expressions and fail if:

* Unknown identifiers are used.
* Functions are unknown or wrong arity.
* Type errors in obvious places (e.g., `daysSince('abc')`).

---

## 7) Editor & CI Integration

* **Prettier‑style JSON**: 2 spaces, LF, keys quoted, stable sort for arrays only when semantic (e.g., `views[]` keep author order; `group.order[]` is semantic).
* **JSON Schema validation** in CI using `ajv` or `djv`. Example `ajv` command:

```bash
ajv validate -s bases-2025-09.schema.json -d "**/*.base" --errors=text --strict=true
```

* **VS Code**: associate `"*.base"` with `jsonc` for comments during authoring, but strip comments on save via formatter step.

* **Git hooks**: pre‑commit to run schema validate + expression parse + path normalization.

---

## 8) Common Failure Modes (and How to Fix Them)

1. **Unknown view type** `kanban` → change to `table`/`card` or move to future feature flag.
2. **Sort references unknown column** → either add the column with matching `id` or sort by a field name (e.g., `file.mtime`).
3. **Regex won’t compile** → validate pattern; prefer anchors `^...$` where appropriate.
4. **Non‑ISO dates** in filters or frontmatter → convert to ISO‑8601.
5. **Duplicate ids** in columns/views → make them unique; ids are used as state keys.
6. **Windows backslashes in paths** → replace with `/`.
7. **Card image from local file not found** → ensure relative path correct or use frontmatter URL.

---

## 9) Migration Tips (Dataview → Bases)

* Convert **queries** into **filters** + view config; push logic into computed expressions when needed.
* Normalize property names (snake or camel; pick one). Add missing frontmatter to legacy notes via a one‑off script.
* Start with a **table view** (easier to reason about), then add a card view for browsing.

---

## 10) Reference: Built‑in Field Names

* `file.name`, `file.basename`, `file.path`, `file.ext`, `file.size`
* `file.ctime`, `file.mtime` (datetimes, ISO)

---

## 11) Authoring Checklist (Pass Formatting & Schema Checks)

* [ ] File encoded UTF‑8 LF, no BOM
* [ ] Top‑level keys: `id`, `name`, `version`, `source`, `views`
* [ ] All ids slug‑cased, unique
* [ ] `source.folders[]` present; no `..` segments
* [ ] Filters use allowed `op` values; regex compiles
* [ ] Each view has valid `type`; columns’ `source` resolvable
* [ ] Sort/group reference valid columns or fields
* [ ] Dates ISO‑8601; booleans literal; numbers numeric
* [ ] Optional: computed expressions parsed and typed
* [ ] CI ajv validation passes; pre‑commit lint clean

---

## 12) Appendix — YAML Frontmatter Conventions for Notes

Keep note frontmatter consistent so Bases can index reliably:

```yaml
---
# Book note example
title: The Pragmatic Programmer
author: Andrew Hunt & David Thomas
type: book
status: reading
started: 2025-08-22
finished: null
coverUrl: https://images.example.com/pragprog.jpg
rating: 4
---
```
