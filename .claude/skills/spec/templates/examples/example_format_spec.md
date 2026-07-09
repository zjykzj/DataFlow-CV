# Data Format Specification Example

> **Version:** v1.0 | **Last Updated:** 2026-07-07

## Overview

This example demonstrates how to write a data format specification. It defines the structure, required fields, and validation rules for a hypothetical JSON-based configuration file.

## Scope

**In scope:**
- JSON structure and required fields
- Data types and validation rules
- Edge case handling

**Out of scope:**
- How the configuration is loaded (see `spec_config_loader.md`)
- UI for editing configuration

## File Structure

```json
{
  "version": "1.0",
  "name": "string",
  "enabled": true,
  "settings": {
    "timeout": 30,
    "retries": 3
  }
}
```

### Required Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `version` | string | Semver format | Configuration schema version |
| `name` | string | 1-100 chars | Human-readable name |
| `enabled` | boolean | - | Whether feature is active |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `settings` | object | `{}` | Feature-specific settings |

## Edge Cases & Constraints

| Scenario | Behavior | Rationale |
|----------|----------|-----------|
| Missing `version` | Reject with error | Cannot determine schema compatibility |
| `timeout < 0` | Clamp to 0 | Negative timeout makes no sense |
| Unknown fields | Warn, continue | Forward compatibility |

## References

- Related specs: `spec_config_loader.md`, `spec_validation.md`
- External standards: [JSON Schema Draft 7](https://json-schema.org/draft-07/schema)
