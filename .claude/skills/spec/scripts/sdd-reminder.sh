#!/bin/bash
# SDD methodology reminder — invoked by PreToolUse hook on specs/ file edits.
# Reads the tool input JSON from stdin, checks if the file path contains "specs/",
# and if so, outputs a hook payload that injects the /spec reminder into Claude's context.
#
# Requirements: python3 (for JSON parsing from stdin)
#
# Install: see CONFIGURATION.md §4 for the hook configuration to add to .claude/settings.json

# Extract file_path from the harness's stdin JSON payload
file_path=$(python3 -c "
import json, sys
payload = json.load(sys.stdin)
tool_input = payload.get('tool_input') or {}
print(tool_input.get('file_path', ''))
" 2>/dev/null)

# Only trigger for paths containing "specs/"
if echo "$file_path" | grep -q 'specs/'; then
  cat <<'JSON'
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "additionalContext": "You are about to modify a specs/ file. Follow the /spec skill SDD methodology: spec-first workflow, version bump per change type, classification rules. Invoke the spec skill first if it is not already loaded in this session."
  },
  "systemMessage": "specs/ edit detected — /spec SDD methodology applies"
}
JSON
fi
