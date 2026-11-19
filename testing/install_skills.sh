#!/usr/bin/env bash
# Copy the testing skills (test-app, language-tutor-conversation-reviewer) into a skill folder.
#   bash testing/install_skills.sh            -> .claude/skills   (Claude Code, this project only)
#   bash testing/install_skills.sh --user     -> ~/.claude/skills (Claude Code, all your projects)
#   bash testing/install_skills.sh <folder>   -> any other folder
# Run it again after you pull changes to testing/skills/.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
case "${1:-}" in
  "")      target="$here/../.claude/skills" ;;
  --user)  target="$HOME/.claude/skills" ;;
  *)       target="$1" ;;
esac
mkdir -p "$target"
for skill in "$here"/skills/*/; do
  name="$(basename "$skill")"
  mkdir -p "$target/$name"
  cp "$skill"SKILL.md "$target/$name/SKILL.md"
  echo "Installed $name -> $target/$name"
done
