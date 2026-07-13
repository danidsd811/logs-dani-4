#!/bin/bash
# Stop hook: reminds to run /code-review when .py or .jsx files are staged for commit
if git diff --cached --name-only 2>/dev/null | grep -qE '\.(py|jsx)$'; then
  echo '{"systemMessage":"💡 Hay cambios .py/.jsx en staging — considera ejecutar /code-review medium antes de hacer commit"}'
fi
