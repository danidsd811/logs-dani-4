#!/bin/bash
# PostToolUse/Edit hook: runs Python syntax check on .py files after Claude edits them
f=$(python -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null)
case "$f" in
  *.py)
    if err=$(python -m py_compile "$f" 2>&1); then
      exit 0
    else
      firstline=$(echo "$err" | head -1)
      python -c "import sys,json; print(json.dumps({'systemMessage': '⚠️ Error de sintaxis Python en ' + sys.argv[1] + ': ' + sys.argv[2]}))" "$f" "$firstline"
    fi
    ;;
esac
