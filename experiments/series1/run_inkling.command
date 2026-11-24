#!/bin/bash
set -euo pipefail
script_directory="$(cd "$(dirname "$0")" && pwd)"
project_directory="$(dirname "$script_directory")"
python_executable="$project_directory/work/validation-env/bin/python"
if ! test -x "$python_executable"; then
    echo "The prepared Python environment is missing from this project."
    echo "See HOW_TO_RUN.txt for the pinned installation instructions."
    exit 1
fi
exec "$python_executable" "$script_directory/ask_inkling.py" "$@"
