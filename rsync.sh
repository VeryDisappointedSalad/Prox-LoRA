#!/bin/bash

set -euo pipefail

# -r recursive
# -l recreate symlinks
# -p preserve (transfer) all perms
# -t preserve mtimes
# -C omit *~ *.bak *.o *.exe .svn .git
# -c checksum, don't rely on mtimes
# -x don't cross file-system boundaries
# -y try to find fuzzy similar if you have to copy
# -z compress
# --max-size don't transfer (nor delete) files larger than

# Unused:
# -n --dry-run
# -u skip files that are newer on receiver
# --delete delete things non-existent on source
# -E preserve x permissions, other permissions created like cp
# -A preserve acls (assuming compatible)
# -X preserve xattrs

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

direction="$1"
destination="$2"
shift 2

local="$SCRIPT_DIR/"
if [ "$destination" = "entropy" ]; then
    remote="entropy:Prox-LoRA/"
elif [ "$destination" = "industrial" ]; then
    remote="industrial:Prox-LoRA/"
elif [ "$destination" = "stationary" ]; then
    remote="stationary:Prox-LoRA/"
elif [ "$destination" = "jerryrig" ]; then
    remote="jerryrig:Prox-LoRA/"
elif [ "$destination" = "aws" ]; then
    remote="aws:Prox-LoRA/"
else
    echo "Invalid destination. Usage: $0 push|pull entropy|industrial|... --dry-run"
    exit 1
fi

args=(
    -crtlpDxyz \
    --max-size=100M \
    --progress --info=progress2 --stats -hhh \
    --exclude=.mypy_cache --exclude=.pyc_cache --exclude=*__pycache__* --exclude=.pytest_cache --exclude=.ruff_cache \
    --exclude=.env --exclude=.venv --exclude=.git --exclude=tmp --exclude=runs --exclude=data
)
# --delete --max-delete=1

args=(${args[@]} $@)  # Concatenate remaining arguments.

if [ "$direction" == "push" ]
then
    set -o xtrace
    rsync "${args[@]}" "${local%/}"/ "$remote"
elif [ "$direction" == "pull" ]
then
    set -o xtrace
    rsync "${args[@]}" "${remote%/}"/ "$local"
else
    echo "Invalid direction. Usage: $0 push|pull entropy|industrial|... --dry-run"
    exit 1
fi
