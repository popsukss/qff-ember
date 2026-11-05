"""Fail if the working tree changed files outside a unit's allowed paths.

Usage:
    python scripts/check_scope.py --allow 'ember/data/**' --allow 'tests/test_data.py' \
        [--ignore 'ember/reservoir.py' ...]

Changed = modified, added, deleted or untracked relative to HEAD (git status).
--ignore covers files owned by other units running at the same time.
Exit 0 if every changed file matches an --allow or --ignore glob, else exit 1.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys


def changed_files() -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    files = []
    for line in out.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path.strip('"'))
    return files


def matches(path: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow", action="append", default=[])
    ap.add_argument("--ignore", action="append", default=[])
    args = ap.parse_args()
    files = changed_files()
    bad = [f for f in files if not matches(f, args.allow + args.ignore)]
    mine = [f for f in files if matches(f, args.allow)]
    print(f"changed in scope: {len(mine)}")
    for f in mine:
        print(f"  ok   {f}")
    if bad:
        print(f"OUT OF SCOPE: {len(bad)}")
        for f in bad:
            print(f"  FAIL {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
