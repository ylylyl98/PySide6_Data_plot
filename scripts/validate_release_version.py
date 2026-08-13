from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


VERSION_RE = re.compile(r"^(?:v)?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
TAG_RE = re.compile(r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def read_app_version(path: str | Path) -> str:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, str) and VERSION_RE.fullmatch(value):
                        return value.removeprefix("v")
    raise ValueError("app_version.py must define a valid __version__ = \"X.Y.Z\"")


def validate_tag(tag: str, app_version: str) -> None:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError(f"Invalid release tag: {tag!r}")
    tag_version = tag.removeprefix("v")
    if tag_version != app_version:
        raise ValueError(f"Release tag {tag!r} does not match application version {app_version!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-file", default="app_version.py")
    parser.add_argument("--tag")
    parser.add_argument("--print-version", action="store_true")
    args = parser.parse_args()
    version = read_app_version(args.version_file)
    if args.tag:
        validate_tag(args.tag, version)
    if args.print_version or not args.tag:
        print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
