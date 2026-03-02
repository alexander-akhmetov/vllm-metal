#!/usr/bin/env python3
"""Patch transformers rope validation list vs set bug.

Fixed upstream in transformers PR #44272 but not yet in any release.
vLLM passes ignore_keys_at_rope_validation as a list, but transformers
5.2 uses the | operator which requires a set.

Safe to run multiple times — skips if already patched.
"""

import os
import sys


def main() -> int:
    try:
        import transformers
    except ImportError:
        print("transformers not installed, skipping")
        return 0

    path = os.path.join(
        os.path.dirname(transformers.__file__), "modeling_rope_utils.py"
    )
    if not os.path.exists(path):
        print(f"modeling_rope_utils.py not found at {path}, skipping")
        return 0

    with open(path) as f:
        content = f.read()

    old = "set() if ignore_keys_at_rope_validation is None else ignore_keys_at_rope_validation"
    new = "set() if ignore_keys_at_rope_validation is None else set(ignore_keys_at_rope_validation)"

    if old not in content:
        print(f"Already patched or pattern not found in {path}")
        return 0

    with open(path, "w") as f:
        f.write(content.replace(old, new))

    print(f"Patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
