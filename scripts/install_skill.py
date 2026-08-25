"""Repository convenience wrapper for the packaged skill installer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_wiki_v3.skill_install import main


if __name__ == "__main__":
    raise SystemExit(main())
