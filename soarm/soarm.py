#!/usr/bin/env python3
"""兼容旧调用方式的薄壳，等价于 `python -m soarm`。

    python soarm/soarm.py check all
    python -m soarm check all          # 推荐用这种

真正的实现按功能拆在 soarm/ 包的各模块里，见 soarm/__init__.py。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soarm.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
