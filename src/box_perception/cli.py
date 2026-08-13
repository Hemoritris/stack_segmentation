"""命令行入口（骨架）。"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="box-perception",
        description="箱体 4DoF 位姿估计与垛堆建模",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.parse_args(argv)
    print("box-perception CLI 当前为骨架入口，实际流程请使用 scripts/ 下的脚本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

