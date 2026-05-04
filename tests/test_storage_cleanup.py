import os
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.render_service import PosterRenderer


def test_cleanup_directory_removes_old_and_overflow_files(tmp_path: Path):
    renderer = PosterRenderer()
    renderer.cleanup_max_age_days = 1
    renderer.cleanup_max_files = 2

    target_dir = tmp_path / "cleanup-target"
    target_dir.mkdir(parents=True, exist_ok=True)

    old_file = target_dir / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    # 2 天前，应该被 age 清理
    old_ts = time.time() - 2 * 24 * 3600
    os.utime(old_file, (old_ts, old_ts))

    # 三个新文件，按 count 最终只保留 2 个
    f1 = target_dir / "f1.txt"
    f2 = target_dir / "f2.txt"
    f3 = target_dir / "f3.txt"
    f1.write_text("1", encoding="utf-8")
    time.sleep(0.01)
    f2.write_text("2", encoding="utf-8")
    time.sleep(0.01)
    f3.write_text("3", encoding="utf-8")

    result = renderer.cleanup_directory(str(target_dir))
    assert result["deleted_by_age"] >= 1
    assert result["deleted_by_count"] >= 1

    remain = [p for p in target_dir.iterdir() if p.is_file()]
    assert len(remain) <= 2
