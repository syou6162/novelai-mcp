"""古い生成画像・パラメータJSONの削除ユーティリティ"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")


def parse_timestamp_from_filename(filename: str) -> datetime | None:
    """ファイル名からYYYYMMDD_HHMMSSタイムスタンプを抽出する

    対応パターン:
    - image_YYYYMMDD_HHMMSS_N.png
    - params_YYYYMMDD_HHMMSS.json

    Returns:
        パースできた場合はJSTのdatetime、できない場合はNone
    """
    parts = Path(filename).stem.split("_")
    if len(parts) < 3:
        return None
    try:
        timestamp_str = f"{parts[1]}_{parts[2]}"
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").replace(tzinfo=JST)
    except (ValueError, IndexError):
        return None


def cleanup_old_files(
    output_dir: Path,
    *,
    image_retention_days: int = 7,
    json_retention_days: int = 30,
    now: datetime,
) -> tuple[int, int]:
    """保持期間を過ぎた古い画像・JSONファイルを削除する

    ファイル名のタイムスタンプ（YYYYMMDD_HHMMSS）を基準に判定する。
    stat()は呼ばない。

    Args:
        output_dir: 画像・JSONファイルが保存されているディレクトリ
        image_retention_days: 画像ファイルの保持日数
        json_retention_days: JSONファイルの保持日数
        now: 現在時刻（削除判定の基準）

    Returns:
        (削除した画像数, 削除したJSON数) のタプル
    """
    if not output_dir.exists():
        return (0, 0)

    image_cutoff = now - timedelta(days=image_retention_days)
    json_cutoff = now - timedelta(days=json_retention_days)

    deleted_images = _delete_old_files_by_name(output_dir, "image_*.png", image_cutoff)
    deleted_jsons = _delete_old_files_by_name(output_dir, "params_*.json", json_cutoff)

    if deleted_images > 0 or deleted_jsons > 0:
        logger.info(f"古いファイルを削除: 画像{deleted_images}件, JSON{deleted_jsons}件")

    return (deleted_images, deleted_jsons)


def _delete_old_files_by_name(output_dir: Path, glob_pattern: str, cutoff: datetime) -> int:
    """指定パターンのファイルのうち、cutoff より古いものを削除する"""
    deleted = 0
    for file_path in output_dir.glob(glob_pattern):
        file_time = parse_timestamp_from_filename(file_path.name)
        if file_time is None:
            continue
        if file_time < cutoff:
            file_path.unlink()
            deleted += 1
    return deleted
