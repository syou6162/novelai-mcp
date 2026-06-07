"""古い画像・JSONファイルのクリーンアップユーティリティのテスト"""

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from novelai_mcp.cleanup import (
    cleanup_old_files,
    parse_timestamp_from_filename,
)

JST = ZoneInfo("Asia/Tokyo")


class TestParseTimestampFromFilename:
    """parse_timestamp_from_filename のテスト"""

    def test_image_filename_returns_datetime(self) -> None:
        """画像ファイル名からdatetimeをパースできる"""
        result = parse_timestamp_from_filename("image_20260315_100000_1.png")
        assert result == datetime(2026, 3, 15, 10, 0, 0, tzinfo=JST)

    def test_params_filename_returns_datetime(self) -> None:
        """paramsファイル名からdatetimeをパースできる"""
        result = parse_timestamp_from_filename("params_20260315_100000.json")
        assert result == datetime(2026, 3, 15, 10, 0, 0, tzinfo=JST)

    def test_invalid_filename_returns_none(self) -> None:
        """パース不能なファイル名はNoneを返す"""
        result = parse_timestamp_from_filename("random_file.png")
        assert result is None

    def test_short_filename_returns_none(self) -> None:
        """アンダースコア区切りが少ないファイル名はNoneを返す"""
        result = parse_timestamp_from_filename("image.png")
        assert result is None


class TestCleanupOldFiles:
    """cleanup_old_files のテスト"""

    def test_nonexistent_directory_returns_zeros(self, tmp_path: Path) -> None:
        """存在しないディレクトリの場合 (0, 0) を返す"""
        now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=JST)
        result = cleanup_old_files(tmp_path / "nonexistent", now=now)
        assert result == (0, 0)

    def test_empty_directory_returns_zeros(self, tmp_path: Path) -> None:
        """空ディレクトリの場合 (0, 0) を返す"""
        now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=JST)
        result = cleanup_old_files(tmp_path, now=now)
        assert result == (0, 0)

    def test_deletes_old_images_keeps_recent(self, tmp_path: Path) -> None:
        """8日前の画像は削除、5日前の画像は残る → (1, 0)"""
        now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=JST)
        old_image = tmp_path / "image_20260307_120000_1.png"
        old_image.write_bytes(b"old")
        recent_image = tmp_path / "image_20260310_120000_1.png"
        recent_image.write_bytes(b"recent")

        result = cleanup_old_files(tmp_path, image_retention_days=7, json_retention_days=30, now=now)

        assert result == (1, 0)
        assert not old_image.exists()
        assert recent_image.exists()

    def test_deletes_old_json_keeps_recent(self, tmp_path: Path) -> None:
        """35日前のJSONは削除、25日前のJSONは残る → (0, 1)"""
        now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=JST)
        old_json = tmp_path / "params_20260209_120000.json"
        old_json.write_text("{}", encoding="utf-8")
        recent_json = tmp_path / "params_20260219_120000.json"
        recent_json.write_text("{}", encoding="utf-8")

        result = cleanup_old_files(tmp_path, image_retention_days=7, json_retention_days=30, now=now)

        assert result == (0, 1)
        assert not old_json.exists()
        assert recent_json.exists()

    def test_different_retention_days_for_image_and_json(self, tmp_path: Path) -> None:
        """10日前のファイル: 画像は削除（7日保持）、JSONは残る（30日保持） → (1, 0)"""
        now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=JST)
        image = tmp_path / "image_20260305_120000_1.png"
        image.write_bytes(b"img")
        json_file = tmp_path / "params_20260305_120000.json"
        json_file.write_text("{}", encoding="utf-8")

        result = cleanup_old_files(tmp_path, image_retention_days=7, json_retention_days=30, now=now)

        assert result == (1, 0)
        assert not image.exists()
        assert json_file.exists()

    def test_unparseable_filename_skipped(self, tmp_path: Path) -> None:
        """パース不能なファイル名はスキップされ削除されない"""
        now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=JST)
        weird_image = tmp_path / "image_not_a_timestamp.png"
        weird_image.write_bytes(b"weird")

        result = cleanup_old_files(tmp_path, now=now)

        assert result == (0, 0)
        assert weird_image.exists()

    def test_logs_when_files_deleted(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """削除時にINFOログが出力される"""
        now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=JST)
        old_image = tmp_path / "image_20260307_120000_1.png"
        old_image.write_bytes(b"old")

        with caplog.at_level(logging.INFO, logger="novelai_mcp.cleanup"):
            cleanup_old_files(tmp_path, image_retention_days=7, json_retention_days=30, now=now)

        assert caplog.messages == ["古いファイルを削除: 画像1件, JSON0件"]

    def test_no_log_when_nothing_deleted(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """削除なしの場合ログが出ない"""
        now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=JST)
        recent_image = tmp_path / "image_20260314_120000_1.png"
        recent_image.write_bytes(b"recent")

        with caplog.at_level(logging.INFO, logger="novelai_mcp.cleanup"):
            cleanup_old_files(tmp_path, image_retention_days=7, json_retention_days=30, now=now)

        assert caplog.messages == []
