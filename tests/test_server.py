"""NovelAI MCPサーバーのテスト"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from novelai_mcp.types import CleanupResult, GenerateImageResult

_NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, missing fingers, extra digits"


def _valid_params_dict() -> dict[str, object]:
    return {
        "description": "テスト画像",
        "prompt": "1girl, solo, garden",
        "negative_prompt": _NEGATIVE_PROMPT,
        "characters": [{"prompt": "blonde hair, blue eyes", "negative_prompt": "bad anatomy"}],
        "n_samples": 1,
    }


def _valid_v5_params_dict() -> dict[str, object]:
    return {
        "description": "透過立ち絵「こんにちは」",
        "prompt": "日本語の自然文で描かれた庭園",
        "negative_prompt": _NEGATIVE_PROMPT,
        "characters": [{"prompt": "blonde hair, blue eyes", "negative_prompt": "bad anatomy", "position": [0.3, 0.7]}],
        "model": "nai-diffusion-5-full",
        "straight_alpha": True,
        "tag_hint_transparent_background": True,
        "n_samples": 1,
        "meta": {"source": "test"},
    }


class TestGenerateImage:
    """generate_image ツールのテスト"""

    @pytest.mark.asyncio
    async def test_generate_with_params_file(self, tmp_path: Path) -> None:
        """params_fileから画像生成できること（NovelAI APIはモック）"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        params_file = output_dir / "params.json"
        params_file.write_text(json.dumps(_valid_params_dict()), encoding="utf-8")

        mock_image = MagicMock()
        mock_image.save = MagicMock()

        mock_client = MagicMock()
        mock_client.image.generate.return_value = [mock_image]

        with (
            patch("novelai_mcp.server._get_output_dir", return_value=output_dir),
            patch("novelai_mcp.server.NovelAI", return_value=mock_client),
            patch("novelai_mcp.server.DEFAULT_WAIT_SECONDS", 0),
        ):
            from novelai_mcp.server import generate_image

            result = await generate_image(params_file=str(params_file))

        assert isinstance(result, GenerateImageResult)
        assert result.count == 1
        assert len(result.image_paths) == 1
        assert result.params_file == str(params_file)
        mock_client.image.generate.assert_called_once()
        mock_image.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_params_file_not_found(self) -> None:
        """params_fileが見つからない場合のエラー"""
        from novelai_mcp.server import generate_image

        with pytest.raises(ValueError, match="パラメータファイルが見つかりません"):
            await generate_image(params_file="/nonexistent/params.json")

    @pytest.mark.asyncio
    async def test_params_file_invalid_json(self, tmp_path: Path) -> None:
        """params_fileが不正なJSONの場合のエラー"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        params_file = output_dir / "params.json"
        params_file.write_text("not json", encoding="utf-8")

        from novelai_mcp.server import generate_image

        with pytest.raises(ValueError, match="JSONの解析に失敗しました"):
            await generate_image(params_file=str(params_file))

    @pytest.mark.asyncio
    async def test_no_args_raises_error(self) -> None:
        """params_fileが指定されていない場合のエラー"""
        from novelai_mcp.server import generate_image

        with pytest.raises(TypeError):
            await generate_image()


class TestGenerateImageV5:
    """generate_image_v5 ツールのテスト"""

    @pytest.mark.asyncio
    async def test_generate_v5_wires_transparency_flags_and_preserves_params_file(self, tmp_path: Path) -> None:
        """V5の透過フラグをSDKへ渡し、入力JSONを保持する"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        params_file = output_dir / "v5_params.json"
        params_data = _valid_v5_params_dict()
        params_file.write_text(json.dumps(params_data), encoding="utf-8")

        mock_image = MagicMock()
        mock_client = MagicMock()
        mock_client.image.generate.return_value = [mock_image]

        with (
            patch("novelai_mcp.server._get_output_dir", return_value=output_dir),
            patch("novelai_mcp.server.NovelAI", return_value=mock_client),
            patch("novelai_mcp.server.DEFAULT_WAIT_SECONDS", 0),
        ):
            from novelai_mcp.server import generate_image_v5

            result = await generate_image_v5(params_file=str(params_file))

        assert isinstance(result, GenerateImageResult)
        assert result.count == 1
        generated_params = mock_client.image.generate.call_args.args[0]
        assert generated_params.model == "nai-diffusion-5-full"
        assert generated_params.straight_alpha is True
        assert generated_params.tag_hint_transparent_background is True
        assert json.loads(params_file.read_text(encoding="utf-8")) == params_data

    @pytest.mark.asyncio
    async def test_non_v5_params_are_rejected(self, tmp_path: Path) -> None:
        """V5ツールでV5以外のモデルを拒否する"""
        params_file = tmp_path / "params.json"
        data = _valid_v5_params_dict()
        data["model"] = "nai-diffusion-4-5-full"
        params_file.write_text(json.dumps(data), encoding="utf-8")

        from novelai_mcp.server import generate_image_v5

        with pytest.raises(ValueError, match="V5パラメータのバリデーション"):
            await generate_image_v5(params_file=str(params_file))


class TestCleanupOldImageFiles:
    """cleanup_old_image_files ツールのテスト"""

    @pytest.mark.asyncio
    async def test_cleanup_empty_dir(self, tmp_path: Path) -> None:
        """空ディレクトリでのクリーンアップ"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("novelai_mcp.server._get_output_dir", return_value=output_dir):
            from novelai_mcp.server import cleanup_old_image_files

            result = await cleanup_old_image_files()

        assert isinstance(result, CleanupResult)
        assert result.deleted_images == 0
        assert result.deleted_jsons == 0

    @pytest.mark.asyncio
    async def test_cleanup_with_old_files(self, tmp_path: Path) -> None:
        """古いファイルがあるディレクトリでのクリーンアップ"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        old_image = output_dir / "image_20200101_120000_1.png"
        old_image.write_bytes(b"old")

        with patch("novelai_mcp.server._get_output_dir", return_value=output_dir):
            from novelai_mcp.server import cleanup_old_image_files

            result = await cleanup_old_image_files()

        assert isinstance(result, CleanupResult)
        assert result.deleted_images == 1
        assert result.deleted_jsons == 0
        assert not old_image.exists()
