"""NovelAI ヘルパー関数のテスト"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from novelai_mcp.helpers import load_params_from_file

_NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, missing fingers, extra digits"


def _valid_params_dict() -> dict[str, object]:
    return {
        "description": "テスト画像",
        "prompt": "1girl, solo, garden",
        "negative_prompt": _NEGATIVE_PROMPT,
        "characters": [{"prompt": "blonde hair, blue eyes", "negative_prompt": "bad anatomy"}],
    }


class TestLoadParamsFromFile:
    """load_params_from_file のテスト"""

    def test_valid_json_file(self, tmp_path: Path) -> None:
        """正常なJSONファイルからの読み込み"""
        params_file = tmp_path / "params.json"
        params_file.write_text(json.dumps(_valid_params_dict()), encoding="utf-8")

        result = load_params_from_file(str(params_file))

        assert result.description == "テスト画像"
        assert result.prompt == "1girl, solo, garden"
        assert result.negative_prompt == _NEGATIVE_PROMPT
        assert len(result.characters) == 1
        assert result.characters[0].prompt == "blonde hair, blue eyes"

    def test_file_not_found(self) -> None:
        """ファイルが存在しない場合のFileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="パラメータファイルが見つかりません: /nonexistent/params.json"):
            load_params_from_file("/nonexistent/params.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        """不正なJSONの場合のJSONDecodeError"""
        params_file = tmp_path / "bad.json"
        params_file.write_text("{invalid json}", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_params_from_file(str(params_file))

    def test_validation_error(self, tmp_path: Path) -> None:
        """バリデーションエラーの場合"""
        params_file = tmp_path / "invalid_params.json"
        params_file.write_text(json.dumps({"description": "test"}), encoding="utf-8")

        with pytest.raises(ValidationError, match="prompt"):
            load_params_from_file(str(params_file))
