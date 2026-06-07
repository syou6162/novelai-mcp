"""NovelAI連携ヘルパー関数"""

import json
from pathlib import Path

from novelai_mcp.types import NovelAIParams


def load_params_from_file(params_file: str) -> NovelAIParams:
    """JSONファイルからパラメータを読み込み、バリデーションして返す

    Args:
        params_file: パラメータ JSON ファイルのパス

    Returns:
        バリデーション済みの NovelAIParams

    Raises:
        FileNotFoundError: ファイルが見つからない場合
        json.JSONDecodeError: JSON の解析に失敗した場合
        ValidationError: Pydantic バリデーションに失敗した場合
    """
    path = Path(params_file)
    if not path.exists():
        raise FileNotFoundError(f"パラメータファイルが見つかりません: {params_file}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return NovelAIParams.model_validate(data)
