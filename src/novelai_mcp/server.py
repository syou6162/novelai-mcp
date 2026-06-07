"""NovelAI画像生成MCPサーバー"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP
from novelai import NovelAI
from novelai.types import Character, ControlNet, ControlNetImage, GenerateImageParams
from pydantic import ValidationError

from novelai_mcp.cleanup import cleanup_old_files
from novelai_mcp.helpers import load_params_from_file
from novelai_mcp.types import NovelAIParams

logger = logging.getLogger(__name__)

def _get_output_dir() -> Path:
    val = os.environ.get("NOVELAI_OUTPUT_DIR")
    if not val:
        msg = "環境変数 NOVELAI_OUTPUT_DIR が設定されていません"
        raise ValueError(msg)
    return Path(val)

# レート制限対策
_generation_lock = asyncio.Lock()
DEFAULT_WAIT_SECONDS = 5.0

_OVERRIDE_KEYS = (
    "description",
    "prompt",
    "negative_prompt",
    "characters",
    "model",
    "size",
    "steps",
    "scale",
    "n_samples",
    "reference_images",
)

mcp = FastMCP("novelai")


def _build_generation_params(params: NovelAIParams, seed: int) -> GenerateImageParams:
    """NovelAIParamsからGenerateImageParamsを構築"""
    characters = [
        Character(
            prompt=c.prompt,
            negative_prompt=c.negative_prompt,
            enabled=True,
            position=(c.position[0], c.position[1]),
        )
        for c in params.characters
    ]

    controlnet = None
    if params.reference_images:
        controlnet_images = [
            ControlNetImage(
                image=Path(ref.image_path),
                info_extracted=ref.info_extracted,
                strength=ref.strength,
            )
            for ref in params.reference_images
        ]
        controlnet = ControlNet(images=controlnet_images)

    kwargs: dict[str, Any] = {
        "prompt": params.prompt,
        "negative_prompt": params.negative_prompt,
        "model": params.model,
        "size": params.size,
        "steps": params.steps,
        "scale": params.scale,
        "n_samples": 1,
        "seed": seed,
        "characters": characters,
        "controlnet": controlnet,
    }

    return GenerateImageParams(**kwargs)


def _collect_overrides(kwargs: dict[str, Any]) -> dict[str, Any]:
    """None でないキーのみ抽出してオーバーライド辞書を作る"""
    return {k: v for k, v in kwargs.items() if k in _OVERRIDE_KEYS and v is not None}


def _resolve_params(params_file: str | None, kwargs: dict[str, Any]) -> NovelAIParams:
    """params_file と直接引数から NovelAIParams を構築する"""
    overrides = _collect_overrides(kwargs)

    if params_file:
        try:
            params = load_params_from_file(params_file)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONの解析に失敗しました: {e}") from e
        except ValidationError as e:
            raise ValueError(f"パラメータのバリデーションに失敗しました: {e}") from e

        if overrides:
            merged = params.model_dump()
            merged.update(overrides)
            try:
                return NovelAIParams.model_validate(merged)
            except ValidationError as e:
                raise ValueError(f"パラメータのバリデーションに失敗しました: {e}") from e
        return params

    if not overrides:
        raise ValueError("params_file または直接引数のいずれかを指定してください")

    try:
        return NovelAIParams.model_validate(overrides)
    except ValidationError as e:
        raise ValueError(f"パラメータのバリデーションに失敗しました: {e}") from e


async def _run_generation(params: NovelAIParams) -> list[str]:
    """NovelAI API を呼び出して画像を生成し、保存パスのリストを返す"""
    output_dir = _get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=ZoneInfo("Asia/Tokyo"))
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    client = NovelAI()
    generated_paths: list[str] = []

    for i in range(params.n_samples):
        if i > 0:
            logger.info(f"NovelAI レート制限: {DEFAULT_WAIT_SECONDS}秒待機 ({i + 1}/{params.n_samples}枚目)")
        else:
            logger.info(f"NovelAI 画像生成開始: {DEFAULT_WAIT_SECONDS}秒待機 (1/{params.n_samples}枚目)")
        await asyncio.sleep(DEFAULT_WAIT_SECONDS)

        seed = int(time.time_ns() // 1000) % 1_000_000_000
        gen_params = _build_generation_params(params, seed)
        result = client.image.generate(gen_params)
        if not result:
            raise RuntimeError("画像が生成されませんでした")

        output_path = output_dir / f"image_{timestamp}_{i + 1}.png"
        result[0].save(str(output_path))
        generated_paths.append(str(output_path))
        logger.info(f"画像保存: {output_path}")

    # パラメータJSONも保存
    params_output = output_dir / f"params_{timestamp}.json"
    params_output.write_text(params.model_dump_json(indent=2), encoding="utf-8")

    return generated_paths


@mcp.tool()
async def generate_image(  # noqa: PLR0913
    params_file: str | None = None,
    description: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    characters: list[dict[str, Any]] | None = None,
    model: str | None = None,
    size: str | None = None,
    steps: int | None = None,
    scale: float | None = None,
    n_samples: int | None = None,
    reference_images: list[dict[str, Any]] | None = None,
) -> str:
    """NovelAIで画像を生成する。

    params_file（JSONファイルパス）または直接引数でパラメータを指定する。
    両方指定した場合はファイルをベースに直接引数で上書きする（部分修正に使える）。

    Args:
        params_file: パラメータJSONファイルのパス
        description: 画像の内容を日本語で説明（3000字以内）
        prompt: 背景・構図・全体の雰囲気
        negative_prompt: 背景・全体で避けたい要素（必須ワードを含める）
        characters: 人物設定のリスト
        model: 使用モデル（デフォルト: nai-diffusion-4-5-full）
        size: 画像サイズ（デフォルト: portrait）
        steps: ステップ数（1-50、デフォルト: 28）
        scale: スケール（1.0-10.0、デフォルト: 5.0）
        n_samples: 生成枚数（1-4、デフォルト: 2）
        reference_images: 参照画像設定（Vibe Transfer）

    Returns:
        生成した画像ファイルパスのリスト（改行区切り）
    """
    async with _generation_lock:
        params = _resolve_params(params_file, locals())
        generated_paths = await _run_generation(params)
        return f"{len(generated_paths)}枚の画像を生成しました:\n" + "\n".join(f"- {p}" for p in generated_paths)


@mcp.tool()
async def cleanup_old_image_files(
    image_retention_days: int = 7,
    json_retention_days: int = 30,
) -> str:
    """OUTPUT_DIR配下の古い画像・JSONファイルを削除する。

    Args:
        image_retention_days: 画像ファイルの保持日数（デフォルト: 7）
        json_retention_days: JSONファイルの保持日数（デフォルト: 30）

    Returns:
        削除件数のサマリー
    """
    now = datetime.now(tz=ZoneInfo("Asia/Tokyo"))
    deleted_images, deleted_jsons = cleanup_old_files(
        _get_output_dir(),
        image_retention_days=image_retention_days,
        json_retention_days=json_retention_days,
        now=now,
    )
    return f"削除完了: 画像{deleted_images}件, JSON{deleted_jsons}件"
