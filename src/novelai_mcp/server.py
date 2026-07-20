"""NovelAI画像生成MCPサーバー"""

import asyncio
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
from novelai_mcp.types import CleanupResult, GenerateImageResult, NovelAIParams

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


def _load_params(params_file: str) -> NovelAIParams:
    """params_file から NovelAIParams を構築する"""
    try:
        return load_params_from_file(params_file)
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e
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

    return generated_paths


@mcp.tool()
async def generate_image(params_file: str) -> GenerateImageResult:
    """NovelAIで画像を生成する。

    Args:
        params_file: パラメータJSONファイルのパス

    Returns:
        GenerateImageResult: 生成結果（画像パス・パラメータファイル・枚数）
    """
    async with _generation_lock:
        params = _load_params(params_file)
        generated_paths = await _run_generation(params)
        return GenerateImageResult(
            image_paths=generated_paths,
            params_file=params_file,
            count=len(generated_paths),
        )


@mcp.tool()
async def cleanup_old_image_files(
    image_retention_days: int = 7,
    json_retention_days: int = 30,
) -> CleanupResult:
    """OUTPUT_DIR配下の古い画像・JSONファイルを削除する。

    Args:
        image_retention_days: 画像ファイルの保持日数（デフォルト: 7）
        json_retention_days: JSONファイルの保持日数（デフォルト: 30）

    Returns:
        CleanupResult: 削除件数（画像・JSON）
    """
    now = datetime.now(tz=ZoneInfo("Asia/Tokyo"))
    deleted_images, deleted_jsons = cleanup_old_files(
        _get_output_dir(),
        image_retention_days=image_retention_days,
        json_retention_days=json_retention_days,
        now=now,
    )
    return CleanupResult(
        deleted_images=deleted_images,
        deleted_jsons=deleted_jsons,
    )
