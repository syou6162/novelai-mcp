"""NovelAI パラメータのPydanticスキーマ定義

## プロンプトの使い分けルール

- **全体(prompt/negative_prompt)**: 背景・構図・全体的な雰囲気に関すること
  - 例: "masterpiece, best quality, garden, sunshine, blue sky"
  - 例: "lowres, bad anatomy, bad hands, missing fingers, extra digits"

- **キャラクター(characters[].prompt/negative_prompt)**: 人物の特徴に関すること
  - 例: "1girl, cat ears, red hair, blue eyes"
  - 例: "bad anatomy, bad hands"

全体のプロンプトに人物の特徴を入れないこと。
キャラクターごとに異なるネガティブプロンプトを設定することで、
複数キャラクターの描き分けが可能になる。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# 必須ネガティブワード（全体のnegative_promptに含める）
REQUIRED_NEGATIVE_WORDS: list[str] = [
    "lowres",
    "bad anatomy",
    "bad hands",
    "missing fingers",
    "extra digits",
]


class ReferenceImageParams(BaseModel):
    """参照画像（Vibe Transfer）のパラメータ"""

    image_path: str = Field(
        min_length=1,
        description="参照画像のファイルパス",
    )
    info_extracted: float = Field(
        default=0.8,
        ge=0.01,
        le=1.0,
        description="情報抽出レベル（低い=構図重視、高い=スタイル重視）",
    )
    strength: float = Field(
        default=0.8,
        ge=0.01,
        le=1.0,
        description="参照強度",
    )

    model_config = {"extra": "forbid"}


class CharacterParams(BaseModel):
    """キャラクター設定のパラメータ

    人物の特徴をここに記述する。
    背景に関することは全体のprompt/negative_promptに記述すること。
    """

    prompt: str = Field(
        min_length=1,
        description="人物の特徴（髪色、表情、服装など）",
    )
    negative_prompt: str = Field(
        min_length=1,
        description="人物に関して避けたい要素",
    )
    position: list[float] = Field(default=[0.5, 0.5])

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: list[float]) -> list[float]:
        """positionが2要素かつ0.0〜1.0の範囲内かチェック"""
        if len(v) != 2:
            raise ValueError("positionは2要素のリストで指定してください")
        x, y = v
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("positionは0.0〜1.0の範囲で指定してください")
        return v

    model_config = {"extra": "forbid"}


class NovelAIParams(BaseModel):
    """NovelAI画像生成パラメータ

    ## プロンプトの使い分け

    - prompt/negative_prompt: 背景・構図・全体の雰囲気（人物の特徴は入れない）
    - characters[].prompt/negative_prompt: 人物の特徴（髪色、服装など）
    """

    # 必須フィールド
    description: str = Field(
        min_length=1,
        max_length=3000,
        description="画像の内容を日本語で説明（3000字以内）。生成したい内容を正確に記述してください",
    )
    prompt: str = Field(
        min_length=1,
        description="背景・構図・全体の雰囲気（人物の特徴は入れない）",
    )
    negative_prompt: str = Field(
        description="背景・全体で避けたい要素（必須ワードを含める）。人物の特徴はcharactersに記述",
    )
    characters: list[CharacterParams] = Field(
        min_length=1,
        description="人物設定。人物の特徴・ネガティブプロンプトはここに記述",
    )

    # オプションフィールド（デフォルト値あり）
    model: str = "nai-diffusion-4-5-full"
    size: str = "portrait"
    steps: int = Field(default=28, ge=1, le=50)
    scale: float = Field(default=5.0, ge=1.0, le=10.0)
    n_samples: int = Field(default=2, ge=1, le=4, description="生成する画像の枚数（1-4枚）")
    reference_images: list[ReferenceImageParams] | None = Field(
        default=None,
        description="参照画像設定（Vibe Transfer）。複数指定可",
    )

    @field_validator("negative_prompt")
    @classmethod
    def check_required_negative_words(cls, v: str) -> str:
        """必須ネガティブワードが含まれているかチェック"""
        missing = [w for w in REQUIRED_NEGATIVE_WORDS if w.lower() not in v.lower()]
        if missing:
            raise ValueError(f"必須ワードが不足しています: {missing}")
        return v

    model_config = {"extra": "forbid"}


class GenerateImageResult(BaseModel):
    """generate_image ツールの結果"""

    image_paths: list[str] = Field(description="生成された画像ファイルパスのリスト")
    params_file: str = Field(description="保存されたパラメータJSONファイルのパス")
    count: int = Field(description="生成された画像の枚数")


class CleanupResult(BaseModel):
    """cleanup_old_image_files ツールの結果"""

    deleted_images: int = Field(description="削除された画像ファイルの件数")
    deleted_jsons: int = Field(description="削除されたJSONファイルの件数")
