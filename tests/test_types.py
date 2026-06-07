"""NovelAI パラメータスキーマのテスト"""

import pytest
from pydantic import ValidationError

from novelai_mcp.types import (
    REQUIRED_NEGATIVE_WORDS,
    CharacterParams,
    NovelAIParams,
    ReferenceImageParams,
)

_NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, missing fingers, extra digits"


class TestReferenceImageParams:
    """ReferenceImageParams のテスト"""

    def test_valid_reference_image(self) -> None:
        """正常な参照画像パラメータ"""
        ref = ReferenceImageParams(image_path="/path/to/image.png")
        assert ref.image_path == "/path/to/image.png"
        assert ref.info_extracted == 0.8
        assert ref.strength == 0.8

    def test_image_path_required(self) -> None:
        """image_pathは必須"""
        with pytest.raises(ValidationError, match="image_path"):
            ReferenceImageParams()  # type: ignore[call-arg]

    def test_image_path_min_length(self) -> None:
        """image_pathは1文字以上"""
        with pytest.raises(ValidationError, match="image_path"):
            ReferenceImageParams(image_path="")

    def test_info_extracted_custom(self) -> None:
        """info_extractedのカスタム値"""
        ref = ReferenceImageParams(image_path="/path/to/image.png", info_extracted=0.5)
        assert ref.info_extracted == 0.5

    def test_info_extracted_min(self) -> None:
        """info_extractedの最小値（0.01未満はエラー）"""
        with pytest.raises(ValidationError, match="info_extracted"):
            ReferenceImageParams(image_path="/path/to/image.png", info_extracted=0.0)

    def test_info_extracted_max(self) -> None:
        """info_extractedの最大値（1.0超過はエラー）"""
        with pytest.raises(ValidationError, match="info_extracted"):
            ReferenceImageParams(image_path="/path/to/image.png", info_extracted=1.1)

    def test_strength_custom(self) -> None:
        """strengthのカスタム値"""
        ref = ReferenceImageParams(image_path="/path/to/image.png", strength=0.6)
        assert ref.strength == 0.6

    def test_strength_min(self) -> None:
        """strengthの最小値（0.01未満はエラー）"""
        with pytest.raises(ValidationError, match="strength"):
            ReferenceImageParams(image_path="/path/to/image.png", strength=0.0)

    def test_strength_max(self) -> None:
        """strengthの最大値（1.0超過はエラー）"""
        with pytest.raises(ValidationError, match="strength"):
            ReferenceImageParams(image_path="/path/to/image.png", strength=1.1)

    def test_extra_fields_forbidden(self) -> None:
        """未知のフィールドは禁止"""
        with pytest.raises(ValidationError, match="(?i)extra"):
            ReferenceImageParams(image_path="/path/to/image.png", unknown_field="value")  # type: ignore[call-arg]


class TestCharacterParams:
    """CharacterParams のテスト"""

    def test_valid_character(self) -> None:
        """正常なキャラクターパラメータ"""
        char = CharacterParams(prompt="blonde hair, blue eyes", negative_prompt="bad anatomy")
        assert char.prompt == "blonde hair, blue eyes"
        assert char.negative_prompt == "bad anatomy"
        assert char.position == [0.5, 0.5]

    def test_prompt_required(self) -> None:
        """promptは必須"""
        with pytest.raises(ValidationError, match="prompt"):
            CharacterParams()  # type: ignore[call-arg]

    def test_prompt_min_length(self) -> None:
        """promptは1文字以上"""
        with pytest.raises(ValidationError, match="prompt"):
            CharacterParams.model_validate({"prompt": "", "negative_prompt": "bad anatomy"})

    def test_position_default(self) -> None:
        """positionのデフォルト値"""
        char = CharacterParams(prompt="test", negative_prompt="bad anatomy")
        assert char.position == [0.5, 0.5]

    def test_position_custom(self) -> None:
        """positionのカスタム値"""
        char = CharacterParams(prompt="test", negative_prompt="bad anatomy", position=[0.3, 0.7])
        assert char.position == [0.3, 0.7]

    def test_position_out_of_range_x(self) -> None:
        """positionのx座標が範囲外"""
        with pytest.raises(ValidationError, match="position"):
            CharacterParams.model_validate({"prompt": "test", "negative_prompt": "bad anatomy", "position": [1.5, 0.5]})

    def test_position_out_of_range_y(self) -> None:
        """positionのy座標が範囲外"""
        with pytest.raises(ValidationError, match="position"):
            CharacterParams.model_validate(
                {"prompt": "test", "negative_prompt": "bad anatomy", "position": [0.5, -0.1]}
            )

    def test_position_wrong_length(self) -> None:
        """positionが2要素でない場合エラー"""
        with pytest.raises(ValidationError, match="position"):
            CharacterParams.model_validate({"prompt": "test", "negative_prompt": "bad anatomy", "position": [0.5]})

    def test_negative_prompt_required(self) -> None:
        """negative_promptは必須"""
        with pytest.raises(ValidationError, match="negative_prompt"):
            CharacterParams.model_validate({"prompt": "test"})

    def test_negative_prompt_min_length(self) -> None:
        """negative_promptは1文字以上"""
        with pytest.raises(ValidationError, match="negative_prompt"):
            CharacterParams.model_validate({"prompt": "test", "negative_prompt": ""})

    def test_extra_fields_forbidden(self) -> None:
        """未知のフィールドは禁止"""
        with pytest.raises(ValidationError, match="(?i)extra"):
            CharacterParams(prompt="test", negative_prompt="bad", unknown="value")  # type: ignore[call-arg]


class TestNovelAIParams:
    """NovelAIParams のテスト"""

    def test_valid_params(self) -> None:
        """正常なパラメータ"""
        params = NovelAIParams(
            description="テスト画像",
            prompt="1girl, solo, garden",
            negative_prompt=_NEGATIVE_PROMPT,
            characters=[CharacterParams(prompt="blonde hair, blue eyes", negative_prompt="bad anatomy")],
        )
        assert params.description == "テスト画像"
        assert params.prompt == "1girl, solo, garden"
        assert params.model == "nai-diffusion-4-5-full"
        assert params.size == "portrait"
        assert params.steps == 28
        assert params.scale == 5.0
        assert params.n_samples == 2
        assert params.reference_images is None

    def test_required_negative_words_missing(self) -> None:
        """必須ネガティブワード不足時のバリデーションエラー"""
        with pytest.raises(ValidationError, match="必須ワードが不足しています"):
            NovelAIParams(
                description="テスト",
                prompt="1girl, solo",
                negative_prompt="lowres",
                characters=[CharacterParams(prompt="blonde hair", negative_prompt="bad anatomy")],
            )

    def test_required_negative_words_all_present(self) -> None:
        """全ての必須ネガティブワードが含まれていればOK"""
        params = NovelAIParams(
            description="テスト",
            prompt="1girl, solo",
            negative_prompt=_NEGATIVE_PROMPT,
            characters=[CharacterParams(prompt="blonde hair", negative_prompt="bad anatomy")],
        )
        for word in REQUIRED_NEGATIVE_WORDS:
            assert word in params.negative_prompt

    def test_characters_required(self) -> None:
        """characters は1つ以上必須"""
        with pytest.raises(ValidationError, match="characters"):
            NovelAIParams(
                description="テスト",
                prompt="1girl, solo",
                negative_prompt=_NEGATIVE_PROMPT,
                characters=[],
            )

    def test_n_samples_range(self) -> None:
        """n_samples は 1-4 の範囲"""
        with pytest.raises(ValidationError, match="n_samples"):
            NovelAIParams(
                description="テスト",
                prompt="1girl, solo",
                negative_prompt=_NEGATIVE_PROMPT,
                characters=[CharacterParams(prompt="blonde hair", negative_prompt="bad anatomy")],
                n_samples=5,
            )

    def test_steps_range(self) -> None:
        """steps は 1-50 の範囲"""
        with pytest.raises(ValidationError, match="steps"):
            NovelAIParams(
                description="テスト",
                prompt="1girl, solo",
                negative_prompt=_NEGATIVE_PROMPT,
                characters=[CharacterParams(prompt="blonde hair", negative_prompt="bad anatomy")],
                steps=51,
            )

    def test_scale_range(self) -> None:
        """scale は 1.0-10.0 の範囲"""
        with pytest.raises(ValidationError, match="scale"):
            NovelAIParams(
                description="テスト",
                prompt="1girl, solo",
                negative_prompt=_NEGATIVE_PROMPT,
                characters=[CharacterParams(prompt="blonde hair", negative_prompt="bad anatomy")],
                scale=0.5,
            )

    def test_extra_fields_forbidden(self) -> None:
        """未知のフィールドは禁止"""
        with pytest.raises(ValidationError, match="(?i)extra"):
            NovelAIParams(
                description="テスト",
                prompt="1girl, solo",
                negative_prompt=_NEGATIVE_PROMPT,
                characters=[CharacterParams(prompt="blonde hair", negative_prompt="bad anatomy")],
                unknown_field="value",  # type: ignore[call-arg]
            )

    def test_with_reference_images(self) -> None:
        """参照画像付きパラメータ"""
        params = NovelAIParams(
            description="テスト",
            prompt="1girl, solo",
            negative_prompt=_NEGATIVE_PROMPT,
            characters=[CharacterParams(prompt="blonde hair", negative_prompt="bad anatomy")],
            reference_images=[ReferenceImageParams(image_path="/path/to/ref.png")],
        )
        assert params.reference_images is not None
        assert len(params.reference_images) == 1
        assert params.reference_images[0].image_path == "/path/to/ref.png"

    def test_description_max_length(self) -> None:
        """description は 3000 字以内"""
        with pytest.raises(ValidationError, match="description"):
            NovelAIParams(
                description="a" * 3001,
                prompt="1girl, solo",
                negative_prompt=_NEGATIVE_PROMPT,
                characters=[CharacterParams(prompt="blonde hair", negative_prompt="bad anatomy")],
            )
