# novelai-mcp

NovelAI 画像生成のための汎用 stdio MCP サーバー。

## 機能

- **generate_image**: NovelAI API で画像を生成し、ローカルに保存。パラメータは JSON ファイルで指定
- **generate_image_v5**: NovelAI Diffusion V5専用。日本語自然文・テキスト描画・透過画像に対応（Vibe Transfer/ControlNetは非対応）
- **cleanup_old_image_files**: 生成した古い画像・JSON ファイルを保持期間ベースで削除

## 環境変数

| 変数名 | 必須 | 説明 |
|--------|------|------|
| `NOVELAI_API_KEY` | Yes | NovelAI API キー（novelai-sdk が自動で読み取る） |
| `NOVELAI_OUTPUT_DIR` | Yes | 画像出力先ディレクトリ |

## セットアップ

```bash
uv sync
```

## MCP サーバーとして使用

```json
{
  "mcpServers": {
    "novelai": {
      "command": "uv",
      "args": ["run", "--from", "git+https://github.com/syou6162/novelai-mcp", "novelai-mcp"],
      "env": {
        "NOVELAI_API_KEY": "your-api-key",
        "NOVELAI_OUTPUT_DIR": "/path/to/output"
      }
    }
  }
}
```

## 開発

```bash
# リント
uv run ruff check .

# テスト
uv run pytest tests/

# 型チェック
uv run ty check src/
```
