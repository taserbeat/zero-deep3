# ゼロから作るディープラーニング 3

書籍を勉強する際に書いたコードを保管する。

# 動作環境

| 名称   | バージョン          | 備考 |
| ------ | ------------------- | ---- |
| Python | 3.8.x or later      |      |
| pipenv | 2020.11.15 or later |      |

# 環境構築

```bash
bash setup.sh
```

# ユニットテスト

## 単一ファイルのテストを実行する場合

```bash
pipenv run python -m unittest steps/step10.py
```

## 特定ディレクトリ配下の複数ファイルでテストを実行する場合

テスト対象のディレクトリが`./tests/`の場合

```bash
pipenv run python -m unittest discover tests
```
