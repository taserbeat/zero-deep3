# DOT 言語で graphviz のグラフを作成

DOT 言語と graphviz を使ってグラフを作成することができる。

# 環境構築

## graphviz のインストール

- MacOS

  ```bash
  brew install graphviz
  ```

- Ubuntu

  ```bash
  apt install graphviz
  ```

## インストールの確認

インストールができていると`dot`コマンドが使える。  
`-V`オプションでバージョンを確認できる。

```bash
dot -V
>> dot - graphviz version 6.0.1 (20220911.1526)
```

# 使い方

`.dot`ファイルを用意し、以下のようなコマンドで画像(png, svg)や PDF(.pdf)に出力可能。

- png

  ```bash
  dot sample.dot -T png -o sample.png
  ```

- svg

  ```bash
  dot sample.dot -T svg -o sample.svg
  ```

- pdf

  ```bash
  dot sample.dot -T pdf -o sample.pdf
  ```
