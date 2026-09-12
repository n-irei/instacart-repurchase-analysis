# Instacart 再購入予測：独立分析とリーク監査

過去に購入した商品のうち、次の注文で再購入される商品を予測します。ユーザーを学習・閾値調整・最終評価に分離した結果、**最終評価の注文平均F1は0.38260**でした。

| モデル | 調整用F1 | 最終評価F1 |
| --- | ---: | ---: |
| 過去購入頻度 | 0.32888 | 0.33095 |
| ロジスティック回帰 | 0.36438 | 0.36167 |
| LightGBM（調整用データで選択） | 0.38321 | 0.38260 |

**Kaggle公式スコアではありません。** 2026-09-12に公式ページで、主催者によるデータ削除とLate Submissionの無効化を確認しました。実提出は0件、公式スコアは未取得です。第三者保存版の利用はユーザー承認済みで、元の公式ZIPとのバイト一致は未確認です。

- [分析レポート](REPORT_JA.md)：欠損、EDA、特徴量、分割、評価、限界
- [実行コード](analysis.py)・[F1実装](metrics.py)・[指標テスト](test_metrics.py)
- [結合・リーク監査](join_leakage_audit.json)・[独立特徴量検証](independent_feature_checks.json)
- [最終評価](validation_results.csv)・[閾値探索378通り](threshold_search.csv)・[選択固定記録](selection_locked.json)
- [提出形式CSV](submission.csv)・[提出形式監査](submission_checks.json)
- [データ出所とSHA-256](data_provenance.json)・[Kaggle状況](kaggle_status.json)
- [受入チェックリスト](ACCEPTANCE.md)

## 再現

Python 3.12で新しい仮想環境を作り、`requirements.txt`をインストールします。過去の分析環境・データ・モデル・特徴量・成果物をコピーしていません。GitHubの既存認証のみ利用します。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s . -p test_metrics.py
.\.venv\Scripts\python.exe analysis.py --data C:\path\to\instacart-csv --private private
.\.venv\Scripts\python.exe verify_features.py --data C:\path\to\instacart-csv --private private
```

入力は`aisles.csv, departments.csv, orders.csv, products.csv, order_products__prior.csv, order_products__train.csv`です。既存`data_provenance.json`と入力ハッシュが異なる場合、処理を停止します。取得元は[psparks保存版 Version 1](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)。`download_data.py`で同版をダウンロードできます。

```powershell
.\.venv\Scripts\python.exe download_data.py --destination private\source
```

この取得コマンドはZIP、展開CSV、取得記録を指定フォルダに保存し、公開済み出所記録は上書きしません。展開先は`private\source\data`です。

Linuxでは`.venv/bin/python`を利用できますが、固定バージョンの配布状況・プラットフォーム差により完全な数値一致を保証するものではありません。今回の実行はWindows / Python 3.12.14。乱数seedは20260912、モデルはCPU6スレッドです。必要メモリの実測値は`runtime_prepare.json`と`runtime_train.json`を参照してください。

## 公開範囲

コード、集計、レポート、図、75,000注文の提出形式CSVを公開します。元データ、ユーザー分割表、個別検証ラベル、特徴量、中間予測、モデルファイル、認証情報は公開しません。ローカルと公開コミットの全追跡ファイルは`verify_publication.py`でSHA-256照合できます。検証結果の`PUBLICATION_VERIFICATION.json`は自己参照を避けるためローカル専用です。
