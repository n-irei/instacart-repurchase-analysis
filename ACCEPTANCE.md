# 最終受入チェックリスト

PASSは添付証拠で確認した項目、FAILは未達項目です。Kaggleの採点停止をローカル評価で代替してPASSにはしません。

| 要件 | 判定 | 証拠・補足 |
| --- | --- | --- |
| 過去タスクと独立したデータ・環境・モデル | PASS | 新規仮想環境、新規コード、Instacart専用ZIP |
| 承認済み第三者保存版の取得・出所記録 | PASS | data_provenance.json、Version 1、各SHA-256 |
| 元の公式ZIPとの完全一致 | FAIL | 主催者による削除で公式比較対象を取得不可 |
| 欠損値処理の根拠 | PASS | 初回注文だけが欠損、REPORT_JA.md第4節 |
| EDAの妥当性 | PASS | priorのみの集計、母集団を区別、eda_summary.json |
| 複数テーブルのキー・結合監査 | PASS | join_leakage_audit.jsonの全構造チェック |
| prior/train/testの最終注文構造 | PASS | 各ユーザーの最終1注文、連番、明細の所属を全件検査 |
| ユーザー単位group split | PASS | 学習78,725 / 調整26,242 / 最終26,242、4集合の交差0 |
| 未来情報・ラベルリーク防止 | PASS | priorのみの候補・統計、時点全件監査、独立特徴量再計算 |
| mean F1 across orders / None | PASS | metrics.py、境界条件・2,500注文の集合実装比較 |
| 不均衡と閾値選定根拠 | PASS | 正例率9.76%、調整集合のみで378通りを比較 |
| 未使用holdoutによる自己評価 | PASS | validation_summary.json、F1 0.382596、bootstrap区間 |
| メモリ効率化 | PASS | 小さいdtype、集約先行、カテゴリ、分割Parquet、実測メモリ |
| 再現可能なコード・固定依存 | PASS | analysis.py、metrics.py、requirements.txt、README手順 |
| Markdownレポート・分析の限界 | PASS | REPORT_JA.md |
| 検証結果と閾値探索結果 | PASS | validation_results.csv、threshold_search.csv |
| 75,000注文のsubmission.csv | PASS | submission_checks.json、履歴候補・ID完全網羅・重複なし |
| Kaggle実提出とスコア確認 | FAIL | 主催者による競技無効化、提出0件、公式スコアなし |
| GitHub公開・push | PASS | https://github.com/n-irei/instacart-repurchase-analysis、PUBLIC・push確認済み |
| GitHubとローカルの一致確認 | PASS | 匿名cloneとローカルの全32追跡ファイルでSHA-256・内容一致。証拠はローカルPUBLICATION_VERIFICATION.json |

公式取得から第三者保存版への変更、および公式採点が現状利用できない点は、ユーザーのGOを受けて進めています。FAILはその外部制約を残した記録です。
