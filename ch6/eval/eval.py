import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


# RMSEを計算する関数
def rmse_score(validation, target):
    return np.sqrt(mean_squared_error(validation, target))


def parse_args():
    # 引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_data_path", type=str, help="予測値の入力パス")
    parser.add_argument("--label_data_path", type=str, help="ラベルデータの入力パス")
    parser.add_argument(
        "--model_performance_report_path",
        type=str,
        help="モデルパフォーマンスレポートの出力パス",
    )

    args = parser.parse_args()
    return args


def evaluate_model(y_test, y_pred):
    # データのサンプル数のロギング
    mlflow.log_metric("テストデータのサンプル数", len(y_test))

    # モデル評価
    rmse = rmse_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 精度メトリックのロギング
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)


def plot_actuals_predictions(y_test, y_pred, report_path):
    # 出力パス
    output_path = str(Path(report_path) / "actuals_vs_predictions.png")

    # 実測値と予測値のプロット
    plt.figure(figsize=(10, 7))
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, color="r")
    plt.title("Actual VS Predicted Values (Test set)")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.savefig(output_path)

    # プロット画像のロギング
    mlflow.log_artifact(output_path)


def main(args):
    # 引数の確認
    lines = [
        f"予測値データのパス: {args.predicted_data_path}",
        f"ラベルデータのパス: {args.label_data_path}",
        f"モデルパフォーマンスレポートのパス: {args.model_performance_report_path}",
    ]
    [print(line) for line in lines]

    # 予測値データの読み込み
    y_pred = pd.read_csv(Path(args.predicted_data_path) / "pred.csv")

    # ラベルデータの読み込み
    y_test = pd.read_csv(Path(args.label_data_path) / "label.csv")

    # モデル評価指標の算出
    evaluate_model(y_test, y_pred)

    # ラベルと予測値のプロット
    plot_actuals_predictions(y_test, y_pred, args.model_performance_report_path)


if __name__ == "__main__":
    # 引数の処理
    args = parse_args()

    # main 関数の実行
    main(args)
