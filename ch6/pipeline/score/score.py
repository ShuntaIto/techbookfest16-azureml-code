import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd


def parse_args():
    # 引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input_path", type=str, help="学習済みモデルの入力パス")
    parser.add_argument("--testing_data_path", type=str, help="テストデータのパス")
    parser.add_argument("--predicted_data_path", type=str, help="予測値の出力パス")
    parser.add_argument("--label_data_path", type=str, help="ラベルデータの出力パス")

    args = parser.parse_args()
    return args


def get_model(model_input_path):
    return mlflow.lightgbm.load_model(model_input_path)


def score_model(X_test, model):
    pred = model.predict(X_test)
    return pred


def save_data(pred, data_path, filename):
    np.savetxt(Path(data_path) / filename, pred, delimiter=",")


def main(args):
    # 引数の確認
    lines = [
        f"モデル入力ファイルのパス: {args.model_input_path}",
        f"テストデータのパス: {args.testing_data_path}",
    ]
    [print(line) for line in lines]

    # テストデータの読み込み
    df_test = pd.read_csv(Path(args.testing_data_path) / "test.csv")

    # ターゲット変数となる列名を指定
    col_target = "Weekly_Sales"

    # 学習データと検証データを、特徴量とターゲット変数に分割
    X_test = df_test.drop(columns=col_target)
    y_test = df_test[col_target].to_numpy().ravel()

    # モデルの取得
    model = get_model(args.model_input_path)

    # 予測
    pred = score_model(X_test, model)

    # 予測値の保存
    save_data(pred, args.predicted_data_path, "pred.csv")

    # ラベルデータの保存
    save_data(y_test, args.label_data_path, "label.csv")


if __name__ == "__main__":
    # 引数の処理
    args = parse_args()

    # main 関数の実行
    main(args)
