import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    # 引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, help="元データの入力パス")
    parser.add_argument(
        "--test_split_ratio", type=float, help="学習データとテストデータの分割比率"
    )
    parser.add_argument(
        "--training_data_path", type=str, help="前処理した学習データの出力先"
    )
    parser.add_argument(
        "--testing_data_path", type=str, help="前処理したテストデータの出力先"
    )

    args = parser.parse_args()
    return args


def process_data(df):
    df_train, df_test = train_test_split(
        df, test_size=args.test_split_ratio, random_state=0
    )
    mlflow.log_metric("Train samples", len(df_train))
    mlflow.log_metric("Test samples", len(df_test))

    # Date列からMonth列とDay列を追加し、Date列を削除
    df_train["Month"] = pd.to_datetime(df_train["Date"], format="%d-%m-%Y").dt.month
    df_train["Day"] = pd.to_datetime(df_train["Date"], format="%d-%m-%Y").dt.day
    df_train = df_train.drop(columns="Date")
    df_test["Month"] = pd.to_datetime(df_test["Date"], format="%d-%m-%Y").dt.month
    df_test["Day"] = pd.to_datetime(df_test["Date"], format="%d-%m-%Y").dt.day
    df_test = df_test.drop(columns="Date")

    # ターゲット変数となる列名を指定
    col_target = "Weekly_Sales"

    # 分割データの出力
    return df_train, df_test


def main(args):
    # 引数の確認
    lines = [
        f"元データのパス: {args.input_data_path}",
        f"分割データのパス (学習データ): {args.training_data_path}",
        f"分割データのパス (テストデータ): {args.testing_data_path}",
    ]
    [print(line) for line in lines]

    # 学習データの読み込み
    df = pd.read_csv(args.input_data_path)

    # データ前処理
    training_data, testing_data = process_data(df)
    training_data.to_csv(Path(args.training_data_path) / "train.csv", index=False)
    testing_data.to_csv(Path(args.testing_data_path) / "test.csv", index=False)


if __name__ == "__main__":
    # 引数の処理
    args = parse_args()

    # main 関数の実行
    main(args)
