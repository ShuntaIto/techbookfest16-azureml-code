import argparse
from pathlib import Path
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# RMSEを計算する関数
def rmse_score(validation, target):
    return np.sqrt(mean_squared_error(validation, target))


def parse_args():
    # 引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path", type=str, help="学習データの入力パス")
    parser.add_argument("--model_output_path", type=str, help="モデル出力フォルダの出力パス")
    parser.add_argument("--num_leaves", type=int, default=31, help="学習率")
    parser.add_argument(
        "--learning_rate", type=float, default=0.05, help="1本の木の最大葉枚数"
    )

    args = parser.parse_args()
    return args


def save_model(model, output_dir):
    # モデルの保存
    mlflow.lightgbm.save_model(model, output_dir)


def main(args):
    # 自動ロギングの有効化
    run = mlflow.start_run()

    # 自動ロギング有効化
    mlflow.lightgbm.autolog()

    # 引数の確認
    lines = [
        f"学習データのパス: {args.training_data_path}",
        f"モデル出力フォルダのパス: {args.model_output_path}",
    ]
    [print(line) for line in lines]

    # 学習データの読み込み
    df = pd.read_csv(Path(args.training_data_path) / "train.csv")
    df_train, df_valid = train_test_split(df, test_size=0.3, random_state=0)

    # ターゲット変数となる列名を指定
    col_target = "Weekly_Sales"

    # 学習データと検証データを、特徴量とターゲット変数に分割
    X_train = df_train.drop(columns=col_target)
    y_train = df_train[col_target].to_numpy().ravel()
    X_valid = df_valid.drop(columns=col_target)
    y_valid = df_valid[col_target].to_numpy().ravel()

    # LightGBMのデータセットに変換
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    # ハイパーパラメータの設定
    params = {
        "objective": "regression",
        "metric": "rmse_score",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
    }

    # モデル学習
    model = lgb.train(
        params=params, 
        train_set=train_data, 
        num_boost_round=100, 
        valid_sets=valid_data
    )

    # モデル保存
    save_model(model, args.model_output_path)

    # 自動ロギング有効化
    mlflow.end_run()

if __name__ == "__main__":
    # 引数の処理
    args = parse_args()

    # main 関数の実行
    main(args)
