import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import lightgbm as lgb
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def main():
    """メイン関数"""

    # パラメータ
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_leaves", type=int, default=31, help="学習率")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="1本の木の最大葉枚数")
    parser.add_argument("--registered_model_name", type=str, help="登録するモデル名")
    #parser.add_argument("--train_data_name", type=str, help="学習データアセット名")
    #parser.add_argument("--train_data_version", type=int, help="学習データアセットバージョン", default=1)
    #parser.add_argument("--valid_data_name", type=str, help="検証データアセット名")
    #parser.add_argument("--valid_data_version", type=int, help="検証データアセットバージョン", default=1)
    parser.add_argument("--train_data_path", type=str, help="学習データアセット名")
    parser.add_argument("--valid_data_path", type=str, help="検証データアセット名")

    args = parser.parse_args()

    # AzureMLワークスペースへの接続
    credential = DefaultAzureCredential(exclude_workload_identity_credential=True)
    ml_client = MLClient(
        credential=credential,
        subscription_id="27a05524-7a59-453b-88d7-df3ebaa2bcc1",
        resource_group_name="azureml-book",
        workspace_name="azureml-book",
    )

    # ロギング開始
    mlflow.start_run()

    # 自動ロギング有効化
    mlflow.lightgbm.autolog()

    ###################
    #<データ準備>
    ###################

    # 学習データと検証データの読み込み
    # data_asset = ml_client.data.get(name=args.train_data_name, version=args.train_data_version)
    #df_train = pd.read_csv(data_asset.path)
    # data_asset = ml_client.data.get(name=args.valid_data_name, version=args.valid_data_version)
    #df_valid = pd.read_csv(data_asset.path)
    df_train = pd.read_csv(args.train_data_path)
    df_valid = pd.read_csv(args.valid_data_path)

    # Date列からMonth列とDay列を追加し、Date列を削除
    df_train['Month'] = pd.to_datetime(df_train['Date']).dt.month
    df_train['Day'] = pd.to_datetime(df_train['Date']).dt.day
    df_train = df_train.drop(columns='Date')
    df_valid['Month'] = pd.to_datetime(df_valid['Date']).dt.month
    df_valid['Day'] = pd.to_datetime(df_valid['Date']).dt.day
    df_valid = df_valid.drop(columns='Date')
    
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

    ####################
    #</データ準備>
    ####################

    ##################
    #<学習>
    ##################
    # ハイパーパラメータの設定
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate
    }

    # モデルの学習
    model = lgb.train(params=params, train_set=train_data,
                        num_boost_round=100, valid_sets=valid_data)


    ###################
    #</学習>
    ###################

    ##########################
    #<モデル登録>
    ##########################
    # 学習済みモデルをAzureMLワークスペースへ登録
    mlflow.lightgbm.log_model(
        lgb_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name
    )

    ###########################
    #</モデル登録>
    ###########################

    # ロギング停止
    mlflow.end_run()

if __name__ == "__main__":
    main()
