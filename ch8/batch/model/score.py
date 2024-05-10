import logging
import os
import mlflow
import pandas as pd


def init():
    global model
    model_path = os.path.join(
        os.environ["AZUREML_MODEL_DIR"],
        "Walmart_store_sales_model",
    )
    model = mlflow.lightgbm.load_model(model_path)
    logging.info("Init complete")


def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")
    results = pd.DataFrame() # 結果を格納する DataFrame
    for input in mini_batch:
        # データの読み込み
        df_batch = pd.read_csv(input)

        # Date列からMonth列とDay列を追加し、Date列を削除
        df_batch["Month"] = pd.to_datetime(df_batch["Date"], format="%d-%m-%Y").dt.month
        df_batch["Day"] = pd.to_datetime(df_batch["Date"], format="%d-%m-%Y").dt.day
        df_batch = df_batch.drop(columns="Date")

        # ターゲット変数となる列名を指定
        col_target = "Weekly_Sales"

        # 学習データと検証データを、特徴量とターゲット変数に分割
        X_batch = df_batch.drop(columns=col_target)
        y_batch = df_batch[col_target].to_numpy().ravel()

        # 予測 
        pred = model.predict(X_batch)

        # 元データへ予測値とファイルパスを追加
        df_batch["input"] = input
        df_batch["pred"] = pred

        results = pd.concat([results, df_batch], ignore_index=True)
    return results