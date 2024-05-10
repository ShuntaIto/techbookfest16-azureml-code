import json
import logging
import os
import mlflow
import numpy as np


# 起動時に呼び出される関数
def init():
    global model
    model_path = os.path.join(
        os.environ["AZUREML_MODEL_DIR"],
        "Walmart_store_sales_model",
    )
    model = mlflow.lightgbm.load_model(model_path) # モデルのロード
    logging.info("Init complete")


# リクエストを受け取り、推論結果を返す関数
def run(raw_data):
    logging.info("model: request received")
    data = json.loads(raw_data)["data"]
    data = np.array(data)
    result = model.predict(data) # 推論
    logging.info("Request processed")
    return result.tolist()
