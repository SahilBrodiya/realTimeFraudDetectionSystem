from flask import Flask, request, jsonify
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import pandas as pd

app = Flask(__name__)

# Initialising Spark Session
spark = SparkSession.builder \
        .appName("FraudDetectionAPI") \
        .getOrCreate()

# loading built model
model_path = "../models/rf_model"
loaded_model = RandomForestClassificationModel.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    """
    process incoming data, pass to model and get output
    :parameter: transaction(dict)
    :return: output of the model
    """

    # Get the incoming transaction data from the request
    transaction_data = request.json['transaction']
    print(type(transaction_data))
    # Assuming transaction_data is dictionary, converting to Pandas DataFrame
    df = pd.DataFrame([transaction_data])
    print("Here")
    # converting pandas df to spark df
    spark_df = spark.createDataFrame(df)

    # Getting feature column from df
    feature_cols = [col for col in spark_df.columns if col != "Class"]

    # Assembler work
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="ScaledFeatures")
    assembled_df = assembler.transform(spark_df)

    # Making Predictions
    predictions = loaded_model.transform(assembled_df)

    result = predictions.select('prediction').first()[0]

    return jsonify({'is_fraud': bool(result)})

if __name__ == "__main__":
    app.run(port=5000)