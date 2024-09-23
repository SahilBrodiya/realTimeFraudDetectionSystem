from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

def start_streaming():
    """
    Start a streaming service that reads transaction data from a kafka topic
    The service processes incoming transactions for fraud detection

    Attention: have kafka running with a 'transaction' topic created
    """

    spark = SparkSession.builder \
            .appName("FraudDetectionStreaming") \
            .getOrCreate()

    # load model
    model_path = "../models/rf_model"
    loaded_model = RandomForestClassificationModel.load(model_path)

    # read kafka topic
    kafkaStreamDf = spark.readStream \
                    .format("kafka") \
                    .option("kafka.bootstrap.servers", "localhost:9202") \
                    .option("subscribe", "transactions") \
                    .load()

    # schema for incoming json transactions
    json_schema = schema_of_json(
        '{"Time": 0.0, "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0, "V6": 0.0, "V7": 0.0, '
        '"V8": 0.0, "V9": 0.0, "V10": 0.0, "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0, '
        '"V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0, "V21": 0.0, "V22": 0.0, "V23": 0.0, '
        '"V24": 0.0, "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0, "Amount": 0.0, "Class": 0}')

    # convert value column from kafka to json format and extract fields based on schema
    transactions_df = kafkaStreamDf.selectExpr("CAST(value as String as json") \
                    .select(from_json(col("json"), json_schema).alias("data")) \
                    .select("data.*")

    # Assemble features into a single vector column for prediction
    feature_columns = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                       "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
                       "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="scaledFeatures")

    # Transform the incoming transaction data to include the features vector
    transaction_with_features = transactions_df.transform(assembler)

    # Make predictions on incoming transaction data using the loaded model
    predictions_df = loaded_model.transform(transaction_with_features)

    # Select relevant columns for output (including prediction)
    output_df = predictions_df.select("data.*", "prediction")

    # write prediction to console or another sink(eg. database)
    query = output_df.writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .start()

    query.awaitTermination()

if __name__ == "__main__":
    start_streaming()