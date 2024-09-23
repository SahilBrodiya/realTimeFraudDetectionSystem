from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler

def prepare_data(file_path):
    """
    Load and processes data at file path. Here credit information
    :param file_path:
    :return: a dataframe precossed using spark ready for model training
    """

    # Initialise spark session
    spark = SparkSession.builder \
            .appName("FraudDetection") \
            .getOrCreate()

    # load dataset into spark DataFrame
    data = spark.read.csv(file_path, header=True, inferSchema=True)

    # Assemble features into a single vector column. Basically taking all column except last
    # which is target. so age | income| gender will be column names features having value
    # [age, income, gender]
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol = "features")
    data = assembler.transform(data)

    # Scale features to have zero mean and unit variance

    scaler = StandardScaler(inputCol="features", outputCol="ScaledFeatures", withMean=True, withStd=True)
    scalerModel = scaler.fit(data)
    data = scalerModel.transform((data))

    return data

if __name__ == "__main__":
    prepared_data = prepare_data("../data/creditcard.csv")
    prepared_data.show(5) # display 5 rows of data