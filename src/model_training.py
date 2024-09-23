from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from src.data_preparation import prepare_data


def train_model(data):
    """
    train the model on spark processed dataframe
    :param data: spark processed dataframe
    :return: trained Random forest model
    """

    print(data.count())
    # Dividing data into train and test
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

    # Initialising Random Forest Classifier
    rf = RandomForestClassifier(labelCol="Class", featuresCol="ScaledFeatures", numTrees=100)

    # train model on training data
    print("Starting training")
    rf_model = rf.fit(train_data)

    # Make predictions on test_data
    print("making prediction")
    predictions = rf_model.transform(test_data)

    # Evaluate model performace using accuracy metric
    evaluator = MulticlassClassificationEvaluator(labelCol="Class",
                                                  predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    print(f"Accuracy: {accuracy:.4f}")

    # Save model
    model_path = "../models/rf_model"
    rf_model.save(model_path)

if __name__ == "__main__":
    prepared_data = prepare_data("../data/creditcard.csv")
    train_model(prepared_data)