# Real-Time Fraud Detection System

## Overview

The **Real-Time Fraud Detection System** is designed to analyze credit card transactions in real-time to identify potentially fraudulent activities. This system utilizes Apache Spark for big data processing, Kafka for real-time data streaming, and Flask for building a RESTful API. The project demonstrates how to build a scalable and efficient fraud detection solution that can handle large volumes of transaction data.

## Features

- Real-time transaction processing using Kafka.
- Machine learning model trained with Apache Spark MLlib.
- RESTful API built with Flask for receiving transaction data and providing predictions.
- Scalable architecture that can handle high throughput.

## Technologies Used

- **Apache Spark**: For distributed data processing and machine learning.
- **Kafka**: For real-time data streaming.
- **Flask**: For building the web API.
- **Python**: Programming language used for development.
- **Docker**: For containerization of the application.

## Project Structure
fraud-detection/
├── data/
│   └── creditcard.csv                  # Dataset for training and testing
├── notebooks/
│   └── eda.ipynb                       # Exploratory Data Analysis (optional)
│   └── model_training.ipynb            # Model training notebook (optional)
├── src/
│   ├── data_preparation.py              # Data preparation script
│   ├── model_training.py                 # Model training script
│   ├── streaming_service.py              # Real-time streaming service script
│   └── app.py                           # Flask API for predictions
├── models/                               # Directory to save the trained model
│   └── random_forest_model               # Trained model directory
├── requirements.txt                     # Python dependencies
└── Dockerfile                           # Dockerfile for containerization

