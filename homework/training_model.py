"""
Script para entrenar el modelo de clasificación de default de crédito usando SVC.
"""

import pandas as pd
import numpy as np
import time
import json
import gzip
import pickle
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix
)
from sklearn.compose import ColumnTransformer


def load_data():
    """Carga los datasets de entrenamiento y prueba."""
    train_data = pd.read_csv('../files/input/train_data.csv.zip', compression='zip')
    test_data = pd.read_csv('../files/input/test_data.csv.zip', compression='zip')
    return train_data, test_data


def clean_datasets(df1, df2):
    """Limpia los datasets removiendo valores faltantes y estandarizando categorías."""
    # Renombrar columna
    df1['Age'] = df1['Year'].apply(lambda x: 2021 - x)
    df2['Age'] = df2['Year'].apply(lambda x: 2021 - x)

    df1.drop(columns=['Year', 'Car_Name'], inplace=True)
    df2.drop(columns=['Year', 'Car_Name'], inplace=True)

    return df1, df2


def split_features_target(train_data, test_data):
    """Divide los datasets en características y variable objetivo."""
    x_train = train_data.drop(columns=['Present_Price'])
    y_train = train_data['Present_Price']
    
    x_test = test_data.drop(columns=['Present_Price'])
    y_test = test_data['Present_Price']
    
    return x_train, y_train, x_test, y_test


def create_pipeline(x_train, y_train):
    """Crea y entrena el pipeline de clasificación con optimización de hiperparámetros."""
    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']

    numeric_features = [col for col in x_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop="first", sparse_output=False), categorical_features),
            ('num', MinMaxScaler(), numeric_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),        
        ('feature_selection', SelectKBest(f_regression)),
        ('regressor', LinearRegression())
    ])

    param_grid = {
        "feature_selection__k": range(1, len(x_train.columns) + 1),
        'regressor__fit_intercept': [True, False],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)

    return grid_search


def calculate_metrics(y_true, y_pred):
    """Calcula las métricas de evaluación."""
    # r2, error cuadratico medio, y error absoluto medio


    return {
        'r2': float(np.round(r2_score(y_true, y_pred), 4)),
        'mse': float(np.round(mean_squared_error(y_true, y_pred), 4)),
        'mad': float(np.round(mean_absolute_error(y_true, y_pred), 4)),
    }


def calculate_confusion_matrix(y_true, y_pred):
    """Calcula la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        'true_0': {
            "predicted_0": int(cm[0][0]),
            "predicted_1": int(cm[0][1])
        },
        'true_1': {
            "predicted_0": int(cm[1][0]),
            "predicted_1": int(cm[1][1])
        }
    }


def save_metrics(train_metrics, test_metrics):
    """Guarda las métricas en formato JSON."""
    os.makedirs('../files/output', exist_ok=True)
    
    train_metrics['type'] = 'metrics'
    train_metrics['dataset'] = 'train'
    
    test_metrics['type'] = 'metrics'
    test_metrics['dataset'] = 'test'
    
    with open("../files/output/metrics.json", "w") as f:
        f.write(json.dumps(train_metrics) + '\n')
        f.write(json.dumps(test_metrics) + '\n')


def save_model(pipeline, model_path):
    """Guarda el modelo entrenado."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with gzip.open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


def load_model(model_path):
    """Carga un modelo guardado."""
    with gzip.open(model_path, 'rb') as f:
        return pickle.load(f)


def main():
    """Función principal que ejecuta el entrenamiento completo."""
    # Cargar y limpiar datos
    train_data, test_data = load_data()
    train_data, test_data = clean_datasets(train_data, test_data)
    
    # Dividir en características y objetivo
    x_train, y_train, x_test, y_test = split_features_target(train_data, test_data)
    
    # Verificar si existe modelo guardado
    model_path = Path('../files/models/model.pkl.gz')
    
    if model_path.exists():
        pipeline = load_model(model_path)
    else:
        # Crear y entrenar pipeline
        pipeline = create_pipeline(x_train, y_train)
        
        # Guardar modelo
        save_model(pipeline, model_path)
    
    # Realizar predicciones
    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)
    
    # Calcular métricas
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Guardar métricas
    save_metrics(train_metrics, test_metrics)
    
    return pipeline, train_metrics, test_metrics


if __name__ == "__main__":
    pipeline, train_metrics, test_metrics = main()
