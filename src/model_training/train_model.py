"""
MÓDULO DE TREINAMENTO DE MODELOS
Autor: Adeilson Souza
Contexto: Quarta camada funcional da arquitetura modular de MLOps.

Boas Práticas Aplicadas:
- Separação de hiperparâmetros do código-fonte utilizando o arquivo params.yaml.
- Codificação isolada da variável target para evitar interferência nas features.
- Controle de Overfitting integrado ao ciclo de treino via Early Stopping.
- Versionamento e salvamento estruturado do modelo compilado e do encoder de rótulos.
"""

import json
import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Configuração do Logger para rastreamento do ciclo de treinamento
logger = logging.getLogger("src.model_training.train_model")


def load_data() -> pd.DataFrame:
    """Carrega os dados de treino processados e escalados na camada anterior.

    Returns:
        pd.DataFrame: DataFrame contendo as features prontas para a modelagem.
    """
    train_path = "data/processed/train_processed.csv"
    logger.info(f"Carregando dados processados de {train_path}")
    train_data = pd.read_csv(train_path)
    return train_data


def load_params() -> dict[str, float | int]:
    """Carrega os hiperparâmetros da rede neural a partir do arquivo params.yaml.

    Returns:
        dict[str, int | float]: Dicionário com parâmetros como camadas, épocas e taxa de aprendizado.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]


def prepare_data(train_data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, OneHotEncoder]:
    """Separa as variáveis preditoras do target e aplica codificação na variável resposta.

    Args:
        train_data (pd.DataFrame): Dataset de treino completo.

    Returns:
        tuple[pd.DataFrame, np.ndarray, OneHotEncoder]: Features, labels codificados e o encoder ajustado.
    """
    # Separação entre os atributos previsores e a classe alvo
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    # Codificação em array binário (One-Hot) para adequação à saída da rede neural
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))

    return X_train, y_train_encoded, encoder


def create_model(
    input_shape: int, num_classes: int, params: dict[str, int | float]
) -> tf.keras.Model:
    """Instancia e compila uma arquitetura de rede neural densa sequencial (Keras).

    Args:
        input_shape (int): Quantidade de colunas de entrada (features).
        num_classes (int): Quantidade de neurônios de saída (classes do target).
        params (dict[str, int | float]): Configurações de neurônios, dropout e learning rate.

    Returns:
        tf.keras.Model: Modelo de Deep Learning compilado e pronto para o ajuste.
    """

    # Estruturação da rede com camadas densas intercaladas com Dropout para regularização
    model = Sequential(
        [
            Dense(
                params["hidden_layer_1_neurons"],
                activation="relu",
                input_shape=(input_shape,)
            ),
            Dropout(params["dropout_rate"]),
            Dense(
                params["hidden_layer_2_neurons"],
                activation="relu",
            ),
            Dropout(params["dropout_rate"]),
            Dense(num_classes, activation="softmax"),
        ]
    )

    # Configuração do otimizador Adam injetando a taxa de aprendizado externa
    optimizer = Adam(learning_rate=params["learning_rate"])

    # Compilação utilizando crossentropy categórica por conta do mapeamento em One-Hot
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def save_training_artifacts(model: tf.keras.Model, encoder: OneHotEncoder) -> None:
    """Persiste o modelo de Deep Learning e o transformador de targets nos diretórios locais.

    Args:
        model (tf.keras.Model): Modelo treinado a ser exportado.
        encoder (OneHotEncoder): Objeto do codificador de classes ajustado.
    """
    artifacts_dir = "artifacts"
    models_dir = "models"

    # Criação garantida dos diretórios de saída para evitar quebras em novos ambientes
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.keras")
    encoder_path = os.path.join(artifacts_dir, "[target]_one_hot_encoder.joblib")

    # Exportação do modelo Keras no formato nativo atual (.keras)
    logger.info(f"Salvando modelo treinado em {model_path}")
    model.save(model_path)

    # Exportação do codificador para tradução inversa dos resultados em tempo de inferência
    logger.info(f"Salvando codificador de categorias em: {encoder_path}")
    joblib.dump(encoder, encoder_path)


def train_model(train_data: pd.DataFrame, params: dict[str, int | float]) -> None:
    """Orquestra o ciclo completo de treinamento, geração de logs e salvamento de métricas.

    Args:
        train_data (pd.DataFrame): Base de modelagem.
        params (dict[str, int | float]): Parâmetros de treino.
    """
    # Set up MLflow experiment
    mlflow.set_experiment("ml_classification")

    # Set up Keras autolog
    mlflow.keras.autolog()

    # Configura o MLflow caso estejamos executando um experimento do DVC
    is_experiment = os.getenv("DVC_EXP_NAME") is not None
    extra_args = {}
    if is_experiment:
        runs = mlflow.search_runs(
            experiment_ids=[os.getenv("MLFLOW_EXPERIMENT_ID")],
            filter_string="tags.dvc_exp = 'True'",
            order_by=["start_time DESC"],
        )
        if runs.empty:
            with mlflow.start_run() as parent_run:
                mlflow.set_tag("dvc_exp", True)
                parent_run_id = parent_run.info.run_id
        else:
            parent_run_id = runs.iloc[0].run_id 
        run_name = os.getenv("DVC_EXP_NAME")
        extra_args = {
            "parent_run_id": parent_run_id,
            "run_name": run_name,
            "nested": True,
        }

    with mlflow.start_run(**extra_args):
        # Log parameters to MLflow
        mlflow.log_params(params)

        # Fixação da semente do TensorFlow antes de iniciar o fluxo para garantir reprodutibilidade
        tf.keras.utils.set_random_seed(params.pop("random_seed"))
        
        # Log preprocessing artifacts
        mlflow.log_artifact("artifacts/[features]_mean_imputer.joblib")
        mlflow.log_artifact("artifacts/[features]_scaler.joblib")

        X_train, y_train, encoder = prepare_data(train_data)
        
        model = create_model(
            input_shape=X_train.shape[1], num_classes=y_train.shape[1], params=params
        )

        # Monitoramento do val_loss com tolerância de 10 épocas para interromper o treino se parar de evoluir
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Execução do treinamento utilizando uma divisão de validação interna de 20% dos dados
        logger.info("Training model...")
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[early_stopping],
        )

        save_training_artifacts(model, encoder)

        # Log the encoder
        mlflow.log_artifact("artifacts/[target]_one_hot_encoder.joblib")

        # Extração automática da última época executada para geração do histórico de métricas
        metrics = {
            metric: float(history.history[metric][-1]) 
            for metric in history.history
        }

        # Criação da pasta de métricas e persistência do arquivo JSON de resultados
        metrics_path = "metrics/training.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # # Log metrics to MLflow
        # mlflow.log_metrics(metrics)


def main() -> None:
    """Orquestrador principal do ciclo de vida de treinamento do modelo."""
    train_data = load_data()
    params = load_params()
    train_model(train_data, params)
    logger.info("Pipeline de treinamento de modelo executado com sucesso.")


if __name__ == "__main__":
    main()