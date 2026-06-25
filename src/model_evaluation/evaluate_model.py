"""
MÓDULO DE AVALIAÇÃO DE MODELOS
Autor: Adeilson Souza
Contexto: Quinta camada funcional da arquitetura modular do pipeline local de MLOps.

Boas Práticas Aplicadas:
- Isolamento estrito da base de teste para validação de performance com dados inéditos.
- Consumo de artefatos serializados (modelo e codificador) simulando ambiente de produção.
- Extração automática de métricas estatísticas consolidadas (Matriz de Confusão e Classification Report).
- Persistência de resultados em formato estruturado (JSON) para auditoria e governança.
"""

import logging
import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Configuração do Logger para rastreabilidade do processo de avaliação
logger = logging.getLogger("src.model_evaluation.evaluate_model")


def load_model() -> tf.keras.Model:
    """Carrega o modelo de Deep Learning previamente treinado e salvo no disco.

    Returns:
        tf.keras.Model: Objeto do modelo Keras carregado e pronto para inferência.
    """
    model_path = "models/model.keras"
    # Recuperação do arquivo nativo do Keras contendo a arquitetura e pesos treinados
    model = tf.keras.models.load_model(model_path)
    return model


def load_encoder() -> LabelEncoder:
    """Carrega o codificador de categorias a partir dos artefatos serializados.

    Returns:
        LabelEncoder: Objeto do codificador carregado para mapeamento de classes.
    """
    encoder_path = "artifacts/[target]_one_hot_encoder.joblib"
    # Recuperação do objeto binário para permitir a interpretação correta dos alvos
    encoder = joblib.load(encoder_path)
    return encoder


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carrega o conjunto de dados de teste final que foi isolado e escalado.

    Returns:
        tuple[pd.DataFrame, pd.Series]: DataFrame de features (X) e a série com os rótulos reais (y).
    """
    data_path = "data/processed/test_processed.csv"
    logger.info(f"Carregando dados de teste de {data_path}")
     # Leitura da base de teste processada de forma idêntica à base de treinamento
    data = pd.read_csv(data_path)
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y


def evaluate_model(
    model: tf.keras.Model, encoder: LabelEncoder, X: pd.DataFrame, y_true: pd.Series
) -> None:
    """Calcula e exporta as métricas de performance do modelo sobre a base de teste.

    Args:
        model (tf.keras.Model): Modelo treinado carregado.
        encoder (LabelEncoder): Objeto do codificador de classes.
        X (pd.DataFrame): Matriz de variáveis preditoras de teste.
        y_true (pd.Series): Série contendo os rótulos reais de teste.
    """
    # Set up MLflow experiment
    mlflow.set_experiment("ml_classification")

    # Get run_id for latest MLflow run
    runs = mlflow.search_runs(
	    experiment_ids=[os.getenv("MLFLOW_EXPERIMENT_ID")], order_by=["start_time DESC"]
    )
    run_id = runs.iloc[0].run_id

    with mlflow.start_run(run_id=run_id):

        # Geração das predições de probabilidade e extração do índice da classe de maior confiança
        y_pred_proba = model.predict(X)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Cálculo dos indicadores estatísticos de classificação e estruturação da matriz de confusão
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred).tolist()
        evaluation = {"classification_report": report, "confusion_matrix": cm}

        # Log metrics (DVC) Registro impresso do relatório técnico de desempenho no fluxo de logs
        logger.info(f"Classification Report:\n{classification_report(y_true, y_pred)}")
        evaluation_path = "metrics/evaluation.json"
        # Salvamento das métricas finais consolidadas em arquivo estruturado para governança
        with open(evaluation_path, "w") as f:
            json.dump(evaluation, f, indent=2)

        # Log metrics (MLflow)
        mlflow.log_metrics(
		    {
                "test_accuracy": report["accuracy"],
                "test_precision_weighted": report["weighted avg"]["precision"],
                "test_recall_weighted": report["weighted avg"]["recall"],
                "test_f1_weighted": report["weighted avg"]["f1-score"],
            }
        )



def main() -> None:
    """Orquestrador principal do ciclo de vida de avaliação e validação do modelo."""
    model = load_model()
    encoder = load_encoder()
    X, y = load_test_data()
    evaluate_model(model, encoder, X, y)
    logger.info("Avaliação do modelo concluída com sucesso.")


if __name__ == "__main__":
    main()