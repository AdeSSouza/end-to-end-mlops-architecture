"""
MÓDULO DE ENGENHARIA DE RECURSOS (FEATURE ENGINEERING)
Autor: Adeilson Souza
Contexto: Terceira camada funcional da arquitetura modular de MLOps.

Boas Práticas Aplicadas:
- Padronização de escala de recursos para algoritmos sensíveis à escala das variáveis.
- Isolamento estrito de parâmetros estatísticos (Média/Desvio Padrão) para prevenção de Data Leakage.
- Persistência e serialização de transformadores estruturais para pipeline de inferência (Scoring).
"""

import logging
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configuração do Logger corporativo para auditoria do pipeline de features
logger = logging.getLogger("src.feature_engineering.engineer_features")


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega as bases de dados tratadas e pré-processadas da camada anterior.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames de treino e teste tratados.
    """
    train_path = "data/preprocessed/train_preprocessed.csv"
    test_path = "data/preprocessed/test_preprocessed.csv"
    logger.info(f"Carregando dados pré-processados de {train_path} and {test_path}")
    train_preprocessed = pd.read_csv(train_path)
    test_preprocessed = pd.read_csv(test_path)
    return train_preprocessed, test_preprocessed


def engineer_features(
    train_preprocessed: pd.DataFrame, test_preprocessed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Aplica transformações estatísticas e padronização nos conjuntos de dados.

    Garante que a distribuição das variáveis preditoras seja normalizada sem que ocorra
    vazamento de informações estatísticas do conjunto de teste para o modelo.

    Args:
        train_preprocessed (pd.DataFrame): Dataset de treino tratado.
        test_preprocessed (pd.DataFrame): Dataset de teste tratado.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, StandardScaler]: Dados escalados e o transformador ajustado.
    """
    logger.info("Iniciando a transformação e padronização das features preditoras...")

    # Identificação dinâmica das colunas preditoras isolando a variável target
    feature_columns = [col for col in train_preprocessed.columns if col != "target"]

    # Inicialização do transformador para centralização da média em 0 e variância em 1
    scaler = StandardScaler()
    
    # Criação de cópias explícitas para evitar o aviso SettingWithCopyWarning do Pandas
    train_processed = train_preprocessed.copy()
    test_processed = test_preprocessed.copy()

    # Cálculo da média/desvio e transformação aplicados exclusivamente no treino
    train_processed[feature_columns] = scaler.fit_transform(train_processed[feature_columns])

    # Aplicação estrita dos parâmetros estatísticos do treino no conjunto de teste
    test_processed[feature_columns] = scaler.transform(test_processed[feature_columns])

    return train_processed, test_processed, scaler


def save_artifacts(
    train_processed: pd.DataFrame, test_processed: pd.DataFrame, scaler: StandardScaler
) -> None:
    """Persiste os conjuntos de dados finais de modelagem e serializa o objeto scaler.

    Args:
        train_processed (pd.DataFrame): Dados de treino finais e escalados.
        test_processed (pd.DataFrame): Dados de teste finais e escalados.
        scaler (StandardScaler): Objeto do scaler ajustado para reprodução em produção.
    """
   
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    logger.info(f"Salvando bases de dados processadas finais no diretório {output_dir}")

    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path = os.path.join(output_dir, "test_processed.csv")

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)

    # Persistência do scaler para garantir consistência estatística em tempo de predição
    scaler_path = os.path.join("artifacts", "[features]_scaler.joblib")
    logger.info(f"Serializando artefato do Scaler em {scaler_path}")
    joblib.dump(scaler, scaler_path)


def main() -> None:
    """Orquestrador principal da camada de engenharia de transformações do pipeline."""
    train_preprocessed, test_preprocessed = load_preprocessed_data()
    train_processed, test_processed, scaler = engineer_features(train_preprocessed, test_preprocessed)
    save_artifacts(train_processed, test_processed, scaler)
    logger.info("Pipeline de engenharia de recursos executado com sucesso.")


if __name__ == "__main__":
    main()
