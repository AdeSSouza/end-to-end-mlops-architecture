"""
MÓDULO DE PRÉ-PROCESSAMENTO E TRATAMENTO DE DADOS
Autor: Adeilson Souza
Contexto: Segunda camada funcional da arquitetura modular de MLOps.

Boas Práticas Aplicadas:
- Isolamento estrito de conjuntos de dados para mitigação de Data Leakage (Vazamento de Dados).
- Reprodutibilidade matemática garantida via parametrização externa (params.yaml).
- Serialização e versionamento de artefatos de engenharia para consistência em tempo de inferência.
"""

import logging
import os
import yaml

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Configuração do Logger corporativo para rastreabilidade do pipeline
logger = logging.getLogger("src.data_preprocessing.preprocess_data")


def load_data() -> pd.DataFrame:
    """Carrega a base de dados bruta a partir do diretório local.

    Returns:
        pd.DataFrame: DataFrame contendo os dados brutos de entrada.
    """
    input_path = "data/raw/raw.csv"
    logger.info(f"Carregando dados brutos de: {input_path}")
    data = pd.read_csv(input_path)
    return data


def load_params() -> dict[str, float | int]:
    """Lê os parâmetros globais de pré-processamento definidos no arquivo de configuração params.yaml.

    Returns:
        dict[str, float | int]: Dicionário de hiperparâmetros (ex: test_size, random_seed).
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess_data"]


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide os dados brutos em conjuntos de treino e teste de forma determinística.

    Args:
        data (pd.DataFrame): Dataset completo de entrada.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames separados de treino e teste.
    """
    params = load_params()
    logger.info("Executando a divisão dos dados entre treino e teste...")
    
    # Garantia de reprodutibilidade do split através dos parâmetros de configuração
    train_data, test_data = train_test_split(
        data, test_size=params["test_size"], random_state=params["random_seed"]
    )
    return train_data, test_data


def preprocess_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Executa o pipeline de engenharia de dados (imputação) nos conjuntos de treino e teste.

    Garante o alinhamento do pipeline de ML para evitar Data Leakage, 
    ajustando o imputer no treino e aplicando-o estritamente no teste.

    Args:
        train_data (pd.DataFrame): Dados de treino originais.
        test_data (pd.DataFrame): Dados de teste originais.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]: Dados processados e o artefato do imputer.
    """
    logger.info("Iniciando o tratamento de dados e imputação de valores nulos.")
    
    # Isolamento da variável target para evitar transformações indevidas nas features
    train_target = train_data['target']
    test_target = test_data['target']
    train_features = train_data.drop('target', axis=1)
    test_features = test_data.drop('target', axis=1)
    
    # Estratégia de imputação baseada na média (morfologia numérica das features)
    imputer = SimpleImputer(strategy="mean")

    # Fit e Transform aplicados separadamente para mitigar o vazamento de dados (Data Leakage)
    train_features_processed = pd.DataFrame(
        imputer.fit_transform(train_features), columns=train_features.columns
    )
    test_features_processed = pd.DataFrame(
        imputer.transform(test_features), columns=test_features.columns
    )
    
    # Reconstrução dos DataFrames consolidando as features processadas e seus respectivos targets
    train_processed = train_features_processed.assign(target=train_target.tolist())
    test_processed = test_features_processed.assign(target=test_target.tolist())
    
    return train_processed, test_processed, imputer


def save_artifacts(
    train_data: pd.DataFrame, test_data: pd.DataFrame, imputer: SimpleImputer
) -> None:
    """Persiste os datasets processados e serializa os artefatos de engenharia.

    Args:
        train_data (pd.DataFrame): Dados de treino processados.
        test_data (pd.DataFrame): Dados de teste processados.
        imputer (SimpleImputer): Objeto do imputer ajustado para reuso em inferências.
    """

    data_dir = "data/preprocessed"
    logger.info(f"Salvando bases de dados processadas no diretório {data_dir}")

    train_path = os.path.join(data_dir, "train_preprocessed.csv")
    test_path = os.path.join(data_dir, "test_preprocessed.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    # Serialização do imputer para garantir consistência no pipeline de MLOps / Scoring
    imputer_path = os.path.join("artifacts", "[features]_mean_imputer.joblib")
    logger.info(f"Saving imputer to {imputer_path}")
    joblib.dump(imputer, imputer_path)


def main() -> None:
    """Orquestrador principal do ciclo de vida de pré-processamento de dados."""
    raw_data = load_data()
    train_data, test_data = split_data(raw_data)
    train_processed, test_processed, imputer = preprocess_data(train_data, test_data)
    save_artifacts(train_processed, test_processed, imputer)
    logger.info("Pipeline de pré-processamento de dados executado com sucesso.")


if __name__ == "__main__":
    main()
