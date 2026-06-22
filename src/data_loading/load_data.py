"""
MÓDULO DE CARREGAMENTO DE DADOS (INGESTÃO)
Autor: Adeilson Souza
Contexto: Primeira camada funcional da arquitetura modular de MLOps.

Boas Práticas Aplicadas:
- Separação clara de responsabilidades (Modularidade).
- Rastreabilidade do fluxo do sistema através de Logs estruturados.
- Preparação de pipeline resiliente a falhas de dados estruturais (NaNs).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Configuração do Logger hierárquico para garantir rastreabilidade em produção
logger = logging.getLogger("src.data_loading.load_data") 

def fetch_data() -> pd.DataFrame: 
    """Coleta o dataset original e converte em estrutura tabular.

    Returns:
        pd.DataFrame: DataFrame contendo as features e a coluna alvo.
    """
    logger.info("Iniciando a coleta dos dados brutos...")
    dataset = load_breast_cancer()
    
    # Construção do DataFrame com nome das colunas
    data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    
    # NOTA PESSOAL: Injeção artificial de 5% de NaNs com semente fixa (seed 42).
    # Decisão estratégica para testar a resiliência e a robustez dos módulos 
    # seguintes do pipeline (Imputação e Pré-processamento) simulando cenários reais.
    np.random.seed(42)
    for col in data.columns:
        mask = np.random.random(len(data)) < 0.05  # 5% chance of NaN
        data.loc[mask, col] = np.nan
    
    # Inclusão da coluna de respostas (target) no DataFrame final
    data["target"] = dataset.target
    
    return data


def save_data(data: pd.DataFrame) -> None:
    """Salva os dados brutos no disco local em formato estruturado.

    Args:
        data (pd.DataFrame): Dataset tabular contendo NaNs artificiais.
    """
    # Definição do caminho relativo padronizado na arquitetura do projeto
    output_path = "data/raw/raw.csv"
    logger.info(f"Saving raw data to {output_path}")
    data.to_csv(output_path, index=False) # index=False evita a criação de colunas redundantes de ID ao recarregar o arquivo


def main() -> None:
    """Orquestrador principal do fluxo de carregamento/ingestão."""
    logger.info("=== Executando Módulo: load_data ===")
    raw_data = fetch_data()
    save_data(raw_data)
    logger.info("Módulo load_data concluído com sucesso.")


if __name__ == "__main__":
    main()
