import logging
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


logger = logging.getLogger("src.register_artifacts")
    
client = MlflowClient()

def get_best_run(experiment_id: str, parent_run_id: str) -> pd.Series:
    """Busca a melhor execução filha com base na acurácia de teste para uma execução mãe específica.
    
    Args:
        client: Instância do cliente MLflow
        parent_run_id: ID da execução mãe (parent run)
        
    Returns:
        A melhor execução estruturada como uma Series do pandas
    """
    # Busca todas as execuções filhas vinculadas à execução pai
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["metrics.test_accuracy DESC"],
        max_results=1000
    )       
    # Retorna a execução com a maior acurácia de teste cadastrada
    return child_runs[0]

def register_model() -> None:
    """Registra o modelo que foi persistido durante a etapa de treinamento."""

    logger.info("Iniciando o registro do modelo a partir da última execução do MLflow")

    # Recupera o ID do experimento com o nome 'ml_classification'
    experiment_id = client.get_experiment_by_name("ml_classification").experiment_id

    # Obtém a execução mais recente do experimento
    latest_run = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )[0]
    
    # Verifica se a execução mais recente possui um vínculo com uma execução pai
    run_id = latest_run.info.run_id
    parent_run_id = latest_run.data.tags.get('mlflow.parentRunId')
    if parent_run_id:
        logger.info(f"Última execução vinculada ao ID da execução pai: {parent_run_id}")
        best_run = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            order_by=["metrics.test_accuracy DESC"],
            max_results=1
        )[0]
        run_id = best_run.info.run_id
        logger.info(f"Utilizando a melhor execução ({run_id}) com acurácia de teste: {best_run.data.metrics['test_accuracy']}")

    # Registra o modelo com base na execução selecionada
    logger.info("Registering model")
    try:
        client.create_registered_model("model")
    except mlflow.exceptions.MlflowException:
        logger.debug("O modelo já existe no registro centralizado")

    model_uri = f"runs:/{run_id}/model"
    client.create_model_version(
        name="model",
        source=model_uri,
        run_id=run_id
    )
    logger.info("Modelo registrado com sucesso")


def main() -> None:
    """Função principal para orquestrar o processo de registro do modelo."""
    register_model()
    logger.info("Processo de registro do modelo concluído")


if __name__ == "__main__":
    main()
