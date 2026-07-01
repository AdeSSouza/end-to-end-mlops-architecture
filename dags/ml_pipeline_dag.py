import yaml
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Define a raiz do projeto dinamicamente a partir da localização deste script
project_root = Path(__file__).resolve().parents[1]

def get_dvc_stages():
    """Lê o arquivo dvc.yaml e retorna a lista com o nome de todos os estágios configurados."""
    dvc_yaml_path = project_root / "dvc.yaml"
    with open(dvc_yaml_path) as f:
        dvc_config = yaml.safe_load(f)
    return list(dvc_config["stages"].keys())

def register_artifacts_callable():
    """Gatilho para executar o script de registro de artefatos e governança do modelo."""
    from src.register_artifacts import main
    main()

# Argumentos padrão para controle de resiliência e retentativas das tarefas
default_args = {
    "owner": "airflow",
    "retries": 1,
}

with DAG(
    "ml_pipeline",
    default_args=default_args
) as dag:
    # Recupera dinamicamente os estágios mapeados no pipeline do DVC
    dvc_stages = get_dvc_stages()

    # Cria instâncias de tarefas para cada estágio identificado no DVC
    dvc_tasks = []
    for stage in dvc_stages:
        task = BashOperator(
            task_id=f"dvc_{stage}",
            cwd=project_root,
            bash_command=f"dvc repro {stage}"
        )
        dvc_tasks.append(task)

    # Tarefa dedicada para registrar as métricas e o modelo campeão no MLflow
    register_artifacts = PythonOperator(
        task_id="register_artifacts",
        python_callable=register_artifacts_callable
    )

    # Etapa de implantação: reconstrói a imagem Docker com o modelo e códigos atualizados
    create_app_image = BashOperator(
        task_id="create_app_image",
        cwd=project_root,
        bash_command = "docker build -t ml-classifier ."
    )

    # Estabelece a dependência sequencial e linear entre os estágios do DVC
    for i in range(len(dvc_tasks) - 1):
        dvc_tasks[i] >> dvc_tasks[i + 1]

    # Conecta o último estágio do DVC ao registro no MLflow e, em seguida, dispara o build do Docker
    dvc_tasks[-1] >> register_artifacts >> create_app_image
