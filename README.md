# 🐳 End-to-End MLOps Architecture

* **Autor:** Adeilson Souza (Mestre em Ciência da Computação)
* **Diretriz Estratégica:** O propósito central deste projeto reside na engenharia, automação e aplicação prática dos pilares de MLOps (Machine Learning Operations). A arquitetura replica o rigor de isolamento, a governança de dados e os padrões modulares exigidos em cenários corporativos reais de produção.

---

## 🏗️ 1. Arquitetura Geral da Solução

```text
[ WSL2 / Ubuntu ] ── Orquestração Local Distribuída
       │
       ├──► 🕒 Apache Airflow ──── LocalExecutor + Postgres + Docker-in-Docker (DinD)
       │         │
       │         └───► ⛓️ DVC repro ── Executa a DAG do Pipeline (Ingestão ao Treino)
       │                   │
       │                   ├──► 📊 DagsHub (Nuvem) ── Armazena Dados/Modelos e Logs MLflow
       │                   │
       │                   └──► 🐳 Docker Hub ───── Build e Push da Imagem Flask (Gunicorn)
```

---

## 📁 2. Estrutura do Projeto

```text
/mlops_project
├── app/                          # Aplicação e interface web de serviço
│   ├── __init__.py               # Entrada, carga do .env e configuração de logs do app
│   ├── main.py                   # Servidor Flask com API de predição integrada
│   └── templates/                # Templates Jinja2 para a interface web
│       └── index.html            # Interface de upload e exibição de resultados
├── artifacts/                    # Objetos de engenharia e transformadores (.joblib)
├── data/                         # Camadas locais de armazenamento de dados (DVC)
│   ├── preprocessed/             # Dados tratados e subdivididos
│   ├── processed/                # Atributos finais (Feature Engineering)
│   └── raw/                      # Base de dados bruta inicial
├── metrics/                      # Arquivos JSON de histórico e avaliação de performance
├── models/                       # Binário do modelo treinado e compilado (.keras)
├── src/                          # MModules centrais de código-fonte (Pipeline)
│   ├── __init__.py               # Entrada, dagshub.init() e carregamento global do .env
│   ├── data_loading/             # Utilitários de ingestão
│   ├── data_preprocessing/       # Tratamento de dados e divisão estrita
│   ├── feature_engineering/      # Transformações e escalonamento
│   ├── model_evaluation/         # Validação e diagnósticos técnicos
│   └── model_training/           # Modelagem matemática e treinamento
├── Dockerfile                    # Instruções de build da API de Produção (Flask/Gunicorn)
├── Dockerfile.airflow            # Imagem customizada do Airflow com suporte a DVC
├── docker-compose.airflow.yaml   # Orquestrador multi-serviço (Airflow, Postgres, DinD)
├── params.yaml                   # Centralização de hiperparâmetros de configuração
└── pyproject.toml                # Metadados do projeto e gerenciamento de dependências
```

---

## 🛠️ 3. Requisitos Prévios e Configuração de Variáveis

O projeto requer **Python 3.12+**, **Docker Desktop** integrado ao **WSL2 (Ubuntu)** e o gerenciador **Miniconda**.

### 3.1. Clonagem e Instalação
```bash
git clone https://github.com/AdeSSouza/end-to-end-mlops-architecture.git
cd mlops_project
conda create -n mlops-env python=3.12 -y
conda activate mlops-env
pip install -e .
```

### 3.2. Configuração Obligatória de Variáveis de Ambiente (`.env`)
Crie um arquivo `.env` na raiz do seu projeto (devidamente listado no seu `.gitignore`) e popule com as suas credenciais de nuvem:
```env
# Servidor de Rastreamento (MLflow na Nuvem via DagsHub)
MLFLOW_TRACKING_URI=https://dagshub.com
DAGSHUB_USER_TOKEN=seu_token_privado_do_dagshub

# Parâmetros de Automação para Deploy de Contêineres (Docker Hub)
DOCKER_HUB_USERNAME=seu_usuario_do_docker_hub
DOCKER_HUB_TOKEN=seu_token_de_acesso_do_docker_hub
```
---

## ⚙️ 4. Governança, Orquestração e Execução do Ecossistema

### 4.1. Execução Manual do Pipeline (Ambiente de Desenvolvimento)
Caso queira rodar as etapas de modelagem isoladamente registrando logs no terminal, utilize os comandos como módulos Python:
```bash
python -m src.data_loading.load_data          # 1. Ingestão dos dados brutos
python -m src.data_preprocessing.preprocess_data # 2. Imputação e split determinístico
python -m src.feature_engineering.engineer_features # 3. Padronização de escala
python -m src.model_training.train_model        # 4. Treinamento TensorFlow/Keras
python -m src.model_evaluation.evaluate_model   # 5. Avaliação estatística de teste
```

### 4.2. Governança de Experimentos e Dados via CLI (DVC & MLflow)
Toda a esteira de dados é governada e pode ser disparada de forma automatizada e reprodutível:
```bash
# Executar o pipeline completo via DVC (Idempotente)
dvc repro

# Enfileirar múltiplos experimentos modificando hiperparâmetros
dvc exp run -S train.batch_size=32 -S train.dropout_rate=0.3 --queue
dvc exp run -S train.batch_size=64 -S train.dropout_rate=0.2 --queue

# Processar todos os experimentos da fila em lote
dvc exp run --run-all

# Abrir o Dashboard local do MLflow para auditar as curvas de métricas
mlflow ui
```

### 4.3. Orquestração Completa e CI/CD Produtivo (Apache Airflow)
Para subir o ambiente multi-serviço robusto integrado ao Docker-in-Docker e disparar a esteira automatizada com deploy no Docker Hub:
```bash
# 1. Construir a imagem customizada do Airflow aplicando alterações de código
docker compose -f docker-compose.airflow.yaml build

# 2. Ligar a stack completa visualizando os logs (Trava o terminal)
docker compose -f docker-compose.airflow.yaml up

# 3. Derrubar o ambiente limpando as redes internas e contêineres fantasmas
docker compose -f docker-compose.airflow.yaml down

# 4. Forçar a recriação da malha de rede em caso de falha de conexão com o banco
docker compose -f docker-compose.airflow.yaml up --force-recreate
```
*A interface web do Apache Airflow estará operacional em: `http://localhost:8080`*

---

## 🌐 5. Inicialização da API e Consumo do Serviço

### 5.1. Inicialização do Contêiner da API Flask (Produção)
Para simular o isolamento produtivo rodando sob o servidor Gunicorn:
```bash
# 1. Construir a imagem Docker local
docker build -t ml-classifier .

# 2. Inicializar o contêiner mapeando as portas de comunicação
docker run -p 5001:5001 ml-classifier
```
*A aplicação web conteinerizada estará disponível no endereço: `http://localhost:5001`*

### 5.2. Consumo do Serviço e Predições
1. **Interface Gráfica:** Acesse o painel pelo navegador em `http://localhost:5001` e carregue um arquivo em formato CSV contendo os atributos preditores.
2. **Endpoint da API:** O endpoint `/upload` recebe requisições do tipo `POST` transmitindo arquivos de dados e retorna as predições decodificadas na tela.

### Layout Obrigatório do Arquivo CSV
O arquivo de entrada deve ser estruturado e conter exatamente os 30 atributos numéricos do dataset original (nomes exatos das colunas como `mean radius`, `mean texture`, `mean perimeter`, conforme mapeado no esquema nativo `sklearn.datasets.load_breast_cancer().feature_names`).
