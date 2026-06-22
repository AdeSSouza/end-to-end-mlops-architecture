# end-to-end-mlops-architecture

**Autor:** Adeilson Souza
**Diretriz Estratégica:** O objetivo principal deste projeto não reside na sofisticação do modelo de classificação ou no ajuste fino da rede neural profunda em si. O propósito central é o estudo prático, a engenharia e a aplicação dos pilares de MLOps (Machine Learning Operations), replicando os padrões, o rigor de isolamento e a arquitetura modular exigidos em cenários corporativos reais.


## 📁 Estrutura do Projeto

```
"""
/mlops_project
├── app/                          # Aplicação e interface web de serviço
│   ├── __init__.py               # Ponto de entrada e configuração de logs do app
│   ├── main.py                   # Servidor Flask com API de predição integrada
│   └── templates/                # Templates Jinja2 para a interface web
│       └── index.html            # Interface de upload e exibição de resultados
├── artifacts/                    # Objetos de engenharia e transformadores serializados
├── data/                         # Camadas locais de armazenamento de dados
│   ├── preprocessed/             # Dados tratados e subdivididos
│   ├── processed/                # Atributos finais (Feature Engineering)
│   └── raw/                      # Base de dados bruta inicial
├── metrics/                      # Arquivos JSON de histórico e avaliação de performance
├── models/                       # Binário do modelo treinado e compilado
├── src/                          # Módulos centrais de código-fonte (Pipeline)
│   ├── __init__.py               # Ponto de entrada e configuração de logs do pipeline
│   ├── data_loading/             # Utilitários de ingestão
│   │   ├── __init__.py
│   │   └── load_data.py          # Carregamento e salvamento do dataset bruto
│   ├── data_preprocessing/       # Tratamento de dados e divisão estrita
│   │   ├── __init__.py
│   │   └── preprocess_data.py    # Imputação de nulos e split determinístico
│   ├── feature_engineering/      # Transformações e escalonamento
│   │   ├── __init__.py
│   │   └── engineer_features.py  # Padronização de escala de recursos
│   ├── model_evaluation/         # Validação e diagnósticos técnicos
│   │   ├── __init__.py
│   │   └── evaluate_model.py     # Extração de relatórios e matriz de confusão
│   └── model_training/           # Modelagem matemática e treinamento
│       ├── __init__.py
│       └── train_model.py        # Treinamento da rede neural via TensorFlow/Keras
├── .dockerignore                 # Filtros de exclusão para o build do Docker
├── Dockerfile                    # Instruções de automação e build do contêiner
├── params.yaml                   # Centralização de hiperparâmetros de configuração
├── pyproject.toml                # Metadados do projeto e gerenciamento de dependências
└── README.md                     # Documentação oficial do repositório
"""
---

## 🚀 Funcionalidades da Arquitetura

* **Esteira Modular de Dados**: Fluxo linear de ETL isolado em 5 camadas funcionais distintas.
* **Rede Neural Profunda**: Modelo de Deep Learning (TensorFlow/Keras) com arquitetura parametrizável externamente.
* **Interface de Entrega**: Serviço web baseado em Flask e Jinja2 para processamento de arquivos em lote.
* **Ciclo de Vida de Artefatos**: Serialização completa de transformadores e estimadores para garantir consistência em produção.
* **Governança de Métricas**: Exportação automatizada de indicadores de performance para auditoria técnica.

---

## 🛠️ Requisitos e Instalação

O projeto requer **Python 3.12+** e os pacotes gerenciados listados no arquivo `pyproject.toml`.

1. Clonar o repositório remoto:
```bash
git clone https://github.com/AdeSSouza/end-to-end-mlops-architecture.git
cd mlops_project
```

2. Instalar o pacote local em modo editável no ambiente virtual (Miniconda):
```bash
pip install -e .
```

---

## ⚙️ Configuração e Parâmetros

A governança e a tunagem de hiperparâmetros do modelo, bem como os critérios de divisão de dados, são controlados centralizadamente através do arquivo externo **`params.yaml`**.

---

## 🧠 Arquitetura do Modelo

A rede neural consiste em um Perceptron Multicamadas (MLP) estruturado sequencialmente com duas camadas ocultas, utilizando ativações ReLU, regularização via camadas de Dropout para mitigação de overfitting, e uma camada de saída Softmax para classificação multiclasse.

---

## 📦 Artefatos Gerados

A execução completa do pipeline de treinamento exporta de forma automatizada os seguintes binários:

No diretório `models/`:
* `model.keras`: Rede neural treinada e compilada pelo TensorFlow.

No diretório `artifacts/`:
* `[features]_mean_imputer.joblib`: Transformador estatístico para tratamento de valores ausentes.
* `[features]_scaler.joblib`: Objeto do StandardScaler para padronização de escala de recursos.
* `[target]_one_hot_encoder.joblib`: Codificador de categorias para mapeamento e tradução reversa da variável resposta.

---

## 📊 Métricas de Performance

Os relatórios de desempenho são extraídos ao final do fluxo e gravados em:
* `metrics/training.json`: Indicadores de perda e acurácia obtidos na última época do treinamento.
* `metrics/evaluation.json`: Matriz de confusão e métricas consolidadas (precisão, recall, f1-score) geradas sobre a base de teste.

---

## 🔄 Execução do Pipeline (Desenvolvimento)

Os módulos operam de forma independente, salvando as saídas em disco de maneira linear. Para executar cada etapa do pipeline registrando os logs estruturados no terminal, utilize os comandos como módulos Python (`python -m`):

```bash
# 1. Executar a ingestão e preparação dos dados brutos
python -m src.data_loading.load_data

# 2. Executar o pré-processamento (imputação e split determinístico)
python -m src.data_preprocessing.preprocess_data

# 3. Executar a engenharia de recursos (padronização de escala das variáveis)
python -m src.feature_engineering.engineer_features

# 4. Executar o treinamento da rede neural profunda
python -m src.model_training.train_model

# 5. Executar a avaliação estatística final sobre a base de teste
python -m src.model_evaluation.evaluate_model
```

---

## 🌐 Inicialização da Interface de Serviço

### Inicialização Nativa (Flask)
Após a conclusão do pipeline de modelagem e geração dos artefatos, dispare o servidor web de desenvolvimento:

```bash
python app/main.py
```
A aplicação web estará operacional para acessos locais através do endereço: `http://localhost:5001`

### Inicialização Isolada (Docker)
Para simular o isolamento e o empacotamento em ambiente produtivo, construa e execute o contêiner via terminal Linux:

```bash
# 1. Construir a imagem Docker a partir do arquivo de instruções
docker build -t ml-classifier .

# 2. Inicializar o contêiner mapeando a porta do servidor Gunicorn
docker run -p 5001:5001 ml-classifier
```
A aplicação web conteinerizada estará disponível no endereço: `http://localhost:5001`

---

## 🎯 Consumo do Serviço e Predições

1. **Interface Gráfica**: Acesse o painel pelo navegador e carregue um arquivo em formato CSV contendo os atributos preditores.
2. **Endpoint da API**: O endpoint `/upload` recebe requisições do tipo `POST` transmitindo arquivos de dados e retorna as predições decodificadas na tela.

### Layout Obrigatório do Arquivo CSV
O arquivo de entrada deve ser estruturado e conter exatamente os 30 atributos numéricos do dataset original (nomes exatos das colunas como `mean radius`, `mean texture`, `mean perimeter`, entre outros, conforme mapeado no esquema nativo `sklearn.datasets.load_breast_cancer().feature_names`).