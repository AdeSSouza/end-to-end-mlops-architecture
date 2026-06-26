# 📝 Notas de Desenvolvimento - Arquitetura End-to-End de MLOps
**Autor:** Adeilson Souza  
**Objetivo:** Registro técnico das decisões de engenharia, conceitos assimilados e resolução de desafios de infraestrutura.

---

## 🛠️ 1. Notas de Infraestrutura e Ambiente Local

### 🔹 Integração Windows-WSL e Ambiente Isolado
- **WSL Ubuntu:** Utilizado como base de simulação de ambiente de produção (servidor Linux), garantindo a portabilidade real do sistema e eliminando o "na minha máquina funciona".
- **Miniconda:** Gerenciamento de ambientes virtuais focado em Ciência de Dados, permitindo o isolamento estrito de versões do Python e pacotes não-Python (binários).
- **Modo Editável (`pip install -e .`):** Configurado via `pyproject.toml` para permitir que alterações nos pacotes internos de `src/` sejam refletidas instantaneamente na aplicação web sem necessidade de reinstalações manuais.
- **Hierarquia de Logging (`load_data`):** O uso de strings separadas por pontos (ex: `"src.data_loading.load_data"`) permite que o logger herde automaticamente todas as configurações globais de formato e nível de criticidade definidas na raiz do projeto, mantendo o código limpo e consistente.

### 1.1. Inicialização do Pacote e Logs (`src/__init__.py`)
* **Importações Absolutas**: Ativa o diretório `src` como pacote raiz, garantindo a estabilidade do modo editável (`pip install -e .`).
* **Logs Centralizados**: Define a configuração global de logs por herança para as 5 camadas do pipeline de dados.
* **Depuração por Linha**: Inclui o metadado `%(lineno)d` no formato do log para agilizar a identificação de erros no terminal WSL.

### 1.2. Inicialização da Aplicação e Logs (`app/__init__.py`)
* **Isolamento de Serviço**: Separa o escopo de entrega da aplicação web (Flask) do fluxo principal de treinamento de Machine Learning.
* **Logs Simplificados**: Omite o número da linha de código na string de formato, gerando uma leitura mais limpa para monitorar requisições de API.

### 1.3. Interface de Visualização (`app/templates/index.html`)
* **Formulário de Upload**: Configura a captura de dados via método `POST` usando a codificação `multipart/form-data`, obrigatória para o tráfego de arquivos de dados.
* **Filtro de Extensão**: Utiliza a validação nativa do navegador (`accept=".csv"`) para mitigar o envio de formatos incompatíveis com o pipeline.
* **Exibição Condicional (Jinja2)**: Controla dinamicamente a interface através de tags `{% if %}`, renderizando blocos de erros ou payloads de predições de forma isolada na tela.

### 1.4. Controlador e Serviço de Inferência (`app/main.py`)
* **Design Pattern de Serviço**: Centraliza o ciclo de vida dos artefatos serializados (`imputer`, `scaler`, `encoder` e `model`) em um objeto de serviço, simulando uma esteira isolada de produção.
* **Mapeamento de Inferência**: Executa a transformação de dados de entrada respeitando a exata cronologia matemática vista na esteira de treino, realizando o mapeamento reverso de matrizes binárias via `inverse_transform`.
* **Validação de Esquema**: Implementa barreira defensiva comparando programaticamente as colunas do CSV de entrada contra os metadados do dataset nativo da scikit-learn antes de disparar o cálculo do TensorFlow.

### 1.5. Configuração de Conteinerização (`Dockerfile`)
* **Otimização de Imagem**: Utiliza a variante `python:3.12-slim` como imagem base para garantir um contêiner mais leve e com menor sobrecarga de pacotes desnecessários.
* **Gerenciamento de Pacotes**: Executa o comando `RUN pip install` apontando para o diretório copiado, automatizando a resolução das dependências corporativas mapeadas no instalador do projeto.
* **Camada de Produção (WSGI)**: Substitui o servidor de desenvolvimento nativo do Flask pelo `gunicorn` em tempo de execução, garantindo resiliência e concorrência para o serviço na porta `5001`.

### 1.6. Documentação de Referência do Repositório (`README.md`)
* **Estruturação de Diretrizes**: Organiza o arquivo como uma Ficha Técnica de Engenharia destacando o foco estratégico em MLOps, os comandos estruturados por pacotes (`-m`) e as diretrizes de conteinerização de forma direta.

---

## 2. Notas de Engenharia e Pré-processamento (`preprocess_data.py`)

### Decisões Técnicas de Implementação
* **Prevenção contra Data Leakage**: O processo de imputação de valores ausentes (`SimpleImputer`) foi isolado. O ajuste do estimador (`.fit_transform()`) ocorre exclusivamente sobre os dados de treino, enquanto o conjunto de teste recebe apenas a aplicação (`.transform()`). O objetivo é evitar o vazamento de estatísticas globais para o teste.
* **Reprodutibilidade do Split**: A divisão das bases (`train_test_split`) utiliza os parâmetros `test_size` e `random_seed` mapeados diretamente do `params.yaml`, garantindo que o comportamento do script seja idêntico em qualquer execução.
* **Ciclo de Vida de Artefatos**: O objeto imputer é persistido via `joblib` na pasta `/artifacts` para garantir que a mesma transformação aplicada no treinamento possa ser reutilizada de forma consistente na etapa de predição.

---

## 3. Notas de Engenharia de Recursos (`engineer_features.py`)

### Decisões Técnicas de Implementação
* **Isolamento Estatístico no Escalonamento**: Assim como na imputação, o cálculo de média e desvio padrão do `StandardScaler` foi restrito ao conjunto de treino. Isso garante que o conjunto de teste simule o comportamento de dados inéditos.
* **Evitando Efeitos Colaterais com Pandas**: Adotei o uso explícito de `.copy()` ao criar as variáveis de dados processados. Essa prática aloca novos blocos de memória independentes, eliminando o aviso *SettingWithCopyWarning* do Pandas e garantindo previsibilidade no fluxo.
* **Criação Dinâmica de Pastas**: Adicionei o comando `os.makedirs(..., exist_ok=True)` antes do salvamento dos arquivos. Isso torna o pipeline autônomo, pois o script cria as pastas necessárias (`data/processed` e `artifacts`) caso elas não existam no ambiente de execução.

---

## 4. Notas de Treinamento do Modelo (`train_model.py`)

### Decisões Técnicas de Implementação
* **Externalização de Hiperparâmetros**: Toda a configuração estrutural da rede neural (neurônios, dropout, épocas, batch size) é lida dinamicamente do `params.yaml`. Isso isola a lógica de código das iterações de tunagem do modelo.
* **Codificação Categórica do Target**: Aplicação do `OneHotEncoder` na variável alvo para adequação ao formato de matriz de probabilidade exigido pela última camada (`softmax`) da rede neural multiclasse. O objeto foi salvo em `/artifacts` para ser reutilizado no mapeamento reverso durante as predições do serviço Web.
* **Reprodutibilidade do Algoritmo**: Uso do `tf.keras.utils.set_random_seed` alimentado pelo parâmetro do YAML. Essa configuração fixa o estado inicial dos pesos da rede e garante resultados idênticos em reexecuções.
* **Prevenção de Overfitting**: Integração da função `EarlyStopping` configurada para monitorar a perda de validação (`val_loss`) com paciência de 10 épocas, restaurando automaticamente os melhores pesos encontrados durante o processo.
* **Automação de Diretórios e Métricas**: Inclusão de tratamentos com `os.makedirs` para as pastas `models` e `metrics`. O script extrai a última linha de performance do histórico e gera automaticamente um arquivo `training.json` para futura auditoria de acurácia.


---

## 5. Notas de Avaliação do Modelo (`evaluate_model.py`)

### Decisões Técnicas de Implementação
* **Simulação de Ambiente Produtivo**: O script executa o carregamento desacoplado do modelo (`model.keras`) e do seu transformador (`[target]_one_hot_encoder.joblib`). Esse design garante a validação da esteira exatamente como ela ocorreria em um microsserviço de produção.
* **Isolamento de Teste**: A leitura e particionamento ocorrem sobre o arquivo `test_processed.csv`. Como esses dados não foram vistos pela rede neural durante o treinamento, o resultado estatístico reflete a real capacidade de generalização do algoritmo.
* **Extração de Diagnósticos de Classificação**: O script utiliza a função `np.argmax()` para traduzir as probabilidades geradas pela última camada `softmax` da rede neural em predições de classes discretas.
* **Estruturação de Métricas para Auditoria**: As métricas geradas via `classification_report` (precisão, recall, f1-score) e `confusion_matrix` são unificadas em um dicionário Python e gravadas de forma automatizada no arquivo `metrics/evaluation.json`, permitindo rastreabilidade histórica de performance.

---

## ⛓️ Governança de Dados e Modelos com DVC (Data Version Control)

Neste módulo, implementei a orquestração e o versionamento do pipeline de Machine Learning utilizando o DVC v3.59.1, integrando o controle de código do Git ao rastreamento de artefatos pesados e experimentos.

### 1. Arquitetura do Pipeline (`dvc.yaml`)
A infraestrutura do pipeline foi mapeada em um Grafo Acíclico Dirigido (DAG) contendo 5 estágios interdependentes. Cada estágio declara estritamente seus comandos de execução (`cmd`), dependências de código/dados (`deps`), parâmetros de entrada (`params`) e artefatos de saída (`outs` / `metrics`):
- **`load_data`**: Ingestão dos dados brutos (`data/raw/raw.csv`).
- **`preprocess_data`**: Tratamento de dados, imputação de nulos (`_mean_imputer.joblib`) e divisão em treino e teste.
- **`engineer_features`**: Transformações e escalonamento de variáveis com salvamento do estado do transformador (`_scaler.joblib`).
- **`train`**: Treinamento do modelo de Deep Learning em Keras (`model.keras`) com salvamento do encoder do alvo (`_one_hot_encoder.joblib`) e exportação de métricas de treino.
- **`evaluate`**: Avaliação de performance contra a base de teste e geração de relatórios de métricas (`metrics/evaluation.json`).

### 2. Automação e Ciclo de Vida da Pipeline
- **Idempotência (`dvc repro`):** Otimização do tempo de computação. O DVC calcula os *hashes* das dependências e pula (*skip*) etapas cujos códigos, parâmetros e dados de entrada não sofreram alterações.
- **Simulação de Drift de Dados:** Adicionado gatilho temporal baseado em `time.time()` no script de carga para embaralhar as linhas e forçar a reexecução completa do pipeline para testes de linhagem.
- **Rastreabilidade Entre Commits:** Sincronização do estado dos dados com versões antigas do código através da combinação dos comandos `git checkout <hash>` e `dvc checkout`.
- **Autostage:** Configurado `dvc config core.autostage true` para automatizar a indexação dos ponteiros de dados diretamente no estágio de preparação do Git.

### 3. Gerenciamento e Rastreamento de Experimentos
Utilização do motor de experimentos do DVC para tunagem de hiperparâmetros sem poluição do histórico de branches do Git:
- **Execução Isolada:** Uso de `dvc exp run -S <parametro>=<valor>` para modificar dinamicamente variáveis como `batch_size`, `test_size` e `dropout_rate`.
- **Fila de Execução (`--queue`):** Criação e enfileiramento de múltiplos experimentos concorrentes para processamento em lote via `dvc exp run --run-all`.
- **Auditoria de Resultados:** Visualização e comparação tabular de métricas e hiperparâmetros através do `dvc exp show`.
- **Persistência de Melhor Modelo:** Aplicação do experimento vencedor de volta ao código produtivo via `dvc exp apply <experiment_id>`.

### 4. Armazenamento Remoto Comercial (DagsHub)
Configuração do DagsHub como o provedor remoto de armazenamento de dados e modelos da arquitetura:
- Armazenamento centralizado seguro e independente do GitHub para manter o repositório de código leve.
- Autenticação local segura via credenciais básicas em formato de token de acesso restrito (evitando vazamento de segredos no histórico do Git).
- Sincronização bidirecional do pipeline por meio dos comandos de transporte de dados `dvc push` e `dvc pull`.

---

## 🧪 Rastreamento Avançado e Ciclo de Vida de Modelos com MLflow

Implementamos a integração do MLflow v2 como o motor central de observabilidade, auditoria e governança do ciclo de vida dos modelos de Machine/Deep Learning, estabelecendo a interoperabilidade direta com o ecossistema DVC.

### 1. Centralização e Segurança de Variáveis de Ambiente
Para desacoplar as configurações de infraestrutura do código-fonte e garantir conformidade com as boas práticas de segurança da informação:
- **`python-dotenv`:** Injetado na arquitetura para gerenciar strings de conexão e parâmetros globais.
- **Isolamento de Credenciais:** Configurado o arquivo `.env` para apontar o servidor de rastreamento (`MLFLOW_TRACKING_URI=http://localhost:5000`) e devidamente indexado no `.gitignore` para mitigar o risco de vazamento de segredos no repositório remoto.
- **Inicialização Nativa:** Centralizado o carregamento automático das variáveis no arquivo `src/__init__.py` via `load_dotenv()`, assegurando que qualquer módulo do pipeline inicialize com o contexto correto de variáveis de ambiente.

### 2. Orquestração e Interoperabilidade DVC + MLflow (Nested Runs)
Desenvolvemos um mecanismo dinâmico no script de treinamento (`train_model.py`) para capturar o contexto de execuções concorrentes originadas pelo gerenciador de experimentos do DVC, resolvendo o desafio de manter as duas ferramentas sincronizadas:
- **Detecção de Contexto:** O pipeline monitora ativamente a existência da variável `DVC_EXP_NAME`.
- **Rastreamento Aninhado (`Nested Runs`):** Caso um experimento seja disparado em lote (via fila do DVC), o MLflow realiza uma varredura via `mlflow.search_runs` procurando a tag `tags.dvc_exp = 'True'`. 
- **Hierarquia Visual:** Se for a primeira execução do lote, uma execução "mãe" (*Parent Run*) é gerada. Os testes subsequentes de hiperparâmetros gerados pelo DVC (`--queue`) são injetados automaticamente como execuções "filhas" (*Child Runs*) usando a propriedade `nested=True`. Isso estruturou uma árvore hierárquica limpa no Dashboard do MLflow.

### 3. Execução de Experimentos em Lote no Terminal
A validação do ecossistema integrado foi consolidada por meio de comandos de linha de comando (CLI), eliminando gargalos manuais de parametrização:
- Enfileiramento de testes concorrentes de hiperparâmetros (como modificações em `test_size`, `dropout_rate` e `random_seed`) utilizando a flag `--queue`.
- Processamento massivo paralelo/sequencial disparado nativamente pelo comando corporativo:
  ```bash
  dvc exp run --run-all
  ```
- Auditoria, comparação de curvas de perda (*loss*) e métricas de acurácia consolidadas centralizadamente por meio da interface gráfica do **MLflow UI** (`mlflow ui`).
