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
