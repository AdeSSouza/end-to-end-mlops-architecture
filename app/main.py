"""
MÓDULO PRINCIPAL DO SERVIDOR DE APLICAÇÃO (MAIN.PY)
Autor: Adeilson Souza
Contexto: Camada de controle e serviço da interface Flask na arquitetura de MLOps.

Boas Práticas Aplicadas:
- Encapsulamento de ciclo de vida de artefatos de ML através de um Service Object dedicado.
- Separação clara de responsabilidades entre rotas HTTP e processamento interno de predições.
- Proteção e tratamento do fluxo de upload por meio de validação de extensão e colunas.
- Log estruturado e traduzido para auditoria de operações em tempo de execução.
"""

import io
import logging
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import load_model

# Configuração do Logger dedicado ao monitoramento do servidor web
logger = logging.getLogger("app.main")


class ModelService:
    def __init__(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Carrega todos os artefatos de Machine Learning a partir do storage do projeto.

        Recupera as transformações e o estimador neural de forma sequencial para 
        garantir que os dados do usuário passem exatamente pelo mesmo pipeline do treino.
        """
        logger.info("Carregando artefatos da pasta local do projeto")

        # Definição de caminhos base mapeados no projeto local
        artifacts_dir = "artifacts"
        models_dir = "models"

        # Mapeamento estruturado dos arquivos binários de engenharia e modelagem
        features_imputer_path = os.path.join(
            artifacts_dir, "[features]_mean_imputer.joblib"
        )
        features_scaler_path = os.path.join(artifacts_dir, "[features]_scaler.joblib")
        target_encoder_path = os.path.join(
            artifacts_dir, "[target]_one_hot_encoder.joblib"
        )
        model_path = os.path.join(models_dir, "model.keras")

        # Desserialização dos componentes estatísticos e da rede neural Keras em memória
        self.features_imputer = joblib.load(features_imputer_path)
        self.features_scaler = joblib.load(features_scaler_path)
        self.target_encoder = joblib.load(target_encoder_path)
        self.model = load_model(model_path)

        logger.info("Todos os artefatos foram carregados com sucesso")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Orquestra a inferência sequencial aplicando as transformações necessárias.

        Args:
            features (pd.DataFrame): Dados estruturados enviados pela requisição web.

        Returns:
            pd.DataFrame: Objeto contendo os resultados das predições decodificadas.
        """
        # Aplicação sequencial do imputer de médias e do escalonador numérico
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)

        # Geração da matriz de probabilidades através da rede neural densa
        y_pred = self.model.predict(X_scaled)

        # Mapeamento reverso do array One-Hot para retornar as classes originais do negócio
        y_decoded = self.target_encoder.inverse_transform(y_pred)

        return pd.DataFrame({"Prediction": y_decoded.ravel()}, index=features.index)


def create_routes(app: Flask) -> None:
    """Configura o mapeamento de endpoints e rotas HTTP da aplicação Flask.

    Args:
        app (Flask): Instância global do servidor web.
    """

    @app.route("/")
    def index() -> str:
        """Renderiza e serve a interface HTML inicial de upload."""
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        """Processa o payload do CSV enviado, valida os atributos e retorna as predições.

        Returns:
            str: Template HTML index.html injetando variáveis de erro ou de resultados.
        """
        file = request.files["file"]
        # Filtro defensivo na interface para rejeitar formatos de arquivos diferentes de CSV
        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a CSV file")

        try:
            # Captura de bytes do arquivo carregado e decodificação em texto plano UTF-8
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            # Validação estrutural de colunas comparando o input com o esquema original do dataset
            expected_features = load_breast_cancer().feature_names
            missing_cols = [
                col for col in expected_features if col not in features.columns
            ]
            # Interrupção do fluxo caso falte algum atributo obrigatório para a rede neural
            if missing_cols:
                return render_template(
                    "index.html",
                    error=f"Missing required columns: {', '.join(missing_cols)}",
                )
            features = features[expected_features]

            # Invocação do Service Object injetado no app para geração dos resultados
            predictions = app.model_service.predict(features)

            # Conversão do DataFrame final em formato de string plana para renderização limpa
            result = predictions.to_string()

            return render_template("index.html", predictions=result)

        except Exception as e:
            # Registro de erro com rastreamento completo da pilha para depuração no WSL
            logger.error(
                f"Error processing file: {e}", exc_info=True
            )
            return render_template(
                "index.html",
                error=f"Error processing file: {str(e)}",  # Ensure e is string
            )


# Inicialização e injeção de dependências no nível do módulo da aplicação
app = Flask(__name__)
app.model_service = ModelService()
create_routes(app)
logger.info("Aplicação inicializada com o serviço do modelo e rotas configuradas")


def main() -> None:
    """Dispara a execução do servidor local de desenvolvimento do Flask."""
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
