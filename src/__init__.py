"""
INICIALIZAÇÃO DO PACOTE E GOVERNANÇA DE LOGS
Autor: Adeilson Souza
Contexto: Ponto de entrada estrutural do projeto end-to-end-mlops-architecture.

Boas Práticas Aplicadas:
- Consolidação do diretório como pacote raiz para viabilizar o desenvolvimento editável.
- Centralização da estratégia de logs em nível de pacote para herança uniforme dos submódulos.
- Padronização de formato de timestamps e rastreabilidade por número de linha.
"""

import logging
from dotenv import load_dotenv

load_dotenv()

# Configuração centralizada da estratégia de logs globais da arquitetura
# Definido no escopo raiz do pacote para que todas as 5 camadas funcionais herdem esta estrutura
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S", 
    handlers=[
        logging.StreamHandler() 
    ]
)

