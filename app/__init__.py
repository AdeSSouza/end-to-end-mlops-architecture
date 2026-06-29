"""
INICIALIZAÇÃO DO MÓDULO DA APLICAÇÃO (APP)
Autor: Adeilson Souza
Contexto: Ponto de entrada estrutural para a interface de serviço do projeto.

Boas Práticas Aplicadas:
- Isolamento de escopo para o diretório de entrega do serviço (Aplicação Flask).
- Centralização de logs independentes para monitoramento das requisições web.
- Padronização de strings de formatação para auditoria em tempo de execução.
"""
import logging
from dotenv import load_dotenv

load_dotenv()

# Configuração da estratégia de logs globais para o escopo do pacote de aplicação
# Garante que os fluxos da interface de serviço herdem o mesmo layout de rastreabilidade
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)