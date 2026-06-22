# --- Definindo estratégia de log pra o projeto --- #

import logging

# Configure the logging strategy 
# configurando o sistema de logs, quando feita a nível de pacote, todos os pacotes e módulos dentro de sourcing herdam essa configuração
# garantido consistência e uniformidade dos registros de log ao longo do projeto.
logging.basicConfig(
    level=logging.INFO, # Nível de log DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s", # Timestamp, nome do módulo, linha do código, nível do log, mensagem quando log é invocado
    datefmt="%Y-%m-%d %H:%M:%S", # Formato de data
    handlers=[
        logging.StreamHandler() # Dertermina pra onde o log será enviado, aqui imprime no console na tela mesmo
    ]
)

