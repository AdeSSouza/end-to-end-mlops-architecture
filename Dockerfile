# ######################################################################
# ARQUIVO DE CONFIGURAÇÃO DE CONTÊINER (DOCKERFILE)
# Autor: Adeilson Souza
# Contexto: Camada de empacotamento e implantação da aplicação em MLOps.
# 
# Boas Práticas Aplicadas:
# - Utilização de imagem base otimizada (slim) para redução da área de ataque e tamanho.
# - Instalação centralizada do pacote local garantindo resolução automática de dependências.
# - Definição explícita de diretório de trabalho e isolamento de porta de rede.
# - Utilização do Gunicorn como servidor WSGI pronto para ambiente de produção.
# ######################################################################

# Imagem base oficial do Python em versão reduzida para eficiência de armazenamento
FROM python:3.12-slim

# Cópia integral dos arquivos do projeto local para dentro do diretório do contêiner
COPY . mlops_project/

# Execução da instalação do pacote local em modo regular com todas as suas dependências
RUN pip install ./mlops_project

# Definição do diretório de trabalho padrão para as próximas instruções em tempo de execução
WORKDIR /mlops_project

# Declaração formal da porta de rede que será aberta no contêiner para tráfego web
EXPOSE 5001

# Comando de inicialização do servidor de produção Gunicorn mapeando o objeto Flask
CMD ["gunicorn", "--bind=0.0.0.0:5001", "app.main:app"]
