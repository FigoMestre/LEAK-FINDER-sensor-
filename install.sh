#!/bin/bash

echo "========================================"
echo "LEAK-FINDER Sensor - Instalacao Rapida"
echo "========================================"
echo

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python3 não encontrado!"
    echo "Por favor, instale Python 3.10 ou superior"
    echo "Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "macOS: brew install python3"
    exit 1
fi

echo "Python encontrado!"
python3 --version
echo

# Verificar se conda está instalado
if command -v conda &> /dev/null; then
    echo "Conda encontrado!"
    conda --version
    echo
    
    # Criar ambiente conda
    echo "Criando ambiente conda..."
    conda create -n leak_finder python=3.10 -y
    
    # Ativar ambiente
    echo "Ativando ambiente..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate leak_finder
    
    # Instalar dependências
    echo "Instalando dependências..."
    cd main
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Configurar variável de ambiente
    export OPENBLAS_NUM_THREADS=1
    
    echo
    echo "========================================"
    echo "Instalação concluída com sucesso!"
    echo "========================================"
    echo
    echo "Para executar a aplicação:"
echo "1. Instale os drivers do UMA-16: https://www.minidsp.com/userdownloads/usb-mic-array-series/uma16-drivers"
echo "2. Ative o ambiente: conda activate leak_finder"
echo "3. Configure: export OPENBLAS_NUM_THREADS=1"
echo "4. Execute: python acoustic_camera.py"
echo
echo "IMPORTANTE: Instale os drivers do UMA-16 antes de conectar o dispositivo!"
echo
    
else
    echo "ERRO: Conda não encontrado!"
    echo
    echo "Por favor, instale o Miniconda primeiro:"
    echo "1. Baixe em: https://docs.conda.io/en/latest/miniconda.html"
    echo "2. Execute o instalador"
    echo "3. Reinicie o terminal"
    echo "4. Execute este script novamente"
    echo
    exit 1
fi

echo "Para mais informações, consulte o README.md"
echo 