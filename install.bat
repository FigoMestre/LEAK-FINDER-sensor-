@echo off
echo ========================================
echo LEAK-FINDER Sensor - Instalacao Rapida
echo ========================================
echo.

REM Verificar se Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado!
    echo Por favor, instale Python 3.10 ou superior
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python encontrado!
python --version
echo.

REM Verificar se conda esta instalado
conda --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Conda nao encontrado!
    echo.
    echo Por favor, instale o Miniconda primeiro:
    echo 1. Baixe em: https://docs.conda.io/en/latest/miniconda.html
    echo 2. Execute o instalador
    echo 3. Reinicie o terminal
    echo 4. Execute este script novamente
    echo.
    pause
    exit /b 1
)

echo Conda encontrado!
conda --version
echo.

REM Criar ambiente conda
echo Criando ambiente conda...
conda create -n leak_finder python=3.10 -y

REM Ativar ambiente
echo Ativando ambiente...
call conda activate leak_finder

REM Instalar dependencias
echo Instalando dependencias...
cd main
pip install --upgrade pip
pip install -r requirements.txt

REM Configurar variavel de ambiente
set OPENBLAS_NUM_THREADS=1

echo.
echo ========================================
echo Instalacao concluida com sucesso!
echo ========================================
echo.
echo Para executar a aplicacao:
echo 1. Instale os drivers do UMA-16: https://www.minidsp.com/userdownloads/usb-mic-array-series/uma16-drivers
echo 2. Ative o ambiente: conda activate leak_finder
echo 3. Configure: set OPENBLAS_NUM_THREADS=1
echo 4. Execute: python acoustic_camera.py
echo.
echo IMPORTANTE: Instale os drivers do UMA-16 antes de conectar o dispositivo!
echo Para mais informacoes, consulte o README.md
echo.
pause 