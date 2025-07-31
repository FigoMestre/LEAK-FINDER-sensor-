# LEAK-FINDER Sensor - Câmera Acústica

Este projeto implementa uma câmera acústica em tempo real usando uma webcam USB e um array de microfones miniDSP UMA-16 (16 microfones) no Windows. Utiliza Python, processamento de áudio avançado e OpenCV para visualização.

## 🎯 Sobre o Projeto

O **LEAK-FINDER Sensor** é uma solução de detecção acústica que permite:
- **Localização de vazamentos**: Identifica fontes de som em tempo real
- **Análise de ruído**: Mapeia a intensidade sonora em diferentes frequências
- **Visualização integrada**: Combina vídeo da câmera com overlay de intensidade sonora
- **Interface intuitiva**: Controles em tempo real para ajuste de filtros e parâmetros

## 🔧 Requisitos de Hardware

### Essenciais:
- **miniDSP UMA-16**: Array de 16 microfones USB
- **Webcam USB**: Qualquer câmera compatível com OpenCV
- **Computador**: Windows 10/11 (compatível com outros SO com modificações)

### Recomendados:
- **Processador**: Intel i5/AMD Ryzen 5 ou superior
- **RAM**: 8GB ou mais
- **USB 3.0**: Para melhor performance do array de microfones

## 💻 Requisitos de Software

- **Python**: 3.10 ou superior
- **Miniconda/Anaconda**: Necessário para gerenciamento de ambiente
  - **Miniconda** (recomendado): https://docs.conda.io/en/latest/miniconda.html
  - **Anaconda** (alternativa): https://www.anaconda.com/products/distribution
  - **IMPORTANTE**: Instale o Miniconda/Anaconda antes de começar
- **Drivers UMA-16**: Drivers oficiais do miniDSP UMA-16
  - Download: https://www.minidsp.com/userdownloads/usb-mic-array-series/uma16-drivers
  - **IMPORTANTE**: Instale os drivers antes de conectar o dispositivo

## 🚀 Instalação e Configuração

### ⚠️ IMPORTANTE: Pré-requisitos

**ANTES de começar a instalação do software, você DEVE instalar:**

1. **Miniconda/Anaconda**:
   - **Miniconda** (recomendado): https://docs.conda.io/en/latest/miniconda.html
   - **Anaconda** (alternativa): https://www.anaconda.com/products/distribution
   - Execute o instalador e siga as instruções
   - Reinicie o terminal após a instalação

2. **Drivers do UMA-16**:
   - Baixe os drivers oficiais: https://www.minidsp.com/userdownloads/usb-mic-array-series/uma16-drivers
   - Execute o instalador como administrador
   - **Reinicie o computador** após a instalação
   - Só então conecte o UMA-16 via USB

### Passo 1: Preparar o Ambiente

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/LEAK-FINDER-sensor-.git
   cd LEAK-FINDER-sensor-
   ```

2. **Verifique se o Conda está instalado:**
   ```bash
   conda --version
   ```
   Se não estiver instalado, instale o Miniconda primeiro.

3. **Crie e ative o ambiente Conda:**
   ```bash
   conda create -n leak_finder python=3.10
   conda activate leak_finder
   ```

### Passo 2: Instalar Dependências

1. **Instale as dependências principais:**
   ```bash
   cd main
   pip install -r requirements.txt
   ```

2. **Configure variáveis de ambiente (Windows):**
   ```bash
   set OPENBLAS_NUM_THREADS=1
   ```

### Passo 3: Configurar Hardware

1. **Instale os drivers do UMA-16**:
   - Baixe os drivers em: https://www.minidsp.com/userdownloads/usb-mic-array-series/uma16-drivers
   - Execute o instalador e siga as instruções
   - Reinicie o computador após a instalação

2. **Conecte o miniDSP UMA-16** via USB
3. **Conecte a webcam** via USB
4. **Configure o UMA-16 como dispositivo de entrada padrão**:
   - Painel de Controle → Som → Dispositivos de Gravação
   - Selecione "miniDSP UMA-16" como padrão

### Passo 4: Executar a Aplicação

```bash
python acoustic_camera.py
```

## 🎮 Como Usar

### Interface Principal

A aplicação abre com uma interface dividida em:

1. **Área da Câmera**: Mostra o feed da webcam com overlay de intensidade sonora
2. **Painel de Controles**: Sliders e botões para ajustar parâmetros
3. **Gráficos**: Visualização em tempo real do espectro de frequências

### Controles Principais

#### Filtros de Frequência:
- **High Pass (HP)**: Filtra frequências baixas (padrão: 800 Hz)
- **Low Pass (LP)**: Filtra frequências altas (padrão: 16000 Hz)

#### Controles de Processamento:
- **Smoothing**: Suavização do mapa de intensidade
- **Sensitivity**: Sensibilidade da detecção
- **Threshold**: Limiar mínimo para detecção

#### Funcionalidades:
- **Pause/Resume**: Pausa o processamento
- **Fullscreen**: Modo tela cheia da câmera
- **Save Frame**: Salva frame atual com overlay
- **Air Leak Mode**: Modo otimizado para detecção de vazamentos

### Atalhos de Teclado:
- **Q**: Sair da aplicação
- **F**: Alternar modo tela cheia
- **S**: Salvar frame atual
- **P**: Pausar/retomar processamento

## 🔍 Detecção de Vazamentos

### Modo Air Leak:
1. Ative o "Air Leak Mode" no painel de controles
2. Ajuste os filtros para a faixa de frequência do vazamento:
   - **Vazamentos de ar**: 2-8 kHz
   - **Vazamentos de gás**: 1-4 kHz
   - **Vazamentos de água**: 500 Hz - 2 kHz

### Dicas para Melhor Detecção:
- Mantenha o array a 30-50 cm da superfície
- Use ambiente silencioso
- Ajuste a sensibilidade conforme necessário
- Monitore o gráfico de frequências para otimização

## ⚙️ Configurações Avançadas

### Parâmetros Técnicos:
- **Sample Rate**: 48000 Hz (padrão UMA-16)
- **Block Size**: 4096 samples
- **Mic Spacing**: 42 mm
- **Sound Speed**: 343 m/s

### Ajustes de Performance:
- Reduza `BLOCK_SIZE` para menor latência
- Aumente `SMOOTHING_FACTOR` para suavização
- Ajuste `FREQUENCY_RANGE` conforme aplicação

## 🛠️ Solução de Problemas

### Problemas Comuns:

1. **Comando 'conda' não encontrado:**
   ```
   Solução: 
   - Instale o Miniconda: https://docs.conda.io/en/latest/miniconda.html
   - Reinicie o terminal após a instalação
   - Verifique com: conda --version
   ```

2. **Erro de dispositivo de áudio:**
   ```
   Solução: Verifique se UMA-16 está conectado e selecionado como dispositivo padrão
   ```

3. **UMA-16 não é reconhecido pelo Windows:**
   ```
   Solução: 
   - Instale os drivers oficiais: https://www.minidsp.com/userdownloads/usb-mic-array-series/uma16-drivers
   - Reinicie o computador após a instalação
   - Conecte o dispositivo após reiniciar
   ```

4. **Erro de canais de áudio:**
   ```
   Solução: Configure o UMA-16 para 16 canais em 48kHz
   ```

5. **Performance lenta:**
   ```
   Solução: Reduza BLOCK_SIZE ou frequência de atualização
   ```

6. **Erro de dependências:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

### Logs e Debug:
- Verifique a saída do console para mensagens de erro
- Use `sd.query_devices()` para listar dispositivos de áudio
- Teste com `sd.test_input_devices()` para verificar microfones

## 📊 Especificações Técnicas

### Algoritmos Implementados:
- **Delay-and-Sum Beamforming**: Localização básica de fontes
- **Adaptive Beamforming**: Melhor resolução angular
- **Frequency Domain Processing**: Análise espectral em tempo real
- **Spatial Filtering**: Redução de ruído e interferência

### Performance:
- **Latência**: < 100ms (dependendo da configuração)
- **Resolução Angular**: ~15° (configurável)
- **Faixa de Frequência**: 20 Hz - 24 kHz
- **Taxa de Amostragem**: 48 kHz

## 🤝 Contribuição

Para contribuir com o projeto:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para detalhes.

## 🙏 Créditos

- **Acoular**: Biblioteca de beamforming acústico
- **OpenCV**: Processamento de vídeo e visão computacional
- **PyQt5**: Interface gráfica
- **SciPy**: Processamento de sinais
- **NumPy**: Computação numérica

## 📞 Suporte

Para suporte técnico ou dúvidas:
- Abra uma issue no GitHub
- Consulte a documentação do Acoular
- Verifique os logs de erro no console

---

**Desenvolvido para detecção eficiente de vazamentos e análise acústica em tempo real.** 
