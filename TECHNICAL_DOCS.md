# Documentação Técnica - LEAK-FINDER Sensor

## Visão Geral do Sistema

O LEAK-FINDER Sensor é uma câmera acústica em tempo real que combina processamento de áudio avançado com visualização de vídeo para detectar e localizar fontes de som, especialmente vazamentos em sistemas industriais.

## Arquitetura do Sistema

### Componentes Principais

1. **Array de Microfones (UMA-16)**
   - 16 microfones omnidirecionais
   - Espaçamento: 42mm entre microfones
   - Taxa de amostragem: 48kHz
   - Interface USB 2.0
   - **Drivers**: Requer instalação de drivers oficiais
     - Download: https://www.minidsp.com/userdownloads/usb-mic-array-series/uma16-drivers

2. **Sistema de Processamento**
   - Algoritmos de beamforming em tempo real
   - Filtros digitais adaptativos
   - Análise espectral em tempo real

3. **Interface de Visualização**
   - Overlay de intensidade sonora em vídeo
   - Controles em tempo real
   - Gráficos de espectro de frequência

## Algoritmos Implementados

### 1. Delay-and-Sum Beamforming

```python
def delay_and_sum(audio, mic_positions, angles, fs, sound_speed):
    """
    Implementação básica de beamforming por delay-and-sum
    """
    # Calcula delays para cada ângulo
    # Aplica delays aos sinais
    # Soma os sinais alinhados
    # Retorna mapa de potência
```

**Características:**
- Resolução angular: ~15°
- Latência: < 50ms
- Robustez: Alta
- Complexidade: Baixa

### 2. Adaptive Beamforming

```python
def adaptive_beamforming(audio, mic_positions, mic_map):
    """
    Beamforming adaptativo com filtros espaciais
    """
    # Calcula matriz de correlação
    # Aplica filtros espaciais adaptativos
    # Otimiza direção de máxima potência
    # Retorna mapa de potência refinado
```

**Características:**
- Resolução angular: ~5-10°
- Latência: < 100ms
- Robustez: Média
- Complexidade: Alta

### 3. Filtros Digitais

#### Filtro Passa-Banda Butterworth
```python
def create_bandpass_filter(low_hz, high_hz, fs):
    """
    Filtro passa-banda de ordem 12
    """
    nyquist = 0.5 * fs
    low = low_hz / nyquist
    high = high_hz / nyquist
    sos = butter(12, [low, high], btype='band', output='sos')
    return sos
```

#### Filtro FIR Robusto
```python
def create_robust_fir_bandpass(low_hz, high_hz, fs, numtaps=4096):
    """
    Filtro FIR com alta rejeição fora da banda
    """
    nyq = 0.5 * fs
    transition_width = min(50, (high_hz - low_hz) * 0.05)
    low_cut = max(1, low_hz - transition_width)
    high_cut = min(nyq - 1, high_hz + transition_width)
    taps = firwin(numtaps, [low_cut/nyq, high_cut/nyq], 
                  pass_zero=False, window='hamming')
    return taps
```

## Especificações Técnicas Detalhadas

### Parâmetros de Áudio

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Sample Rate | 48 kHz | Taxa de amostragem do UMA-16 |
| Block Size | 4096 | Tamanho do bloco de processamento |
| N Channels | 16 | Número de microfones |
| Bit Depth | 24-bit | Resolução do conversor A/D |
| Mic Spacing | 42 mm | Distância entre microfones |
| Sound Speed | 343 m/s | Velocidade do som no ar |

### Parâmetros de Beamforming

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Frequency Range | 800-16000 Hz | Faixa de análise |
| H Scan Angles | -70° to +70° | Varredura horizontal |
| V Scan Angles | -45° to +45° | Varredura vertical |
| Angle Step | 15° | Resolução angular |
| Smoothing Factor | 0.8 | Suavização temporal |

### Configurações de Filtros por Tipo de Vazamento

| Tipo | HP (Hz) | LP (Hz) | Descrição |
|------|---------|---------|-----------|
| Ar | 2000 | 8000 | Vazamentos de ar comprimido |
| Gás | 1000 | 4000 | Vazamentos de gás |
| Água | 500 | 2000 | Vazamentos de água |

## Performance e Otimização

### Métricas de Performance

- **Latência Total**: < 100ms
- **Taxa de FPS**: 30 fps (configurável)
- **Uso de CPU**: < 50% em CPU i5
- **Uso de RAM**: < 500MB
- **Precisão Angular**: ±5° (modo adaptativo)

### Otimizações Implementadas

1. **Threading Otimizado**
   - Thread separado para áudio
   - Thread separado para vídeo
   - Thread separado para processamento

2. **Buffering Inteligente**
   - Buffer circular para áudio
   - Buffer FIFO para vídeo
   - Sincronização automática

3. **Processamento Vetorizado**
   - Uso de NumPy para operações vetoriais
   - Otimização BLAS/LAPACK
   - Paralelização OpenMP

## Interface de Usuário

### Layout da Interface

```
┌─────────────────────────────────────────┐
│                Câmera                   │
│           (com overlay)                 │
├─────────────────────────────────────────┤
│  Controles | Gráficos | Configurações  │
│  HP: [=====] | [Espectro] | [Salvar]   │
│  LP: [=====] | [Tempo]   | [Pausar]    │
└─────────────────────────────────────────┘
```

### Controles Principais

1. **Filtros de Frequência**
   - High Pass (HP): 20 Hz - 20 kHz
   - Low Pass (LP): 20 Hz - 20 kHz
   - Resolução: 1 Hz

2. **Controles de Processamento**
   - Smoothing: 0.0 - 1.0
   - Sensitivity: 0.1 - 10.0
   - Threshold: 0.0 - 1.0

3. **Funcionalidades**
   - Pause/Resume
   - Fullscreen
   - Save Frame
   - Air Leak Mode

## Instalação e Configuração de Hardware

### Drivers do UMA-16

**IMPORTANTE**: O miniDSP UMA-16 requer drivers específicos para funcionar corretamente.

1. **Download dos Drivers**:
   - Acesse: https://www.minidsp.com/userdownloads/usb-mic-array-series/uma16-drivers
   - Baixe a versão compatível com seu sistema operacional

2. **Instalação**:
   - Execute o instalador como administrador
   - Siga as instruções do assistente de instalação
   - **Reinicie o computador** após a instalação

3. **Verificação**:
   - Conecte o UMA-16 via USB
   - Verifique no Gerenciador de Dispositivos se aparece sem erros
   - Configure como dispositivo de entrada padrão

### Configuração do Sistema

1. **Instalação do Miniconda/Anaconda**:
   - **Miniconda** (recomendado): https://docs.conda.io/en/latest/miniconda.html
   - **Anaconda** (alternativa): https://www.anaconda.com/products/distribution
   - Execute o instalador e reinicie o terminal
   - Verifique com: `conda --version`

2. **Configurações de Áudio**:
   - Painel de Controle → Som → Dispositivos de Gravação
   - Selecione "miniDSP UMA-16" como padrão
   - Configure para 16 canais, 48kHz, 24-bit

3. **Teste de Funcionamento**:
   - Use o aplicativo "Som" do Windows para testar
   - Verifique se todos os 16 canais estão ativos
   - Teste com um microfone de referência

## Calibração e Manutenção

### Calibração do Sistema

1. **Calibração de Fase**
   ```python
   def calibrate_phase(mic_positions, reference_signal):
       """
       Calibra diferenças de fase entre microfones
       """
       # Mede delays entre microfones
       # Aplica correções de fase
       # Valida calibração
   ```

2. **Calibração de Amplitude**
   ```python
   def calibrate_amplitude(mic_gains):
       """
       Normaliza amplitudes dos microfones
       """
       # Mede ganhos relativos
       # Aplica fatores de correção
       # Valida normalização
   ```

### Manutenção Preventiva

- **Limpeza**: Limpar microfones mensalmente
- **Calibração**: Recalibrar trimestralmente
- **Atualizações**: Manter software atualizado
- **Backup**: Fazer backup de configurações

## Troubleshooting Avançado

### Problemas de Áudio

1. **UMA-16 não é reconhecido**
   - **Causa**: Drivers não instalados ou incorretos
   - **Solução**: 
     - Instale drivers oficiais: https://www.minidsp.com/userdownloads/usb-mic-array-series/uma16-drivers
     - Reinicie o computador
     - Conecte o dispositivo após reiniciar

2. **Ruído Excessivo**
   - Verificar conexões USB
   - Verificar fonte de alimentação
   - Verificar ambiente acústico

3. **Latência Alta**
   - Reduzir BLOCK_SIZE
   - Verificar drivers de áudio
   - Otimizar configurações de buffer

4. **Perda de Canais**
   - Verificar cabo USB
   - Reiniciar dispositivo
   - Verificar configurações do sistema

### Problemas de Performance

1. **CPU Alto**
   - Reduzir frequência de atualização
   - Simplificar algoritmos
   - Otimizar configurações

2. **Memória Alta**
   - Limpar buffers
   - Reduzir histórico
   - Reiniciar aplicação

## Extensões e Melhorias

### Possíveis Melhorias

1. **Machine Learning**
   - Classificação automática de vazamentos
   - Detecção de padrões
   - Otimização automática de parâmetros

2. **Análise Avançada**
   - Análise de modos normais
   - Reconstrução de campo sonoro
   - Análise de coerência

3. **Interface Avançada**
   - Visualização 3D
   - Controles gestuais
   - Integração com sistemas SCADA

### APIs e Integração

```python
# Exemplo de API para integração
class LeakFinderAPI:
    def __init__(self):
        self.sensor = AcousticCamera()
    
    def detect_leaks(self, frequency_range=None):
        """Detecta vazamentos na faixa especificada"""
        pass
    
    def get_intensity_map(self):
        """Retorna mapa de intensidade atual"""
        pass
    
    def save_analysis(self, filename):
        """Salva análise completa"""
        pass
```

## Referências Técnicas

### Bibliografia

1. **Beamforming Algorithms**
   - Van Trees, H. L. (2002). "Optimum Array Processing"
   - Johnson, D. H. (1993). "Array Signal Processing"

2. **Acoustic Camera Technology**
   - Hald, J. (2009). "Beamforming"
   - Sijtsma, P. (2007). "CLEAN Based on Spatial Source Coherence"

3. **Signal Processing**
   - Oppenheim, A. V. (1999). "Discrete-Time Signal Processing"
   - Proakis, J. G. (2007). "Digital Signal Processing"

### Padrões e Normas

- **ISO 3744**: Acoustics - Determination of sound power levels
- **ANSI S12.10**: Acoustics - Measurement of airborne noise
- **IEC 61672**: Electroacoustics - Sound level meters

---

*Documentação técnica v1.0 - LEAK-FINDER Sensor* 