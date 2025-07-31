#!/usr/bin/env python3
"""
Configurações do LEAK-FINDER Sensor
Câmera Acústica de Baixo Custo
"""

import os

# =============================================================================
# CONFIGURAÇÕES DE ÁUDIO
# =============================================================================

# Configurações do UMA-16
SAMPLE_RATE = 48000  # Taxa de amostragem padrão do UMA-16
BLOCK_SIZE = 4096    # Tamanho do bloco de áudio (maior precisão para voz)
N_MICS = 16          # Número de microfones no UMA-16
AUDIO_DEVICE = None  # None para dispositivo padrão, ou índice específico

# Configurações físicas
MIC_SPACING = 0.042   # Espaçamento entre microfones em metros (42 mm)
SOUND_SPEED = 343.0   # Velocidade do som no ar (m/s)
NYQUIST = SAMPLE_RATE // 2 - 1  # Frequência de Nyquist

# =============================================================================
# CONFIGURAÇÕES DE BEAMFORMING
# =============================================================================

# Faixa de frequência para análise
FREQUENCY_RANGE = [800, 16000]  # Hz

# Ângulos de varredura (horizontal e vertical)
H_SCAN_ANGLES = (-70, 71, 15)   # Horizontal: -70° a +70° em passos de 15°
V_SCAN_ANGLES = (-45, 46, 15)   # Vertical: -45° a +45° em passos de 15°

# Fator de suavização
SMOOTHING_FACTOR = 0.8

# =============================================================================
# CONFIGURAÇÕES DE FILTROS
# =============================================================================

# Filtros padrão
DEFAULT_HP_FREQ = 800    # Frequência do filtro passa-alta (Hz)
DEFAULT_LP_FREQ = 16000  # Frequência do filtro passa-baixa (Hz)

# Configurações de filtros específicos para vazamentos
LEAK_FILTERS = {
    'air': {
        'hp': 2000,    # 2 kHz
        'lp': 8000,    # 8 kHz
        'description': 'Vazamentos de ar'
    },
    'gas': {
        'hp': 1000,    # 1 kHz
        'lp': 4000,    # 4 kHz
        'description': 'Vazamentos de gás'
    },
    'water': {
        'hp': 500,     # 500 Hz
        'lp': 2000,    # 2 kHz
        'description': 'Vazamentos de água'
    }
}

# =============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# =============================================================================

# Configurações da câmera
CAM_INDEX = 0  # Índice da webcam padrão

# Configurações de interface
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Configurações de cores
HEATMAP_COLORMAP = 'jet'  # Colormap para o heatmap
INTENSITY_THRESHOLD = 0.1  # Limiar mínimo para visualização

# =============================================================================
# CONFIGURAÇÕES DE PERFORMANCE
# =============================================================================

# Configurações de threading
AUDIO_THREAD_PRIORITY = 'high'
VIDEO_THREAD_PRIORITY = 'normal'

# Configurações de buffer
AUDIO_BUFFER_SIZE = 8192
VIDEO_BUFFER_SIZE = 10

# =============================================================================
# CONFIGURAÇÕES DE ARQUIVO
# =============================================================================

# Diretórios
SAVED_FRAMES_DIR = 'saved_frames'
LOGS_DIR = 'logs'

# Configurações de salvamento
AUTO_SAVE_INTERVAL = 0  # 0 = desabilitado, >0 = intervalo em segundos
SAVE_FORMAT = 'png'     # Formato das imagens salvas

# =============================================================================
# CONFIGURAÇÕES DE DEBUG
# =============================================================================

# Logging
DEBUG_MODE = False
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR

# Configurações de teste
TEST_MODE = False
MOCK_AUDIO = False

# =============================================================================
# FUNÇÕES DE CONFIGURAÇÃO
# =============================================================================

def get_leak_filter(leak_type):
    """Retorna configurações de filtro para tipo específico de vazamento"""
    return LEAK_FILTERS.get(leak_type, LEAK_FILTERS['air'])

def get_audio_config():
    """Retorna configurações de áudio"""
    return {
        'sample_rate': SAMPLE_RATE,
        'block_size': BLOCK_SIZE,
        'n_mics': N_MICS,
        'device': AUDIO_DEVICE,
        'mic_spacing': MIC_SPACING,
        'sound_speed': SOUND_SPEED
    }

def get_beamforming_config():
    """Retorna configurações de beamforming"""
    return {
        'frequency_range': FREQUENCY_RANGE,
        'h_scan_angles': H_SCAN_ANGLES,
        'v_scan_angles': V_SCAN_ANGLES,
        'smoothing_factor': SMOOTHING_FACTOR
    }

def get_visualization_config():
    """Retorna configurações de visualização"""
    return {
        'window_width': WINDOW_WIDTH,
        'window_height': WINDOW_HEIGHT,
        'camera_width': CAMERA_WIDTH,
        'camera_height': CAMERA_HEIGHT,
        'colormap': HEATMAP_COLORMAP,
        'threshold': INTENSITY_THRESHOLD
    }

# =============================================================================
# CONFIGURAÇÕES DE AMBIENTE
# =============================================================================

def setup_environment():
    """Configura variáveis de ambiente necessárias"""
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Criar diretórios se não existirem
    for directory in [SAVED_FRAMES_DIR, LOGS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

# =============================================================================
# VALIDAÇÃO DE CONFIGURAÇÕES
# =============================================================================

def validate_config():
    """Valida as configurações do sistema"""
    errors = []
    
    # Validar configurações de áudio
    if SAMPLE_RATE <= 0:
        errors.append("SAMPLE_RATE deve ser positivo")
    
    if BLOCK_SIZE <= 0:
        errors.append("BLOCK_SIZE deve ser positivo")
    
    if N_MICS <= 0:
        errors.append("N_MICS deve ser positivo")
    
    # Validar configurações de beamforming
    if FREQUENCY_RANGE[0] >= FREQUENCY_RANGE[1]:
        errors.append("FREQUENCY_RANGE inválida")
    
    if FREQUENCY_RANGE[1] > NYQUIST:
        errors.append(f"FREQUENCY_RANGE máxima excede Nyquist ({NYQUIST} Hz)")
    
    # Validar configurações de filtros
    if DEFAULT_HP_FREQ >= DEFAULT_LP_FREQ:
        errors.append("Frequências de filtro inválidas")
    
    if errors:
        raise ValueError(f"Erros de configuração: {'; '.join(errors)}")
    
    return True 