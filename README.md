# 🎯 LEAK-FINDER-sensor

<div align="center">

![Acoustic Camera Banner](frequencyBanner.jpg)

*Uma câmera acústica 2D para detecção e visualização de vazamentos*

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## 📋 Índice
- [Sobre o Projeto](#-sobre-o-projeto)
- [Funcionalidades](#-funcionalidades)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Interface do Usuário](#-interface-do-usuário)
- [Solução de Problemas](#-solução-de-problemas)
- [Configurações Avançadas](#-configurações-avançadas)
- [Contribuição](#-contribuição)

## 🎯 Sobre o Projeto
O LEAK-FINDER-sensor é uma câmera acústica 2D projetada para detectar e visualizar vazamentos através da combinação de processamento de áudio e vídeo em tempo real. O sistema utiliza um array de microfones UMA-16 para captura de áudio e uma câmera para sobreposição visual dos resultados.

## ✨ Funcionalidades
- 🎤 Captura de áudio multicanal com array UMA-16
- 📹 Integração com câmera em tempo real
- 🌡️ Visualização de beamforming em tempo real
- 🎨 Interface gráfica intuitiva com modo claro/escuro
- 📊 Controles de intensidade e visualização
- 💾 Captura de imagens e gravação de vídeo

## 🔧 Requisitos

### Hardware
- Array de microfones UMA-16
- Câmera USB ou integrada
- Computador com Windows/Linux/macOS

### Software
| Dependência | Versão | Finalidade |
|-------------|---------|------------|
| Python | ≥ 3.8 | Ambiente de execução |
| numpy | Última versão | Processamento numérico |
| scipy | Última versão | Processamento de sinais |
| sounddevice | Última versão | Captura de áudio |
| opencv-python | Última versão | Processamento de vídeo |
| matplotlib | Última versão | Visualização de dados |
| Pillow | Última versão | Processamento de imagens |
| tkinter | Incluído no Python | Interface gráfica |

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/LEAK-FINDER-sensor.git
cd LEAK-FINDER-sensor
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Instalação do Tkinter (se necessário):

**Windows:**
- Já incluído na instalação padrão do Python

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3-tk
```

**macOS:**
```bash
brew install python-tk
```

## 💻 Como Usar

1. Execute o programa principal:
```bash
python detector.py
```

2. Verifique os microfones:
```bash
python mic_check.py
```

## 🖥️ Interface do Usuário

### Painéis
- **Central:** Visualização da câmera com sobreposição de beamforming
- **Esquerdo:** Controles de intensidade e gráfico
- **Direito:** Botões de controle

### Controles
- 🔄 Reiniciar câmera
- 📸 Salvar imagem
- 🎥 Gravar vídeo
- 🌓 Alternar tema (claro/escuro)
- 📊 Controles de frequência
- 🎞️ Botão de frames

## ❗ Solução de Problemas

### Problemas Comuns
1. **Erro de Tkinter:**
   - Verifique a instalação do Tkinter
   - Reinstale se necessário

2. **Microfones não detectados:**
   - Execute `mic_check.py`
   - Verifique as conexões USB
   - Confirme as configurações de áudio do sistema

## ⚙️ Configurações Avançadas

### Configurações de Áudio
- Taxa de amostragem: 48000 Hz
- Faixa de frequência: Configurável
- Canais: 16 (UMA-16)

### Configurações de Vídeo
- ID da câmera: Configurável
- Campo de visão (FOV): Ajustável
- Resolução: Adaptável

## 🤝 Contribuição

1. Faça um Fork do projeto
2. Crie sua Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a Branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

<div align="center">

**Desenvolvido com 💙 pela Equipe Diogo Marques e Dinis Barros**

</div>
