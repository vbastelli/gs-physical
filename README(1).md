# 🛡️ DefenseShield Orbital Intelligence

> **FIAP – Global Solution 2026 | Indústria Espacial**
>
> *"Transformando dados espaciais em inteligência para proteger o futuro."*

---

## 👥 Integrantes

| Nome | RM |
|------|-----|
| Lorenzo Hayashi Mangini | — |
| Victorio Bastelli | — |
| Vitor Bebiano | — |
| Milton Cezar | — |

---

## 📋 Descrição do Projeto

O **DefenseShield Orbital Intelligence** é uma plataforma inteligente de monitoramento em tempo real que utiliza **Visão Computacional** para detectar anomalias, identificar riscos operacionais e auxiliar na tomada de decisão.

O sistema foi inspirado no contexto da **economia espacial**: tecnologias desenvolvidas para ambientes extremos (bases lunares, estações orbitais) que geram benefícios diretos na Terra — monitoramento de infraestrutura crítica, defesa civil, segurança ambiental e gestão de riscos.

O pipeline de visão computacional via webcam simula um nó sensor de uma rede maior que, em escala real, integraria satélites, drones e sensores IoT distribuídos.

---

## 🎯 Objetivo da Solução

### Objetivo de Negócio
Reduzir o tempo de resposta a eventos críticos em instalações monitoradas — seja uma refinaria, aeroporto, usina ou futura base lunar — por meio de **detecção automática de anomalias** sem depender de operadores humanos vigilantes 24/7.

### Objetivo Técnico
Construir um **pipeline de inferência em tempo real** rodando localmente (edge computing), capaz de:
- Detectar pessoas e objetos relevantes com YOLO
- Analisar posturas corporais suspeitas com MediaPipe Pose
- Identificar movimentos abruptos com subtração de fundo
- Exibir um HUD informativo com FPS, log de alertas e status do sistema
- Operar de forma resiliente a falhas de hardware (webcam, frames corrompidos)

---

## 🏗️ Arquitetura da Solução

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEFENSESHIELD PIPELINE                       │
│                                                                 │
│  ┌──────────┐    ┌────────────────────────────────────────┐     │
│  │  WEBCAM  │───▶│           PRÉ-PROCESSAMENTO            │     │
│  │ (sensor) │    │   BGR → RGB  |  resize  |  buffer=1    │     │
│  └──────────┘    └──────────────────┬─────────────────────┘     │
│                                     │                           │
│               ┌─────────────────────┼──────────────────┐        │
│               ▼                     ▼                  ▼        │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐   │
│  │  MÓDULO 1       │  │   MÓDULO 2       │  │  MÓDULO 3    │   │
│  │  Movimento      │  │   Detecção de    │  │  Pose        │   │
│  │  (MOG2 + morph) │  │   Objetos        │  │  (MediaPipe) │   │
│  │                 │  │   YOLOv8n / HOG  │  │              │   │
│  │ → área px²      │  │   → bounding box │  │ → NORMAL     │   │
│  │ → BAIXO/MED/ALT │  │   → classe+conf  │  │ → ARMS_UP    │   │
│  └────────┬────────┘  └────────┬─────────┘  │ → DOWN       │   │
│           │                    │             └──────┬───────┘   │
│           └────────────────────┴────────────────────┘           │
│                                     │                           │
│                                     ▼                           │
│                       ┌─────────────────────────┐              │
│                       │    ALERT MANAGER        │              │
│                       │  cooldown + log deque   │              │
│                       └──────────────┬──────────┘              │
│                                      │                          │
│                                      ▼                          │
│                       ┌─────────────────────────┐              │
│                       │       HUD OVERLAY       │              │
│                       │  FPS | status | log     │              │
│                       └──────────────┬──────────┘              │
│                                      │                          │
│                          ┌───────────┴──────────┐              │
│                          ▼                       ▼              │
│                    cv2.imshow()           logs/*.log            │
│                   (janela display)      (persistência)          │
└─────────────────────────────────────────────────────────────────┘
```

### Módulos do Pipeline

| Módulo | Tecnologia | Função |
|--------|-----------|--------|
| Detecção de Movimento | OpenCV MOG2 + Morfologia | Identifica regiões com variação temporal acima do limiar |
| Detecção de Objetos | YOLOv8n (Ultralytics) ou HOG fallback | Localiza e classifica pessoas e objetos relevantes |
| Análise de Pose | MediaPipe Pose (model_complexity=0) | Interpreta postura corporal e detecta sinais de alerta |
| Alert Manager | Python `collections.deque` + cooldown | Agrega eventos com anti-spam e persiste log de sessão |
| HUD Overlay | OpenCV drawing + alpha blend | Exibe métricas, log e status em tempo real sobre o vídeo |

---

## 🛠️ Stack Tecnológica

```
Python 3.10+
├── opencv-python      4.9+    → captura, pré-proc., HUD, HOG fallback
├── mediapipe          0.10+   → análise de pose (landmarks 3D)
├── ultralytics        8.1+    → YOLOv8n (detecção de objetos)
└── numpy              1.24+   → operações vetoriais e métricas
```

---

## ▶️ Instruções de Execução

### 1. Pré-requisitos

- Python **3.10** ou superior
- Webcam USB ou integrada
- ~2 GB de espaço (modelo YOLO baixado automaticamente na 1ª execução)

### 2. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/defenseshield-orbital.git
cd defenseshield-orbital
```

### 3. Criar e ativar ambiente virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar dependências

```bash
pip install -r requirements.txt
```

> ⚠️ O modelo `yolov8n.pt` (~6 MB) é baixado automaticamente pelo Ultralytics na primeira execução.

### 5. Executar

```bash
python defenseshield.py
```

### Controles durante execução

| Tecla | Ação |
|-------|------|
| `Q` ou `ESC` | Encerra o sistema |
| `S` | Salva screenshot na pasta `screenshots/` |

---

## 📁 Estrutura do Repositório

```
defenseshield-orbital/
├── defenseshield.py        # Script principal (pipeline completo)
├── requirements.txt        # Dependências do projeto
├── README.md               # Este arquivo
├── logs/                   # Logs de sessão (gerados em runtime)
└── screenshots/            # Screenshots salvos com tecla S
```

---

## ⚙️ Configurações Principais

As constantes no início de `defenseshield.py` permitem ajuste sem alterar a lógica:

| Constante | Padrão | Descrição |
|-----------|--------|-----------|
| `CAMERA_INDEX` | `0` | Índice da webcam |
| `CONFIDENCE_THRESH` | `0.45` | Limiar de confiança YOLO |
| `MOTION_THRESHOLD` | `2500` | Área mínima (px²) para alerta de movimento |
| `ALERT_COOLDOWN_SEC` | `3` | Intervalo mínimo entre alertas do mesmo tipo |
| `YOLO_MODEL` | `yolov8n.pt` | Modelo YOLO utilizado |

---

## 🔗 Relação com a Economia Espacial

O projeto demonstra como tecnologias desenvolvidas para **ambientes extremos** (base lunar, estações orbitais) se traduzem em soluções terrestres:

- **Monitoramento autônomo** → necessidade em bases sem presença humana constante
- **Edge computing resiliente** → restrições de banda Terra-Lua exigem processamento local
- **Detecção de anomalias** → falhas em sistemas pressurizados demandam resposta imediata
- **Visão computacional** → robôs espaciais dependem de percepção visual para operar

---

## 📄 Licença

Projeto acadêmico desenvolvido para a FIAP – Global Solution 2026.
