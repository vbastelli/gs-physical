# 🛡️ DefenseShield Orbital Intelligence
### Sistema de Monitoramento Inteligente com Visão Computacional

> **FIAP – Global Solution 2026 | Indústria Espacial**
> *"Transformando dados espaciais em inteligência para proteger o futuro."*

---

## 👥 Integrantes

| Nome | RM |
|------|----|
| Lorenzo Hayashi Mangini | 554901 |
| Victorio Bastelli | 554723 |
| Vitor Bebiano | 555026 |
| Milton Cezar | 555206 |

---

## 💡 Contexto e Motivação

A economia espacial global movimenta hoje **US$ 600 bilhões** e deve ultrapassar **US$ 1,8 trilhão até 2040**. Parte central dessa expansão envolve o uso de satélites, sensores remotos e sistemas autônomos para monitorar a Terra e, no futuro, bases lunares e missões interplanetárias.

Um dos maiores desafios tanto na Terra quanto no espaço é o mesmo: **como monitorar grandes instalações, infraestruturas críticas e ambientes hostis sem depender de operadores humanos 24 horas por dia?**

Governos, empresas de energia, defesa civil e futuras organizações espaciais precisam processar enormes volumes de dados visuais em tempo real — e qualquer atraso na detecção de uma anomalia pode gerar consequências operacionais, ambientais ou humanas graves.

---

## 🚀 O Projeto

O **DefenseShield Orbital Intelligence** é um sistema de **monitoramento inteligente em tempo real** que utiliza Visão Computacional e Inteligência Artificial para:

- Detectar **movimentos suspeitos** ou atividade incomum na área monitorada
- Identificar e classificar **pessoas e objetos** automaticamente
- Analisar a **postura corporal** de indivíduos (braços levantados, pessoa caída, etc.)
- Gerar **alertas automáticos** com log de eventos timestampado
- Exibir tudo isso em um **HUD (painel de controle visual)** em tempo real sobre o vídeo da câmera

O sistema roda localmente na câmera (edge computing), simulando o que seria um **nó sensor inteligente** dentro de uma rede maior que, em escala real, integraria satélites, drones e sensores IoT distribuídos.

---

## 🎯 Objetivos

### Objetivo de Negócio
Reduzir o tempo de resposta a incidentes críticos em instalações monitoradas — refinarias, aeroportos, usinas, bases militares ou futuras bases lunares — eliminando a necessidade de vigilância humana contínua e substituindo-a por **detecção automatizada orientada por IA**.

### Objetivo Técnico
Construir um pipeline de visão computacional robusto, modular e eficiente, capaz de:
- Processar vídeo em tempo real a taxas aceitáveis de FPS
- Combinar múltiplos modelos de IA (YOLO + MediaPipe) no mesmo loop de captura
- Tratar falhas de hardware sem derrubar o sistema
- Registrar eventos com precisão temporal para auditoria posterior

---

## 🔍 O que o Programa Faz — Passo a Passo

Ao iniciar, o script abre a webcam e entra em um loop contínuo de captura. Para cada frame capturado, **três módulos de análise rodam em paralelo**:

### Módulo 1 — Detecção de Movimento
Utiliza a técnica de **subtração de fundo (MOG2)** do OpenCV: o algoritmo aprende como é o cenário "em repouso" e, quando algo muda (uma pessoa entrando, um objeto sendo movido), calcula a área em pixels dessa variação.

- Se a área for pequena → **BAIXO** (ruído normal)
- Se ultrapassar o limiar configurado → **MÉDIO**
- Se for muito grande → **ALTO** + alerta no log

### Módulo 2 — Detecção de Objetos com YOLO
O modelo **YOLOv8n** (You Only Look Once, versão nano) analisa o frame inteiro e identifica o que está na cena em milissegundos. Para cada objeto detectado com confiança acima de 45%:

- Desenha uma **caixa delimitadora** colorida ao redor do objeto
- Exibe o **nome da classe** e o **percentual de confiança**
- Para objetos de interesse (pessoas, celulares, garrafas) → gera alerta no log

Se o YOLO não estiver disponível, o sistema cai automaticamente para o detector **HOG** do OpenCV (fallback para pessoas).

### Módulo 3 — Análise de Postura com MediaPipe Pose
O **MediaPipe** da Google mapeia 33 pontos do corpo humano (landmarks) — ombros, quadris, pulsos, joelhos, etc. — e o sistema analisa a geometria desses pontos para classificar a postura:

- **NORMAL** → postura comum, nenhum alerta
- **ARMS_UP** → pulsos acima dos ombros — possível sinal de emergência ou rendição
- **DOWN** → quadril muito próximo ao nível dos ombros — pessoa no chão

### HUD (Heads-Up Display)
Todos os resultados são renderizados sobre o vídeo em tempo real:

- **Painel superior**: nome do sistema + FPS atual (muda de cor conforme performance)
- **Painel inferior**: contagem de detecções, nível de movimento, estado de postura
- **Log de eventos**: últimos 8 alertas com horário exato
- **Mira central**: referência visual estilo sistema de monitoramento

Ao encerrar, o log completo da sessão é salvo automaticamente em arquivo `.log`.

---

## 🏗️ Arquitetura do Pipeline

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
│                                      ▼                          │
│                       ┌─────────────────────────┐              │
│                       │       HUD OVERLAY       │              │
│                       │  FPS | status | log     │              │
│                       └──────────────┬──────────┘              │
│                          ┌───────────┴──────────┐              │
│                          ▼                       ▼              │
│                    cv2.imshow()           logs/*.log            │
│                   (janela display)      (persistência)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Bibliotecas Utilizadas

| Biblioteca | Versão mínima | Para que serve no projeto |
|------------|--------------|--------------------------|
| `opencv-python` | 4.9+ | Captura da webcam, subtração de fundo (MOG2), desenho do HUD, detector HOG |
| `mediapipe` | 0.10+ | Detecção e análise de postura corporal (33 landmarks) |
| `ultralytics` | 8.1+ | Modelo YOLOv8n para detecção e classificação de objetos |
| `numpy` | 1.24+ | Operações vetoriais, cálculo de médias de FPS, manipulação de arrays de imagem |

> **Por que essas bibliotecas?**
> OpenCV é o padrão da indústria para visão computacional em Python. MediaPipe é mantido pela Google e oferece modelos leves e precisos para análise corporal. YOLO é o detector de objetos em tempo real mais usado no mundo — e a versão nano (yolov8n) foi escolhida por rodar bem mesmo em hardware sem GPU dedicada.

---

## ⚙️ Configurações do Sistema

As constantes no topo do `defenseshield.py` permitem ajustar o comportamento sem alterar a lógica:

| Constante | Padrão | O que controla |
|-----------|--------|----------------|
| `CAMERA_INDEX` | `0` | Índice da webcam (0 = principal) |
| `CONFIDENCE_THRESH` | `0.45` | Confiança mínima para o YOLO reportar um objeto |
| `MOTION_THRESHOLD` | `2500` | Área mínima em pixels² para classificar movimento como médio |
| `ALERT_COOLDOWN_SEC` | `3` | Tempo mínimo entre dois alertas do mesmo tipo (evita spam) |
| `YOLO_MODEL` | `yolov8n.pt` | Qual modelo YOLO usar (nano, small, medium...) |
| `MAX_LOG_LINES` | `8` | Quantas linhas de log exibir no HUD |

---

## ▶️ Como Executar

### Pré-requisitos
- **Python 3.10, 3.11 ou 3.12** (o MediaPipe ainda não suporta Python 3.13+)
- Webcam integrada ou USB
- Cerca de 500 MB livres (modelo YOLO baixado automaticamente)

### Passo 1 — Clonar o repositório
```bash
git clone https://github.com/<seu-usuario>/defenseshield-orbital.git
cd defenseshield-orbital
```

### Passo 2 — Criar ambiente virtual
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

> O ambiente virtual isola as dependências do projeto do resto do sistema. Sempre ative-o antes de rodar.

### Passo 3 — Instalar dependências
```bash
# Se o comando "pip" não for reconhecido (comum no Windows):
python -m pip install -r requirements.txt

# Ou normalmente:
pip install -r requirements.txt
```

> Na primeira execução, o YOLOv8n (~6 MB) será baixado automaticamente pela biblioteca Ultralytics.

### Passo 4 — Executar
```bash
python defenseshield.py
```

Uma janela abrirá mostrando o vídeo da webcam com o HUD sobreposto em tempo real.

### Controles durante a execução

| Tecla | Ação |
|-------|------|
| `Q` ou `ESC` | Encerra o sistema e salva o log |
| `S` | Salva um screenshot na pasta `screenshots/` |

---

## 📁 Estrutura do Repositório

```
defenseshield-orbital/
├── defenseshield.py     # Script principal — pipeline completo de visão computacional
├── requirements.txt     # Dependências com versões mínimas fixadas
├── README.md            # Este arquivo
├── logs/                # Criada automaticamente — logs de cada sessão (.log)
└── screenshots/         # Criada automaticamente — screenshots salvos com a tecla S
```

---

## 🔗 Conexão com a Economia Espacial

O projeto não é apenas um sistema de câmera inteligente — ele demonstra um princípio central da economia espacial: **resolver problemas extremos gera soluções aplicáveis na Terra**.

| Desafio no Espaço | Solução Desenvolvida | Aplicação Terrestre |
|-------------------|---------------------|---------------------|
| Monitorar base lunar sem humanos 24h | Detecção autônoma de anomalias | Segurança de infraestrutura crítica |
| Banda limitada Terra-Lua | Edge computing — processa localmente | Redução de custos de transmissão |
| Robôs precisam enxergar para agir | Visão computacional em tempo real | Automação industrial e logística |
| Ambiente hostil sem operador presente | Sistema resiliente a falhas de hardware | Monitoramento em áreas remotas |

---

## 📄 Licença

Projeto acadêmico desenvolvido para a FIAP – Global Solution 2026.
Uso restrito a fins educacionais.
