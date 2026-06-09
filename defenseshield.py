"""
DefenseShield Orbital Intelligence  —  VERSÃO FINAL CORRIGIDA
====================================
Sistema de monitoramento inteligente com visão computacional.
Utiliza YOLO para detecção de objetos/pessoas, MediaPipe para
análise de pose e movimentos suspeitos, e OpenCV para o pipeline
de captura e exibição em tempo real.

Integrantes:
  - Lorenzo Hayashi Mangini
  - Victorio Bastelli
  - Vitor Bebiano
  - Milton Cezar

FIAP – Global Solution 2026 – Indústria Espacial

"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import time
import datetime
import os
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[AVISO] ultralytics não encontrado. Usando detector HOG como fallback.")


# ══════════════════════════════════════════════
#  CONFIGURAÇÕES GLOBAIS
# ══════════════════════════════════════════════
CAMERA_INDEX        = 0
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
CONFIDENCE_THRESH   = 0.45
MAX_LOG_LINES       = 8
ALERT_COOLDOWN_SEC  = 3
MOTION_THRESHOLD    = 2500
LOG_DIR             = "logs"
YOLO_MODEL          = "yolov8n.pt"

# ── Skip de frames ────────────────────────────
# YOLO_SKIP_FRAMES : executa YOLO a cada N frames
# POSE_SKIP_FRAMES : executa Pose a cada N frames (Pose é computacionalmente pesado)
YOLO_SKIP_FRAMES    = 4
POSE_SKIP_FRAMES    = 3 

# ── Resoluções de inferência ──────────────────
# Frames redimensionados antes de passar aos modelos para reduzir custo computacional
YOLO_INFER_WIDTH    = 320
YOLO_INFER_HEIGHT   = 256
POSE_INFER_WIDTH    = 256
POSE_INFER_HEIGHT   = 192

import torch
YOLO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════
#  CLASSES ALVO
# ══════════════════════════════════════════════
ALERT_CLASSES = {
    # Pessoas
    0:  ("Pessoa detectada",        (0, 255, 100)),

    # Eletrônicos
    63: ("Notebook monitorado",     (255, 180, 0)),
    64: ("Mouse monitorado",        (255, 140, 0)),
    66: ("Teclado monitorado",      (255, 220, 0)),
    67: ("Celular monitorado",      (0, 200, 255)),

    # Objetos comuns
    39: ("Objeto monitorado",       (0, 100, 255)),
    73: ("Livro monitorado",        (180, 255, 0)),
    74: ("Relogio monitorado",      (255, 0, 255)),

    # Itens que costumam chamar atenção
    41: ("Copo monitorado",         (255, 120, 120)),
    76: ("Tesoura monitorada",      (0, 255, 255)),
}

COLOR_ACCENT  = (0,  210, 120)
COLOR_WARN    = (0,  140, 255)
COLOR_DANGER  = (0,   50, 220)
COLOR_TEXT    = (200, 220, 240)
COLOR_DIM     = (100, 110, 130)


# ══════════════════════════════════════════════
#  CLASSE: GERENCIADOR DE ALERTAS
# ══════════════════════════════════════════════
class AlertManager:
    """
    Gerencia o log de alertas com cooldown por tipo de evento,
    evitando spam de mensagens repetidas no painel.
    """

    def __init__(self, cooldown_sec: float = ALERT_COOLDOWN_SEC):
        self.cooldown   = cooldown_sec
        self._last_time = {}
        self.log        = deque(maxlen=MAX_LOG_LINES)

    def trigger(self, alert_type: str, message: str) -> bool:
        """
        Registra um alerta se o cooldown do tipo já expirou.
        Retorna True se o alerta foi registrado, False se suprimido.
        """
        now  = time.time()
        last = self._last_time.get(alert_type, 0)
        if now - last >= self.cooldown:
            self._last_time[alert_type] = now
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.log.appendleft(f"[{ts}] {message}")
            return True
        return False

    def get_log(self):
        return list(self.log)


# ══════════════════════════════════════════════
#  CLASSE: DETECTOR DE MOVIMENTO
# ══════════════════════════════════════════════
class MotionDetector:
    """
    Subtrator de fundo MOG2 para detecção de movimento.
    Recebe frames em resolução reduzida para maior eficiência.
    """

    def __init__(self):
        # history=200: janela temporal para calibrar o fundo
        # varThreshold=50: sensibilidade à variação de pixel
        # detectShadows=False: desativa detecção de sombras (mais rápido)
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray):
        """
        Aplica subtração de fundo e morfologia para remover ruído.
        Retorna: (area_total_de_movimento, contornos, máscara binária)
        """
        mask = self.subtractor.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 300)
        return motion_area, contours, mask


# ══════════════════════════════════════════════
#  CLASSE: ANALISADOR DE POSE (MediaPipe)
# ══════════════════════════════════════════════
class PoseAnalyzer:
    """
    Wrapper do MediaPipe Pose para análise de postura humana.
    Detecta estados: NORMAL, ARMS_UP (braços levantados) e
    FIGHT_GUARD (postura defensiva com punhos próximos ao rosto).
    """

    def __init__(self):
        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        # model_complexity=0: modelo leve (menor acurácia, maior FPS)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame_rgb: np.ndarray):
        """
        Processa um frame RGB e retorna (results, estado_detectado).
        Estado pode ser: "NORMAL", "ARMS_UP", "FIGHT_GUARD".

        """
        results = self.pose.process(frame_rgb)
        state   = "NORMAL"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            PL = self.mp_pose.PoseLandmark

            l_shoulder = lm[PL.LEFT_SHOULDER]
            r_shoulder = lm[PL.RIGHT_SHOULDER]
            l_wrist    = lm[PL.LEFT_WRIST]
            r_wrist    = lm[PL.RIGHT_WRIST]
            l_hip      = lm[PL.LEFT_HIP]
            r_hip      = lm[PL.RIGHT_HIP]

            shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            wrist_y    = (l_wrist.y   + r_wrist.y)    / 2

            nose_y        = lm[PL.NOSE].y
            left_wrist_y  = lm[PL.LEFT_WRIST].y
            right_wrist_y = lm[PL.RIGHT_WRIST].y

            # Postura defensiva: ambos os pulsos próximos ao nariz (< 15% altura normalizada)
            guard_left  = abs(left_wrist_y  - nose_y) < 0.15
            guard_right = abs(right_wrist_y - nose_y) < 0.15

            # Braços levantados: média dos pulsos acima dos ombros com margem
            if wrist_y < shoulder_y - 0.08:
                state = "ARMS_UP"
            elif guard_left and guard_right:
                state = "FIGHT_GUARD"
            else:
                state = "NORMAL"

        return results, state

    def draw(self, frame: np.ndarray, results):
        """Desenha os landmarks de pose sobre o frame de exibição."""
        if results and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 230, 130), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 180, 100), thickness=1
                ),
            )


# ══════════════════════════════════════════════
#  FUNÇÃO: DESENHAR HUD
# ══════════════════════════════════════════════
def draw_hud(
    frame:           np.ndarray,
    fps:             float,
    alert_mgr:       AlertManager,
    motion_lvl:      str,
    pose_state:      str,
    detection_count: int,
) -> np.ndarray:
    """
    Renderiza o painel de interface sobre o frame:
    - Barra superior: título e FPS (com cor por faixa de desempenho)
    - Barra inferior: status de detecções, movimento e pose + log de alertas
    - Mira central: referência visual para alinhamento de câmera
    """
    h, w = frame.shape[:2]

    # Painéis sólidos (sem addWeighted — mais rápido)
    cv2.rectangle(frame, (0, 0), (w, 52), (8, 12, 20), -1)
    cv2.rectangle(frame, (0, 50), (w, 52), COLOR_ACCENT, -1)

    cv2.putText(frame, "DEFENSESHIELD ORBITAL INTELLIGENCE",
                (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_ACCENT, 1, cv2.LINE_AA)

    fps_color = COLOR_ACCENT if fps >= 20 else COLOR_WARN if fps >= 10 else COLOR_DANGER
    cv2.putText(frame, f"FPS: {fps:5.1f}",
                (w - 150, 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, fps_color, 1, cv2.LINE_AA)

    panel_h = 26 + MAX_LOG_LINES * 20 + 10
    cv2.rectangle(frame, (0, h - panel_h), (w, h), (8, 12, 20), -1)
    cv2.rectangle(frame, (0, h - panel_h), (w, h - panel_h + 2), COLOR_ACCENT, -1)

    status_line = (
        f"  DETECCOES: {detection_count:02d}   "
        f"MOVIMENTO: {motion_lvl:<6}   "
        f"POSE: {pose_state}"
    )
    cv2.putText(frame, status_line,
                (8, h - panel_h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.52, COLOR_TEXT, 1, cv2.LINE_AA)

    log_lines = alert_mgr.get_log()
    for i, line in enumerate(log_lines):
        alpha = max(0.4, 1.0 - i * 0.1)
        color = tuple(int(c * alpha) for c in COLOR_WARN)
        cv2.putText(frame, line,
                    (12, h - panel_h + 44 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts,
                (w - 220, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_DIM, 1, cv2.LINE_AA)

    # Mira central para referência de câmera
    cx, cy    = w // 2, h // 2
    size, gap = 20, 8
    cv2.line(frame, (cx - size - gap, cy), (cx - gap, cy),        COLOR_ACCENT, 1)
    cv2.line(frame, (cx + gap, cy),        (cx + size + gap, cy), COLOR_ACCENT, 1)
    cv2.line(frame, (cx, cy - size - gap), (cx, cy - gap),        COLOR_ACCENT, 1)
    cv2.line(frame, (cx, cy + gap),        (cx, cy + size + gap), COLOR_ACCENT, 1)

    return frame


# ══════════════════════════════════════════════
#  FUNÇÃO: INICIALIZAR CÂMERA
# ══════════════════════════════════════════════
def init_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Tenta abrir a câmera nos índices [index, 0, 1, 2].
    Configura resolução, FPS e tamanho de buffer.
    Lança RuntimeError se nenhuma câmera for encontrada.
    """
    for cam_idx in [index, 0, 1, 2]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimiza latência do buffer interno
            print(f"[OK] Câmera inicializada no índice {cam_idx}")
            return cap
    raise RuntimeError("Nenhuma webcam encontrada. Verifique o hardware.")


# ══════════════════════════════════════════════
#  FUNÇÃO: SALVAR LOG
# ══════════════════════════════════════════════
def save_log(alert_mgr: AlertManager):

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        fname = os.path.join(LOG_DIR, datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S.log"))
        with open(fname, "w", encoding="utf-8") as f:
            f.write("DefenseShield Orbital Intelligence – Log de Sessão\n")
            f.write("=" * 50 + "\n")
            for line in reversed(alert_mgr.get_log()):
                f.write(line + "\n")
        print(f"[OK] Log salvo em: {fname}")
    except IOError as e:
        print(f"[AVISO] Não foi possível salvar o log: {e}")


def process_motion(
    frame_small:      np.ndarray,
    frame_display:    np.ndarray,
    motion_detector:  MotionDetector,
    alert_mgr:        AlertManager,
    motion_sx:        float,
    motion_sy:        float,
    motion_thresh_small: float,
) -> str:
    """
    Módulo 1 — Detecção de movimento.
    Roda o subtrator de fundo no frame reduzido, classifica o nível
    (BAIXO / MEDIO / ALTO) e desenha os contornos no frame de exibição.
    Retorna: nível de movimento como string.
    """
    motion_area, contours, _ = motion_detector.detect(frame_small)

    if motion_area > motion_thresh_small * 25:
        motion_lvl = "ALTO"
        alert_mgr.trigger("motion_high", f"Movimento intenso detectado (area={int(motion_area)})")
    elif motion_area > motion_thresh_small:
        motion_lvl = "MEDIO"
    else:
        motion_lvl = "BAIXO"

    # Reescala contornos de movimento para resolução de exibição
    for c in contours:
        if cv2.contourArea(c) > 800 * (motion_thresh_small / MOTION_THRESHOLD):
            x, y, cw, ch = cv2.boundingRect(c)
            x1d = int(x  * motion_sx);       y1d = int(y  * motion_sy)
            x2d = int((x + cw) * motion_sx); y2d = int((y + ch) * motion_sy)
            cv2.rectangle(frame_display, (x1d, y1d), (x2d, y2d), (0, 100, 60), 1)

    return motion_lvl


def process_detections(
    frame_small:      np.ndarray,
    frame_display:    np.ndarray,
    detector,
    hog,
    use_yolo:         bool,
    run_yolo:         bool,
    cached_boxes:     list,
    cached_names:     dict,
    alert_mgr:        AlertManager,
    scale_x:          float,
    scale_y:          float,
) -> tuple:
    """
    Módulo 2 — Detecção de objetos (YOLO ou HOG como fallback).
    Executa inferência apenas nos frames marcados por run_yolo;
    nos demais reutiliza o cache do frame anterior (frame skipping).
    Retorna: (detection_count, cached_boxes, cached_names)
    """
    detection_count = 0

    if use_yolo:
        if run_yolo:
            try:
                results_yolo = detector.predict(
                    frame_small, conf=CONFIDENCE_THRESH,
                    verbose=False, device=YOLO_DEVICE
                )[0]
                cached_boxes = []
                cached_names = results_yolo.names
                for box in results_yolo.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cached_boxes.append((cls_id, conf, x1, y1, x2, y2))
            except Exception as e:
                print(f"[AVISO] Erro na inferência YOLO: {e}")

        # Renderiza o cache (seja recém-atualizado ou reutilizado)
        for cls_id, conf, x1, y1, x2, y2 in cached_boxes:
            rx1 = int(x1 * scale_x); ry1 = int(y1 * scale_y)
            rx2 = int(x2 * scale_x); ry2 = int(y2 * scale_y)

            label_name, box_color = ALERT_CLASSES.get(
                cls_id,
                (cached_names.get(cls_id, f"cls{cls_id}"), COLOR_DIM)
            )

            if cls_id in ALERT_CLASSES:
                detection_count += 1
                alert_mgr.trigger(f"yolo_{cls_id}", f"{label_name} (conf={conf:.0%})")
                cv2.rectangle(frame_display, (rx1 - 2, ry1 - 2), (rx2 + 2, ry2 + 2), (0, 0, 0), 2)
                cv2.rectangle(frame_display, (rx1,     ry1),     (rx2,     ry2),     box_color, 2)
            else:
                cv2.rectangle(frame_display, (rx1, ry1), (rx2, ry2), COLOR_DIM, 1)

            label_txt = f"{label_name}  {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame_display, (rx1, ry1 - lh - 6), (rx1 + lw + 4, ry1), box_color, -1)
            cv2.putText(frame_display, label_txt, (rx1 + 2, ry1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 1, cv2.LINE_AA)

    else:
        # Fallback: HOG para detecção de pessoas
        if run_yolo:
            try:
                rects, _ = hog.detectMultiScale(
                    frame_small, winStride=(8, 8), padding=(4, 4), scale=1.05
                )
                cached_boxes = [
                    (0, 1.0, x, y, x + cw, y + ch) for (x, y, cw, ch) in rects
                ] if len(rects) else []
            except Exception as e:
                print(f"[AVISO] Erro no detector HOG: {e}")

        for _, _, x1, y1, x2, y2 in cached_boxes:
            detection_count += 1
            rx1 = int(x1 * scale_x); ry1 = int(y1 * scale_y)
            rx2 = int(x2 * scale_x); ry2 = int(y2 * scale_y)
            cv2.rectangle(frame_display, (rx1, ry1), (rx2, ry2), COLOR_ACCENT, 2)
            alert_mgr.trigger("hog_person", "Pessoa detectada (HOG)")

    return detection_count, cached_boxes, cached_names


def process_pose(
    frame:            np.ndarray,
    frame_small_pose: np.ndarray,
    run_pose:         bool,
    pose_analyzer:    PoseAnalyzer,
    alert_mgr:        AlertManager,
    cached_results,
    cached_state:     str,
) -> tuple:
    """
    Módulo 3 — Análise de pose (MediaPipe).
    Executa inferência apenas nos frames marcados por run_pose;
    nos demais reutiliza o cache. A conversão BGR→RGB ocorre apenas
    quando a inferência é necessária (economia de ~2ms/frame).
    Retorna: (cached_results, cached_state)
    """
    if run_pose:
        # Conversão de cor realizada somente quando Pose vai rodar
        frame_rgb = cv2.cvtColor(frame_small_pose, cv2.COLOR_BGR2RGB)
        try:
            cached_results, cached_state = pose_analyzer.analyze(frame_rgb)
        except Exception as e:
            cached_state   = "ERRO"
            cached_results = None
            print(f"[AVISO] Erro na análise de pose: {e}")

    # Desenha landmarks usando resultado cacheado (a cada frame, não só quando roda)
    pose_analyzer.draw(frame, cached_results)

    # Dispara alertas com base no estado de pose
    if cached_state == "ARMS_UP":
        alert_mgr.trigger("pose_arms", "Bracos levantados — possivel sinal de alerta")
    elif cached_state == "FIGHT_GUARD":
        alert_mgr.trigger("pose_guard", "Postura defensiva detectada")

    return cached_results, cached_state


def main():
    print("=" * 60)
    print("  DefenseShield Orbital Intelligence  [VERSÃO FINAL]")
    print("  FIAP Global Solution 2026")
    print("=" * 60)
    print("Pressione  Q  para encerrar.")
    print("Pressione  S  para salvar screenshot.")
    print()

    # ── Inicialização de câmera ───────────────
    try:
        cap = init_camera()
    except RuntimeError as e:
        print(f"[ERRO FATAL] {e}")
        sys.exit(1)

    # ── Carregamento dos modelos ──────────────
    use_yolo = False
    detector = None
    hog      = None

    if YOLO_AVAILABLE:
        try:
            detector = YOLO(YOLO_MODEL)
            print(f"[OK] YOLO carregado: {YOLO_MODEL}  (device={YOLO_DEVICE})")
            use_yolo = True
        except Exception as e:
            print(f"[AVISO] Falha ao carregar YOLO: {e}. Usando HOG.")

    if not use_yolo:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("[OK] Detector HOG (pessoas) inicializado.")

    pose_analyzer   = PoseAnalyzer()
    motion_detector = MotionDetector()
    alert_mgr       = AlertManager()
    print("[OK] MediaPipe Pose inicializado.")

    # ── Fatores de escala para reprojeção de coordenadas ──
    # As inferências rodam em resolução reduzida; os bounding boxes
    # precisam ser escalados de volta para a resolução de exibição.
    scale_x  = FRAME_WIDTH  / YOLO_INFER_WIDTH
    scale_y  = FRAME_HEIGHT / YOLO_INFER_HEIGHT
    motion_sx = scale_x
    motion_sy = scale_y

    # Threshold de movimento ajustado proporcionalmente à resolução reduzida
    motion_thresh_small = MOTION_THRESHOLD * (
        (YOLO_INFER_WIDTH * YOLO_INFER_HEIGHT) /
        (FRAME_WIDTH      * FRAME_HEIGHT)
    )

    # ── Variáveis de controle de loop ─────────
    fps_alpha   = 0.1    # Peso do frame atual na média exponencial de FPS
    fps_ema     = 0.0    # FPS via Exponential Moving Average (O(1), sem buffer)
    prev_time   = time.time()
    frame_count = 0
    consecutive_failures = 0
    MAX_FAILURES         = 30

    # ── Cache de resultados (frame skipping) ──
    cached_yolo_boxes   = []
    cached_yolo_names   = {}
    cached_pose_results = None
    cached_pose_state   = "NORMAL"
    yolo_frame_counter  = 0
    pose_frame_counter  = 0

    # Buffers de resize pré-alocados (evita realocação por frame)
    frame_small_yolo = np.empty((YOLO_INFER_HEIGHT, YOLO_INFER_WIDTH, 3), dtype=np.uint8)
    frame_small_pose = np.empty((POSE_INFER_HEIGHT, POSE_INFER_WIDTH, 3), dtype=np.uint8)

    # ── Loop de captura e processamento ───────
    while True:
        # grab() captura o frame sem decodificar; retrieve() decodifica quando necessário.
        # Essa separação evita decodificação de frames que seriam descartados.
        grabbed = cap.grab()
        if not grabbed:
            consecutive_failures += 1
            print(f"[AVISO] Frame inválido ({consecutive_failures}/{MAX_FAILURES})")
            if consecutive_failures >= MAX_FAILURES:
                # Tenta reabrir câmera automaticamente após falhas consecutivas
                print("[ERRO] Tentando reabrir câmera...")
                cap.release()
                time.sleep(1.0)
                try:
                    cap = init_camera()
                    consecutive_failures = 0
                except RuntimeError:
                    print("[ERRO FATAL] Não foi possível reabrir câmera. Encerrando.")
                    break
            else:
                time.sleep(0.03)
            continue

        ret, frame = cap.retrieve()
        if not ret or frame is None:
            consecutive_failures += 1
            time.sleep(0.03)
            continue

        consecutive_failures = 0
        frame_count += 1

        # ── FPS via EMA (O(1)) ────────────────
        now         = time.time()
        elapsed     = now - prev_time
        prev_time   = now
        instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        fps_ema     = fps_alpha * instant_fps + (1 - fps_alpha) * fps_ema
        fps         = fps_ema

        # ── Resize para inferência (in-place nos buffers pré-alocados) ──
        cv2.resize(frame, (YOLO_INFER_WIDTH,  YOLO_INFER_HEIGHT),  dst=frame_small_yolo)
        cv2.resize(frame, (POSE_INFER_WIDTH,  POSE_INFER_HEIGHT),  dst=frame_small_pose)

        # Contadores determinam quais módulos pesados rodam neste frame
        yolo_frame_counter += 1
        pose_frame_counter += 1
        run_yolo = (yolo_frame_counter % YOLO_SKIP_FRAMES == 0)
        run_pose = (pose_frame_counter % POSE_SKIP_FRAMES == 0)

        # ── Módulo 1: Movimento ───────────────
        motion_lvl = process_motion(
            frame_small_yolo, frame, motion_detector, alert_mgr,
            motion_sx, motion_sy, motion_thresh_small
        )

        # ── Módulo 2: Detecção de objetos ─────
        detection_count, cached_yolo_boxes, cached_yolo_names = process_detections(
            frame_small_yolo, frame, detector, hog, use_yolo, run_yolo,
            cached_yolo_boxes, cached_yolo_names, alert_mgr, scale_x, scale_y
        )

        # ── Módulo 3: Análise de pose ─────────
        cached_pose_results, cached_pose_state = process_pose(
            frame, frame_small_pose, run_pose,
            pose_analyzer, alert_mgr,
            cached_pose_results, cached_pose_state
        )

        # ── Módulo 4: HUD ─────────────────────
        frame = draw_hud(frame, fps, alert_mgr, motion_lvl, cached_pose_state, detection_count)

        cv2.imshow("DefenseShield Orbital Intelligence", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("[OK] Encerrando por comando do usuário.")
            break
        elif key == ord("s"):
            os.makedirs("screenshots", exist_ok=True)
            fname = os.path.join(
                "screenshots",
                datetime.datetime.now().strftime("shot_%Y%m%d_%H%M%S.jpg")
            )
            cv2.imwrite(fname, frame)
            print(f"[OK] Screenshot salvo: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    save_log(alert_mgr)
    print(f"[OK] Total de frames processados: {frame_count}")
    print("DefenseShield encerrado.")


if __name__ == "__main__":
    main()
