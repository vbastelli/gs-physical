"""
DefenseShield Orbital Intelligence
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

# ─────────────────────────────────────────────
#  Tentativa de importar YOLO (Ultralytics)
#  Se não estiver disponível, usa fallback HOG
# ─────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[AVISO] ultralytics não encontrado. Usando detector HOG como fallback.")


# ══════════════════════════════════════════════
#  CONFIGURAÇÕES GLOBAIS
# ══════════════════════════════════════════════
CAMERA_INDEX        = 0          # índice da webcam
FRAME_WIDTH         = 1280       # resolução desejada (largura)
FRAME_HEIGHT        = 720        # resolução desejada (altura)
CONFIDENCE_THRESH   = 0.45       # limiar de confiança YOLO
MAX_LOG_LINES       = 8          # linhas do log exibido no HUD
ALERT_COOLDOWN_SEC  = 3          # segundos entre alertas do mesmo tipo
MOTION_HISTORY_LEN  = 10         # frames para cálculo de movimento
MOTION_THRESHOLD    = 2500       # área mínima de movimento (pixels²)
LOG_DIR             = "logs"     # pasta para salvar logs
YOLO_MODEL          = "yolov8n.pt"  # modelo YOLO (nano = mais leve)


# ══════════════════════════════════════════════
#  CLASSES ALVO PARA ALERTA (COCO dataset IDs)
#  0 = person | 67 = cell phone | 39 = bottle
#  Adicione IDs conforme necessidade do cenário
# ══════════════════════════════════════════════
ALERT_CLASSES = {
    0:  ("Pessoa detectada",      (0, 255, 100)),   # verde
    67: ("Dispositivo movel",     (0, 200, 255)),   # amarelo
    39: ("Objeto suspeito",       (0, 100, 255)),   # laranja
}

# Paleta de cores do HUD (BGR)
COLOR_BG_DARK   = (10,  15,  25)
COLOR_ACCENT    = (0,  210, 120)
COLOR_WARN      = (0,  140, 255)
COLOR_DANGER    = (0,   50, 220)
COLOR_TEXT      = (200, 220, 240)
COLOR_DIM       = (100, 110, 130)


# ══════════════════════════════════════════════
#  CLASSE: GERENCIADOR DE ALERTAS
# ══════════════════════════════════════════════
class AlertManager:
    """Controla cooldown e registro de alertas para evitar spam."""

    def __init__(self, cooldown_sec: float = ALERT_COOLDOWN_SEC):
        self.cooldown   = cooldown_sec
        self._last_time = {}   # tipo_alerta -> timestamp
        self.log        = deque(maxlen=MAX_LOG_LINES)

    def trigger(self, alert_type: str, message: str) -> bool:
        """Retorna True se o alerta deve ser disparado (fora do cooldown)."""
        now = time.time()
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
#  CLASSE: DETECTOR DE MOVIMENTO (background subtraction)
# ══════════════════════════════════════════════
class MotionDetector:
    """Detecta movimento usando subtração de fundo (MOG2)."""

    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray):
        """
        Retorna:
          - motion_area: área total de pixels em movimento
          - contours:    lista de contornos de regiões em movimento
          - mask:        máscara binária do foreground
        """
        mask = self.subtractor.apply(frame)
        # Morfologia para remover ruído
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
    Analisa poses humanas com MediaPipe Pose.
    Detecta estado postural básico: em pé, sentado, agachado ou
    braços levantados (potencial sinalização de emergência).
    """

    def __init__(self):
        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose       = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,          # 0=lite (mais rápido)
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame_rgb: np.ndarray):
        """
        Retorna:
          - results: objeto MediaPipe com landmarks
          - state:   string com estado postural ("NORMAL", "ARMS_UP", "DOWN")
        """
        results = self.pose.process(frame_rgb)
        state   = "NORMAL"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            PL = self.mp_pose.PoseLandmark

            # Ombros, pulsos e quadris
            l_shoulder = lm[PL.LEFT_SHOULDER]
            r_shoulder = lm[PL.RIGHT_SHOULDER]
            l_wrist    = lm[PL.LEFT_WRIST]
            r_wrist    = lm[PL.RIGHT_WRIST]
            l_hip      = lm[PL.LEFT_HIP]
            r_hip      = lm[PL.RIGHT_HIP]

            shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            wrist_y    = (l_wrist.y   + r_wrist.y)    / 2
            hip_y      = (l_hip.y     + r_hip.y)      / 2

            # Braços levantados acima dos ombros → possível alerta/emergência
            if wrist_y < shoulder_y - 0.08:
                state = "ARMS_UP"
            # Pessoa no chão (quadril próximo ou abaixo dos ombros na imagem)
            elif hip_y > shoulder_y + 0.35:
                state = "DOWN"

        return results, state

    def draw(self, frame: np.ndarray, results):
        """Desenha landmarks de pose no frame."""
        if results.pose_landmarks:
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
#  FUNÇÃO: DESENHAR HUD (Heads-Up Display)
# ══════════════════════════════════════════════
def draw_hud(
    frame:      np.ndarray,
    fps:        float,
    alert_mgr:  AlertManager,
    motion_lvl: str,
    pose_state: str,
    detection_count: int,
) -> np.ndarray:
    """
    Renderiza o painel HUD estilo militar/espacial sobre o frame.
    Inclui: status, FPS, nível de ameaça, log de eventos.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Painel superior ──────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 52), (8, 12, 20), -1)
    cv2.rectangle(overlay, (0, 50), (w, 52), COLOR_ACCENT, -1)

    # Título
    cv2.putText(overlay, "DEFENSESHIELD ORBITAL INTELLIGENCE",
                (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_ACCENT, 1, cv2.LINE_AA)

    # FPS (canto direito do painel)
    fps_color = COLOR_ACCENT if fps >= 20 else COLOR_WARN if fps >= 10 else COLOR_DANGER
    cv2.putText(overlay, f"FPS: {fps:5.1f}",
                (w - 150, 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, fps_color, 1, cv2.LINE_AA)

    # ── Painel inferior ───────────────────────────────────
    panel_h = 26 + MAX_LOG_LINES * 20 + 10
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (8, 12, 20), -1)
    cv2.rectangle(overlay, (0, h - panel_h), (w, h - panel_h + 2), COLOR_ACCENT, -1)

    # Status de detecção e movimento
    status_line = (
        f"  DETECCOES: {detection_count:02d}   "
        f"MOVIMENTO: {motion_lvl:<6}   "
        f"POSE: {pose_state}"
    )
    cv2.putText(overlay, status_line,
                (8, h - panel_h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.52, COLOR_TEXT, 1, cv2.LINE_AA)

    # Log de alertas
    log_lines = alert_mgr.get_log()
    for i, line in enumerate(log_lines):
        alpha = max(0.4, 1.0 - i * 0.1)
        color = tuple(int(c * alpha) for c in COLOR_WARN)
        cv2.putText(overlay, line,
                    (12, h - panel_h + 44 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

    # Timestamp canto inferior direito
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(overlay, ts,
                (w - 220, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_DIM, 1, cv2.LINE_AA)

    # ── Mira central (decorativa / referência visual) ────
    cx, cy = w // 2, h // 2
    size, gap = 20, 8
    cv2.line(overlay, (cx - size - gap, cy), (cx - gap, cy), COLOR_ACCENT, 1)
    cv2.line(overlay, (cx + gap, cy),        (cx + size + gap, cy), COLOR_ACCENT, 1)
    cv2.line(overlay, (cx, cy - size - gap), (cx, cy - gap), COLOR_ACCENT, 1)
    cv2.line(overlay, (cx, cy + gap),        (cx, cy + size + gap), COLOR_ACCENT, 1)

    # Blend suave para transparência
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    return frame


# ══════════════════════════════════════════════
#  FUNÇÃO: INICIALIZAR CÂMERA COM TRATAMENTO DE FALHA
# ══════════════════════════════════════════════
def init_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Abre a webcam com tratamento robusto de exceção.
    Tenta índices alternativos se o principal falhar.
    """
    for cam_idx in [index, 0, 1, 2]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimiza latência de buffer
            print(f"[OK] Câmera inicializada no índice {cam_idx}")
            return cap
    raise RuntimeError("Nenhuma webcam encontrada. Verifique o hardware.")


# ══════════════════════════════════════════════
#  FUNÇÃO: SALVAR LOG EM ARQUIVO
# ══════════════════════════════════════════════
def save_log(alert_mgr: AlertManager):
    """Persiste o log de alertas em arquivo texto na pasta logs/."""
    os.makedirs(LOG_DIR, exist_ok=True)
    fname = os.path.join(LOG_DIR, datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S.log"))
    with open(fname, "w", encoding="utf-8") as f:
        f.write("DefenseShield Orbital Intelligence – Log de Sessão\n")
        f.write("=" * 50 + "\n")
        for line in reversed(alert_mgr.get_log()):
            f.write(line + "\n")
    print(f"[OK] Log salvo em: {fname}")


# ══════════════════════════════════════════════
#  LOOP PRINCIPAL
# ══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  DefenseShield Orbital Intelligence")
    print("  FIAP Global Solution 2026")
    print("=" * 60)
    print("Pressione  Q  para encerrar.")
    print("Pressione  S  para salvar screenshot.")
    print()

    # ── Inicializar câmera ───────────────────────────────
    try:
        cap = init_camera()
    except RuntimeError as e:
        print(f"[ERRO FATAL] {e}")
        sys.exit(1)

    # ── Inicializar modelos ───────────────────────────────
    # YOLO ou HOG como fallback
    if YOLO_AVAILABLE:
        try:
            detector = YOLO(YOLO_MODEL)
            print(f"[OK] YOLO carregado: {YOLO_MODEL}")
            use_yolo = True
        except Exception as e:
            print(f"[AVISO] Falha ao carregar YOLO: {e}. Usando HOG.")
            use_yolo = False
    else:
        use_yolo = False

    if not use_yolo:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("[OK] Detector HOG (pessoas) inicializado.")

    pose_analyzer  = PoseAnalyzer()
    motion_detector = MotionDetector()
    alert_mgr      = AlertManager()
    print("[OK] MediaPipe Pose inicializado.")

    # ── Variáveis de controle ────────────────────────────
    fps_history = deque(maxlen=30)
    prev_time   = time.time()
    frame_count = 0
    consecutive_failures = 0   # contador de frames inválidos consecutivos
    MAX_FAILURES = 30          # tolerância antes de tentar reabrir câmera

    # ── Loop de captura ───────────────────────────────────
    while True:
        ret, frame = cap.read()

        # ── Tratamento de falha de frame ─────────────────
        if not ret or frame is None:
            consecutive_failures += 1
            print(f"[AVISO] Frame inválido ({consecutive_failures}/{MAX_FAILURES})")

            if consecutive_failures >= MAX_FAILURES:
                print("[ERRO] Muitos frames inválidos. Tentando reabrir câmera...")
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

        consecutive_failures = 0
        frame_count += 1

        # ── Cálculo de FPS ───────────────────────────────
        now      = time.time()
        elapsed  = now - prev_time
        prev_time = now
        fps_history.append(1.0 / elapsed if elapsed > 0 else 0)
        fps = np.mean(fps_history)

        # ── Pré-processamento ────────────────────────────
        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display = frame.copy()

        # ─────────────────────────────────────────────────
        #  MÓDULO 1: DETECÇÃO DE MOVIMENTO
        # ─────────────────────────────────────────────────
        motion_area, contours, _ = motion_detector.detect(frame)

        if motion_area > MOTION_THRESHOLD * 3:
            motion_lvl = "ALTO"
            if alert_mgr.trigger("motion_high", f"Movimento intenso detectado (area={int(motion_area)})"):
                pass  # log já registrado internamente
        elif motion_area > MOTION_THRESHOLD:
            motion_lvl = "MEDIO"
        else:
            motion_lvl = "BAIXO"

        # Desenha contornos de movimento em verde escuro
        for c in contours:
            if cv2.contourArea(c) > 800:
                x, y, cw, ch = cv2.boundingRect(c)
                cv2.rectangle(frame_display, (x, y), (x + cw, y + ch), (0, 100, 60), 1)

        # ─────────────────────────────────────────────────
        #  MÓDULO 2: DETECÇÃO DE OBJETOS (YOLO ou HOG)
        # ─────────────────────────────────────────────────
        detection_count = 0

        if use_yolo:
            # Inferência YOLO (stream=False para batch único)
            try:
                results_yolo = detector.predict(
                    frame, conf=CONFIDENCE_THRESH, verbose=False
                )[0]

                for box in results_yolo.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    label_name, box_color = ALERT_CLASSES.get(
                        cls_id,
                        (results_yolo.names.get(cls_id, f"cls{cls_id}"), COLOR_DIM)
                    )

                    # Destaque especial para classes de alerta
                    if cls_id in ALERT_CLASSES:
                        detection_count += 1
                        alert_mgr.trigger(
                            f"yolo_{cls_id}",
                            f"{label_name} (conf={conf:.0%})"
                        )
                        # Caixa com borda dupla para destaque
                        cv2.rectangle(frame_display, (x1-2, y1-2), (x2+2, y2+2), (0,0,0), 2)
                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), box_color, 2)
                    else:
                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), COLOR_DIM, 1)

                    # Rótulo com fundo
                    label_txt = f"{label_name}  {conf:.0%}"
                    (lw, lh), _ = cv2.getTextSize(
                        label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
                    )
                    cv2.rectangle(frame_display,
                                  (x1, y1 - lh - 6), (x1 + lw + 4, y1), box_color, -1)
                    cv2.putText(frame_display, label_txt,
                                (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 1, cv2.LINE_AA)

            except Exception as e:
                # Falha de inferência não deve derrubar o sistema
                print(f"[AVISO] Erro na inferência YOLO: {e}")

        else:
            # Fallback: HOG detector de pessoas
            try:
                rects, _ = hog.detectMultiScale(
                    frame, winStride=(8, 8), padding=(4, 4), scale=1.05
                )
                for (x, y, cw, ch) in rects:
                    detection_count += 1
                    cv2.rectangle(frame_display, (x, y), (x+cw, y+ch), COLOR_ACCENT, 2)
                    alert_mgr.trigger("hog_person", "Pessoa detectada (HOG)")
            except Exception as e:
                print(f"[AVISO] Erro no detector HOG: {e}")

        # ─────────────────────────────────────────────────
        #  MÓDULO 3: ANÁLISE DE POSE (MediaPipe)
        # ─────────────────────────────────────────────────
        try:
            pose_results, pose_state = pose_analyzer.analyze(frame_rgb)
            pose_analyzer.draw(frame_display, pose_results)

            if pose_state == "ARMS_UP":
                alert_mgr.trigger("pose_arms", "Bracos levantados – possivel sinal de alerta")
            elif pose_state == "DOWN":
                alert_mgr.trigger("pose_down", "Pessoa no solo detectada")
        except Exception as e:
            pose_state = "ERRO"
            print(f"[AVISO] Erro na análise de pose: {e}")

        # ─────────────────────────────────────────────────
        #  MÓDULO 4: HUD OVERLAY
        # ─────────────────────────────────────────────────
        frame_display = draw_hud(
            frame_display, fps, alert_mgr,
            motion_lvl, pose_state, detection_count
        )

        # ── Exibição ────────────────────────────────────
        cv2.imshow("DefenseShield Orbital Intelligence", frame_display)

        # ── Controles de teclado ─────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:   # Q ou ESC para sair
            print("[OK] Encerrando por comando do usuário.")
            break
        elif key == ord("s"):              # S para screenshot
            os.makedirs("screenshots", exist_ok=True)
            fname = os.path.join(
                "screenshots",
                datetime.datetime.now().strftime("shot_%Y%m%d_%H%M%S.jpg")
            )
            cv2.imwrite(fname, frame_display)
            print(f"[OK] Screenshot salvo: {fname}")

    # ── Finalização ──────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    save_log(alert_mgr)
    print(f"[OK] Total de frames processados: {frame_count}")
    print("DefenseShield encerrado.")


# ══════════════════════════════════════════════
if __name__ == "__main__":
    main()
