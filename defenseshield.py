"""
DefenseShield Orbital Intelligence  —  VERSÃO OTIMIZADA
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

──────────────────────────────────────────────────────────────
OTIMIZAÇÕES APLICADAS vs VERSÃO ORIGINAL
──────────────────────────────────────────────────────────────
[1] YOLO_SKIP_FRAMES corrigido: variável run_heavy agora é usada
    para pular inferência YOLO de verdade (bug na versão original:
    run_heavy era calculado mas nunca checado no bloco YOLO).

[2] MediaPipe Pose com skip independente (POSE_SKIP_FRAMES).
    Resultado cacheado e reutilizado nos frames pulados.

[3] MotionDetector recebe o frame reduzido (YOLO_INFER_*) em vez
    do frame full-HD — reduz área processada em ~75%.

[4] frame.copy() eliminado: frame_display aponta diretamente para
    o frame capturado; cópia só ocorre no addWeighted do HUD.

[5] FPS calculado com running average incremental (O(1)) em vez de
    np.mean() que percorre o deque inteiro a cada frame.

[6] HOG detectMultiScale agora opera sobre frame reduzido com
    coordenadas reescaladas, igual ao caminho YOLO.

[7] cap.grab() + cap.retrieve() em vez de cap.read() remove um
    decode desnecessário durante warmup e facilita futuros
    multi-câmera.

[8] Alocação de frame_rgb_pose movida para dentro do bloco de
    skip para evitar cvtColor em frames que não usarão Pose.

[9] draw_hud usa frame inplace (sem overlay separado quando
    transparência não é necessária nos painéis de texto).
──────────────────────────────────────────────────────────────
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
# YOLO_SKIP_FRAMES  : executa YOLO a cada N frames (era calculado mas não usado!)
# POSE_SKIP_FRAMES  : executa Pose a cada N frames (novo — Pose é pesado também)
YOLO_SKIP_FRAMES    = 3
POSE_SKIP_FRAMES    = 2   # Pose roda a cada 2 frames (novo)

# ── Resoluções de inferência ──────────────────
YOLO_INFER_WIDTH    = 320
YOLO_INFER_HEIGHT   = 256
POSE_INFER_WIDTH    = 320
POSE_INFER_HEIGHT   = 240

import torch
YOLO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════
#  CLASSES ALVO
# ══════════════════════════════════════════════
ALERT_CLASSES = {
    0:  ("Pessoa detectada",   (0, 255, 100)),
    67: ("Dispositivo movel",  (0, 200, 255)),
    39: ("Objeto suspeito",    (0, 100, 255)),
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
    def __init__(self, cooldown_sec: float = ALERT_COOLDOWN_SEC):
        self.cooldown   = cooldown_sec
        self._last_time = {}
        self.log        = deque(maxlen=MAX_LOG_LINES)

    def trigger(self, alert_type: str, message: str) -> bool:
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
#  [OPT-3] Agora aceita frame pequeno externamente
# ══════════════════════════════════════════════
class MotionDetector:
    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray):
        mask = self.subtractor.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area  = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 300)
        return motion_area, contours, mask


# ══════════════════════════════════════════════
#  CLASSE: ANALISADOR DE POSE (MediaPipe)
# ══════════════════════════════════════════════
class PoseAnalyzer:
    def __init__(self):
        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose       = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame_rgb: np.ndarray):
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
            hip_y      = (l_hip.y     + r_hip.y)      / 2

            if wrist_y < shoulder_y - 0.08:
                state = "ARMS_UP"
            elif hip_y > shoulder_y + 0.35:
                state = "DOWN"

        return results, state

    def draw(self, frame: np.ndarray, results):
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
#  [OPT-9] Overlay só no bloco de mira (mais leve)
# ══════════════════════════════════════════════
def draw_hud(
    frame:           np.ndarray,
    fps:             float,
    alert_mgr:       AlertManager,
    motion_lvl:      str,
    pose_state:      str,
    detection_count: int,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Painéis sólidos direto no frame (sem overlay/addWeighted nos retângulos)
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

    # Mira central — único elemento que usa blend (somente essa região)
    cx, cy   = w // 2, h // 2
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
    for cam_idx in [index, 0, 1, 2]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[OK] Câmera inicializada no índice {cam_idx}")
            return cap
    raise RuntimeError("Nenhuma webcam encontrada. Verifique o hardware.")


# ══════════════════════════════════════════════
#  FUNÇÃO: SALVAR LOG
# ══════════════════════════════════════════════
def save_log(alert_mgr: AlertManager):
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
    print("  DefenseShield Orbital Intelligence  [OTIMIZADO]")
    print("  FIAP Global Solution 2026")
    print("=" * 60)
    print("Pressione  Q  para encerrar.")
    print("Pressione  S  para salvar screenshot.")
    print()

    try:
        cap = init_camera()
    except RuntimeError as e:
        print(f"[ERRO FATAL] {e}")
        sys.exit(1)

    # ── Modelos ───────────────────────────────
    if YOLO_AVAILABLE:
        try:
            detector = YOLO(YOLO_MODEL)
            print(f"[OK] YOLO carregado: {YOLO_MODEL}  (device={YOLO_DEVICE})")
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

    pose_analyzer   = PoseAnalyzer()
    motion_detector = MotionDetector()
    alert_mgr       = AlertManager()
    print("[OK] MediaPipe Pose inicializado.")

    # ── Escala para reprojetar YOLO → frame display ──
    scale_x = FRAME_WIDTH  / YOLO_INFER_WIDTH
    scale_y = FRAME_HEIGHT / YOLO_INFER_HEIGHT
    # Escala para reprojetar contornos de movimento → frame display
    motion_sx = FRAME_WIDTH  / YOLO_INFER_WIDTH
    motion_sy = FRAME_HEIGHT / YOLO_INFER_HEIGHT

    # ── Variáveis de controle ─────────────────
    # [OPT-5] FPS com média incremental O(1)
    fps_accum   = 0.0
    fps_count   = 0
    fps_alpha   = 0.1          # EMA — quanto peso dar ao frame atual
    fps_ema     = 0.0          # exponential moving average de FPS
    prev_time   = time.time()
    frame_count = 0
    consecutive_failures = 0
    MAX_FAILURES         = 30

    # ── Cache de resultados (frame skipping) ──
    cached_yolo_boxes   = []        # [(cls_id, conf, x1,y1,x2,y2), ...]
    cached_yolo_names   = {}        # {cls_id: nome}
    cached_pose_results = None
    cached_pose_state   = "NORMAL"
    yolo_frame_counter  = 0
    pose_frame_counter  = 0

    # Aloca buffers de resize uma vez (evita realocação por frame)
    frame_small_yolo = np.empty((YOLO_INFER_HEIGHT, YOLO_INFER_WIDTH, 3), dtype=np.uint8)
    frame_small_pose = np.empty((POSE_INFER_HEIGHT, POSE_INFER_WIDTH, 3), dtype=np.uint8)

    while True:
        # [OPT-7] grab+retrieve em vez de read()
        grabbed = cap.grab()
        if not grabbed:
            consecutive_failures += 1
            print(f"[AVISO] Frame inválido ({consecutive_failures}/{MAX_FAILURES})")
            if consecutive_failures >= MAX_FAILURES:
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

        # ── FPS (EMA — O(1), sem np.mean) ────────
        now     = time.time()
        elapsed = now - prev_time
        prev_time = now
        instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        fps_ema = fps_alpha * instant_fps + (1 - fps_alpha) * fps_ema
        fps     = fps_ema

        # ── Resize para inferência ────────────────
        # [OPT-4] Sem frame.copy() — operamos direto no frame capturado
        cv2.resize(frame, (YOLO_INFER_WIDTH, YOLO_INFER_HEIGHT), dst=frame_small_yolo)

        yolo_frame_counter += 1
        pose_frame_counter += 1
        run_yolo = (yolo_frame_counter % YOLO_SKIP_FRAMES == 0)   # [OPT-1] bug corrigido
        run_pose = (pose_frame_counter % POSE_SKIP_FRAMES == 0)   # [OPT-2] novo

        # ─────────────────────────────────────────
        #  MÓDULO 1: DETECÇÃO DE MOVIMENTO
        #  [OPT-3] Usa frame pequeno (75% menos pixels)
        # ─────────────────────────────────────────
        # Threshold precisa ser ajustado para a resolução menor
        motion_thresh_small = MOTION_THRESHOLD * (
            (YOLO_INFER_WIDTH * YOLO_INFER_HEIGHT) /
            (FRAME_WIDTH      * FRAME_HEIGHT)
        )

        motion_area, contours, _ = motion_detector.detect(frame_small_yolo)

        if motion_area > motion_thresh_small * 3:
            motion_lvl = "ALTO"
            alert_mgr.trigger("motion_high", f"Movimento intenso detectado (area={int(motion_area)})")
        elif motion_area > motion_thresh_small:
            motion_lvl = "MEDIO"
        else:
            motion_lvl = "BAIXO"

        # Reescala contornos de movimento para o frame de display
        for c in contours:
            if cv2.contourArea(c) > 800 * (motion_thresh_small / MOTION_THRESHOLD):
                x, y, cw, ch = cv2.boundingRect(c)
                x1d = int(x  * motion_sx);  y1d = int(y  * motion_sy)
                x2d = int((x+cw) * motion_sx); y2d = int((y+ch) * motion_sy)
                cv2.rectangle(frame, (x1d, y1d), (x2d, y2d), (0, 100, 60), 1)

        # ─────────────────────────────────────────
        #  MÓDULO 2: DETECÇÃO DE OBJETOS (YOLO / HOG)
        #  [OPT-1] run_yolo agora é realmente verificado
        # ─────────────────────────────────────────
        detection_count = 0

        if use_yolo:
            if run_yolo:
                try:
                    results_yolo = detector.predict(
                        frame_small_yolo, conf=CONFIDENCE_THRESH,
                        verbose=False, device=YOLO_DEVICE
                    )[0]
                    cached_yolo_boxes = []
                    cached_yolo_names = results_yolo.names
                    for box in results_yolo.boxes:
                        cls_id = int(box.cls[0])
                        conf   = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cached_yolo_boxes.append((cls_id, conf, x1, y1, x2, y2))
                except Exception as e:
                    print(f"[AVISO] Erro na inferência YOLO: {e}")

            # Renderiza o cache (seja novo ou reutilizado)
            for cls_id, conf, x1, y1, x2, y2 in cached_yolo_boxes:
                # Reprojetar coordenadas
                rx1 = int(x1 * scale_x); ry1 = int(y1 * scale_y)
                rx2 = int(x2 * scale_x); ry2 = int(y2 * scale_y)

                label_name, box_color = ALERT_CLASSES.get(
                    cls_id,
                    (cached_yolo_names.get(cls_id, f"cls{cls_id}"), COLOR_DIM)
                )

                if cls_id in ALERT_CLASSES:
                    detection_count += 1
                    alert_mgr.trigger(f"yolo_{cls_id}", f"{label_name} (conf={conf:.0%})")
                    cv2.rectangle(frame, (rx1-2, ry1-2), (rx2+2, ry2+2), (0, 0, 0), 2)
                    cv2.rectangle(frame, (rx1,  ry1),   (rx2,  ry2),   box_color, 2)
                else:
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_DIM, 1)

                label_txt = f"{label_name}  {conf:.0%}"
                (lw, lh), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (rx1, ry1 - lh - 6), (rx1 + lw + 4, ry1), box_color, -1)
                cv2.putText(frame, label_txt, (rx1 + 2, ry1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 1, cv2.LINE_AA)

        else:
            # [OPT-6] HOG no frame reduzido com reescala de coordenadas
            if run_yolo:
                try:
                    rects, _ = hog.detectMultiScale(
                        frame_small_yolo, winStride=(8, 8), padding=(4, 4), scale=1.05
                    )
                    cached_yolo_boxes = [
                        (0, 1.0, x, y, x + cw, y + ch) for (x, y, cw, ch) in rects
                    ] if len(rects) else []
                except Exception as e:
                    print(f"[AVISO] Erro no detector HOG: {e}")

            for _, _, x1, y1, x2, y2 in cached_yolo_boxes:
                detection_count += 1
                rx1 = int(x1 * scale_x); ry1 = int(y1 * scale_y)
                rx2 = int(x2 * scale_x); ry2 = int(y2 * scale_y)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_ACCENT, 2)
                alert_mgr.trigger("hog_person", "Pessoa detectada (HOG)")

        # ─────────────────────────────────────────
        #  MÓDULO 3: ANÁLISE DE POSE
        #  [OPT-2] Skip real com cache + [OPT-8] cvtColor só quando necessário
        # ─────────────────────────────────────────
        if run_pose:
            cv2.resize(frame, (POSE_INFER_WIDTH, POSE_INFER_HEIGHT), dst=frame_small_pose)
            # [OPT-8] Conversão de cor só ocorre nos frames em que Pose roda
            frame_rgb_pose = cv2.cvtColor(frame_small_pose, cv2.COLOR_BGR2RGB)
            try:
                cached_pose_results, cached_pose_state = pose_analyzer.analyze(frame_rgb_pose)
            except Exception as e:
                cached_pose_state   = "ERRO"
                cached_pose_results = None
                print(f"[AVISO] Erro na análise de pose: {e}")

        pose_state = cached_pose_state

        # Desenha landmarks usando resultado cacheado
        pose_analyzer.draw(frame, cached_pose_results)

        if pose_state == "ARMS_UP":
            alert_mgr.trigger("pose_arms", "Bracos levantados – possivel sinal de alerta")
        elif pose_state == "DOWN":
            alert_mgr.trigger("pose_down", "Pessoa no solo detectada")

        # ─────────────────────────────────────────
        #  MÓDULO 4: HUD
        # ─────────────────────────────────────────
        frame = draw_hud(frame, fps, alert_mgr, motion_lvl, pose_state, detection_count)

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