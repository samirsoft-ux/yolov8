import argparse
import torch
import time
import cv2
import numpy as np
import supervision as sv
import subprocess
from ultralytics import YOLO
import threading
import queue
import signal  # Importar el módulo signal
import json
import pymunk  # Asegúrate de haber instalado Pymunk

# Intenta la calibración antes de iniciar el procesamiento de video
try:
    result = subprocess.run(['python', 'calibracion_mesa.py'], capture_output=True, text=True, check=True)
    print("Calibración completada exitosamente.\nSalida:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error durante la calibración. Código de salida:", e.returncode)
    print("Salida de error:", e.stderr)
    exit(1)

COLORS = sv.ColorPalette.default()

# Configura la GPU deseada
torch.cuda.set_device(0)

class VideoProcessor:
    def __init__(self, source_weights_path: str, ip_address: str, port: int, confidence_threshold: float = 0.3, iou_threshold: float = 0.7):
        self.ip_address = ip_address
        self.port = port
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Carga de las coordenadas de las esquinas de la mesa desde data.json
        with open('data.json', 'r') as file:
            data = json.load(file)
            self.mesa_corners = np.array(data['l_circle_projector'])
            
        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(color=COLORS)

        self.frame_queue = queue.Queue(maxsize=10)
        self.shutdown_event = threading.Event()

        self.setup_pymunk()  # Llama a la configuración de Pymunk

    def setup_pymunk(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # Ajusta la gravedad a tus necesidades

    def capture_video(self):
        #video_path = f"http://{self.ip_address}:{self.port}/video"
        video_path = "data/sec.mp4"
        vid = cv2.VideoCapture(video_path)
        while not self.shutdown_event.is_set():
            ret, frame = vid.read()
            if not ret:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def process_video(self):
        frame_times = queue.Queue(maxsize=30)
        total_frames_processed = 0
        total_time_elapsed = 0

        while not self.shutdown_event.is_set() or not self.frame_queue.empty():
            if not self.frame_queue.empty():
                frame_start_time = time.time()
                frame = self.frame_queue.get()
                
                processed_frame = self.process_frame(frame=frame)
                frame_end_time = time.time()
                frame_times.put(frame_end_time - frame_start_time)
                if frame_times.full():
                    frame_times.get()
                
                total_frames_processed += 1
                total_time_elapsed += frame_end_time - frame_start_time

                average_fps = total_frames_processed / total_time_elapsed
                average_latency = sum(list(frame_times.queue)) / frame_times.qsize()

                cv2.putText(processed_frame, f"FPS: {average_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)
                cv2.putText(processed_frame, f"Latency: {average_latency*1000:.2f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)
                
                cv2.imshow("Processed Frame", processed_frame)
                if cv2.waitKey(150) & 0xFF == ord('q'):
                    self.shutdown_event.set()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filtrar las detecciones de las bolas que están dentro del rectángulo de la mesa
        valid_detections_indices_balls = [i for i, bbox in enumerate(detections.xyxy)
                                        if cv2.pointPolygonTest(self.mesa_corners, ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2), False) >= 0
                                        and detections.class_id[i] in [0, 1]]

        # Permitir que las detecciones del taco ocurran en cualquier parte de la imagen
        valid_detections_indices_cue = [i for i, _ in enumerate(detections.xyxy) if detections.class_id[i] == 2]

        # Combinar los índices de detecciones válidas de bolas y taco
        valid_detections_indices = valid_detections_indices_balls + valid_detections_indices_cue

        if valid_detections_indices:
            valid_xyxy = detections.xyxy[valid_detections_indices]
            valid_class_id = detections.class_id[valid_detections_indices]
            valid_confidence = detections.confidence[valid_detections_indices] if detections.confidence is not None else None
            valid_tracker_id = detections.tracker_id[valid_detections_indices] if detections.tracker_id is not None else None

            valid_detections = sv.Detections(xyxy=valid_xyxy, class_id=valid_class_id, confidence=valid_confidence, tracker_id=valid_tracker_id)
        else:
            valid_detections = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.array([], dtype=int), confidence=np.array([], dtype=float), tracker_id=np.array([], dtype=int))

        detections = self.tracker.update_with_detections(valid_detections)

        white_ball_position, cue_positions = self.get_white_ball_and_cue_positions(detections)

        annotated_frame = self.annotate_frame(frame, detections)

        self.process_taco_trajectory(annotated_frame, cue_positions, white_ball_position)

        return annotated_frame

    def process_taco_trajectory(self, frame, cue_positions, white_ball_position):
        for cue_position in cue_positions:
            _, taco_start, taco_end = self.is_pointing_towards_table(cue_position[2])
            if white_ball_position:
                cv2.line(frame, (int(taco_start[0]), int(taco_start[1])), 
                        (int(white_ball_position[0]), int(white_ball_position[1])), (0, 255, 0), 5)
    
    def get_white_ball_and_cue_positions(self, detections):
        white_ball_position = None
        cue_positions = []

        for i, class_id in enumerate(detections.class_id):
            if class_id == 0:  # ID de la bola blanca
                bbox = detections.xyxy[i]
                white_ball_position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            elif class_id == 2:  # ID del taco
                bbox = detections.xyxy[i]
                cue_positions.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox))

        return white_ball_position, cue_positions
    
    def is_pointing_towards_table(self, taco_bbox):
        mesa_center = np.mean(self.mesa_corners, axis=0)
        taco_start = (taco_bbox[0], taco_bbox[1])
        taco_end = (taco_bbox[2], taco_bbox[3])

        distance_start = np.linalg.norm(mesa_center - np.array(taco_start))
        distance_end = np.linalg.norm(mesa_center - np.array(taco_end))

        if distance_start < distance_end:
            return True, taco_start, taco_end
        else:
            return True, taco_end, taco_start

    def annotate_frame(self, frame: np.ndarray, detections) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for i in range(len(detections.xyxy)):
            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]
            tracker_id = detections.tracker_id[i]
            
            x1, y1, x2, y2 = bbox
            # Asegúrate de que las coordenadas del centro sean enteros
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            radius = int(max(x2 - x1, y2 - y1) / 2)
            
            # Bola blanca: dibujar un círculo blanco
            if class_id == 0:  # Bola blanca
                cv2.circle(annotated_frame, center, radius, (255, 255, 255), 2)  # Blanco
                label = f"Bola blanca #{tracker_id}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Otras bolas: dibujar un círculo azul
            elif class_id == 1:  # Otras bolas
                cv2.circle(annotated_frame, center, radius, (255, 0, 0), 2)  # Azul
                label = f"Bola #{tracker_id}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Taco de billar: dibujar un rectángulo verde
            elif class_id == 2:
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Verde
                label = f"Taco #{tracker_id}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame

    def start(self):
        def signal_handler(sig, frame):
            print('Deteniendo los hilos...')
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)

        capture_thread = threading.Thread(target=self.capture_video)
        processing_thread = threading.Thread(target=self.process_video)

        capture_thread.start()
        processing_thread.start()

        capture_thread.join()
        processing_thread.join()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Flow Analysis with YOLO and ByteTrack")
    parser.add_argument("--source_weights_path", required=True, help="Path to the source weights file", type=str)
    parser.add_argument("--ip_address", required=True, help="IP address of the video camera", type=str)
    parser.add_argument("--port", required=True, help="Port of the video stream", type=int)
    parser.add_argument("--confidence_threshold", default=0.3, help="Confidence threshold for the model", type=float)
    parser.add_argument("--iou_threshold", default=0.7, help="IOU threshold for the model", type=float)

    args = parser.parse_args()

    processor = VideoProcessor(source_weights_path=args.source_weights_path,
                               ip_address=args.ip_address,
                               port=args.port,
                               confidence_threshold=args.confidence_threshold,
                               iou_threshold=args.iou_threshold)

    processor.start()