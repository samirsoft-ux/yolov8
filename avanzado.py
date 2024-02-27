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
import pymunk  # Asegúrate de importar Pymunk

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

# Define una función para calcular la dirección del taco
def calcular_direccion_taco(ubicaciones_taco):
    if len(ubicaciones_taco) >= 2:
        # Calcula la dirección basada en las últimas dos posiciones
        direccion = np.array(ubicaciones_taco[-1]) - np.array(ubicaciones_taco[-2])
        norma = np.linalg.norm(direccion)
        if norma == 0: return None  # Evita división por cero
        return direccion / norma
    return None

def visualizar_trayectoria(frame, inicio, fin, color=(0, 255, 255), grosor=2):
    # Asegurarse de que los puntos sean enteros
    inicio = (int(inicio[0]), int(inicio[1]))
    fin = (int(fin[0]), int(fin[1]))
    cv2.line(frame, inicio, fin, color, grosor)

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
        
        # Agrega una lista para almacenar las posiciones del taco
        self.ubicaciones_taco = []
        
        # Inicialización del espacio de Pymunk
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No necesitamos gravedad en una mesa de billar

        # Lista para almacenar referencias a las bolas y el taco en Pymunk
        self.balls = {}
        self.cue = None  # Podrías tener solo una referencia al taco si siempre hay uno

    def update_or_add_ball_in_pymunk(self, detection, index):
        # Obtén el ID de seguimiento de la bola
        tracker_id = detection.tracker_id[index]
        position = ((detection.xyxy[index][0] + detection.xyxy[index][2]) / 2,
                    (detection.xyxy[index][1] + detection.xyxy[index][3]) / 2)
        radius = (detection.xyxy[index][2] - detection.xyxy[index][0]) / 2

        if tracker_id not in self.balls:
            # Si la bola no existe en el espacio de Pymunk, agrégala
            self.balls[tracker_id] = self.add_ball(position, radius)
        else:
            # Si la bola ya existe, actualiza su posición directamente (opcional)
            # Puede que prefieras dejar que Pymunk maneje la posición por simulación
            pass
    
    def simulate_and_draw_trajectories(self, frame):
        for _ in range(50):
            self.space.step(1/50.0)
        
        for ball_shape in self.balls.values():
            position = ball_shape.body.position
            cv2.circle(frame, (int(position.x), int(position.y)), int(ball_shape.radius), (0, 255, 0), 2)
    
    def add_ball(self, position, radius, mass=1):
        """Agrega una bola al espacio de Pymunk."""
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, moment)
        body.position = position
        shape = pymunk.Circle(body, radius)
        self.space.add(body, shape)
        return shape

    def add_cue(self, position, length, mass=1):
        """Agrega el taco al espacio de Pymunk."""
        # Esta es una implementación simplificada. Podrías necesitar ajustarla.
        moment = pymunk.moment_for_segment(mass, (0, 0), (length, 0), 5)
        body = pymunk.Body(mass, moment, body_type=pymunk.Body.KINEMATIC)
        body.position = position
        shape = pymunk.Segment(body, (0, 0), (length, 0), 5)
        self.space.add(body, shape)
        return shape

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
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.shutdown_event.set()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Preprocesamiento y filtrado de detecciones
        valid_detections_indices = self.filter_detections(detections)
        
        # Actualización de detecciones con ByteTrack
        if valid_detections_indices:
            valid_detections = self.extract_valid_detections(detections, valid_detections_indices)
        else:
            valid_detections = sv.Detections.empty()

        detections = self.tracker.update_with_detections(valid_detections)

        # Procesamiento de detecciones de bolas y taco
        self.handle_detections(frame, detections)

        return self.annotate_frame(frame, detections)

    def filter_detections(self, detections):
        valid_detections_indices_balls = [i for i, bbox in enumerate(detections.xyxy)
                                        if self.is_ball_inside_table(bbox) and detections.class_id[i] in [0, 1]]
        valid_detections_indices_cue = [i for i, class_id in enumerate(detections.class_id) if class_id == 2]
        return valid_detections_indices_balls + valid_detections_indices_cue
    
    def is_ball_inside_table(self, bbox):
        center_point = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        return cv2.pointPolygonTest(self.mesa_corners, center_point, False) >= 0
    
    def extract_valid_detections(self, detections, valid_detections_indices):
        return sv.Detections(xyxy=detections.xyxy[valid_detections_indices],
                            class_id=detections.class_id[valid_detections_indices],
                            confidence=detections.confidence[valid_detections_indices] if detections.confidence is not None else None,
                            tracker_id=detections.tracker_id[valid_detections_indices] if detections.tracker_id is not None else None)

    def handle_detections(self, frame, detections):
        for i in range(len(detections.xyxy)):
            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]
            tracker_id = detections.tracker_id[i]
            
            if class_id in [0, 1]:  # Bolas
                self.update_or_add_ball_in_pymunk(detections, i)
            elif class_id == 2:  # Taco
                centro_taco = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                self.ubicaciones_taco.append(centro_taco)
                if len(self.ubicaciones_taco) > 1:
                    direccion = calcular_direccion_taco(self.ubicaciones_taco)
                    if direccion is not None:
                        punto_final = (centro_taco[0] + direccion[0] * 100, centro_taco[1] + direccion[1] * 100)
                        visualizar_trayectoria(frame, centro_taco, punto_final)
        
        self.simulate_and_draw_trajectories(frame)

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