#import torch
#print(torch.cuda.is_available())

#import torch
#print(torch.__version__)
#print(torch.cuda.is_available())

#-------------------------------
import argparse
import torch
import time
import cv2
import numpy as np
import supervision as sv
import subprocess
from ultralytics import YOLO
import queue

#ip_address = "192.168.137.83"
#port = 8080
#ip = "http://{}:{}/video".format(ip_address, port)

try:
    result = subprocess.run(['python', 'calibracion_mesa.py'], capture_output=True, text=True, check=True)
    print("Calibración completada exitosamente.\nSalida:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error durante la calibración. Código de salida:", e.returncode)
    print("Salida de error:", e.stderr)
    # Aquí puedes decidir cómo manejar el error, por ejemplo, terminar el script
    exit(1)

COLORS = sv.ColorPalette.default()

torch.cuda.set_device(0) # Set to your desired GPU number

class VideoProcessor:
    def __init__(self, source_weights_path: str, ip_address: str, port: int, target_video_path: str = None, confidence_threshold: float = 0.3, iou_threshold: float = 0.7):
        self.ip_address = ip_address
        self.port = port
        self.target_video_path = target_video_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(color=COLORS)

    def process_video(self):
        video_path = f"http://{self.ip_address}:{self.port}/video"
        vid = cv2.VideoCapture(video_path)

        frame_times = queue.Queue(maxsize=30)  # Para calcular la latencia
        total_frames_processed = 0
        total_time_elapsed = 0

        while True:
            ret, frame = vid.read()
            if not ret:
                break
            
            frame_start_time = time.time()  # Marca de tiempo al inicio del procesamiento del frame
            processed_frame = self.process_frame(frame=frame)
            frame_end_time = time.time()  # Marca de tiempo después del procesamiento
            
            frame_times.put(frame_end_time - frame_start_time)
            if frame_times.full():
                frame_times.get()  # Mantener el tamaño de la cola manejable
                
            total_frames_processed += 1
            total_time_elapsed += frame_end_time - frame_start_time

            # Calcula y muestra la tasa de FPS y la latencia
            average_fps = total_frames_processed / total_time_elapsed
            average_latency = sum(list(frame_times.queue)) / frame_times.qsize()

            cv2.putText(processed_frame, f"FPS: {average_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)
            cv2.putText(processed_frame, f"Latency: {average_latency*1000:.2f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)
            
            cv2.imshow("frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
    
    def annotate_frame(
        #CREA UNA COPIA DEL FRAME PARA PODER UTILIZARLO
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        labels = [
            f"#{tracker_id}" 
            for tracker_id 
            in detections.tracker_id
            ]
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        return annotated_frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        #PASA EL FRAME ACTUAL AL MODELO PARA SU ANÁLISIS
        results = self.model(
            frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold
        )[0]
        #TRANSFORMA LOS RESULTADOS EN UN OBJETO "DETECTIONS" CON SUPERVISION DE ROBOFLOW
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        return self.annotate_frame(frame=frame, detections=detections)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )
    
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--ip_address",
        required=True,
        help="IP address of the video camera",
        type=str,
        )
    parser.add_argument(
        "--port",
        required=True,
        help="Port of the video stream",
        type=int,
        )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    
    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        ip_address=args.ip_address,
        port=args.port,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()
    
#------------------------------- BACKUP MULTIHILO
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
    def __init__(self, source_weights_path: str, ip_address: str, port: int, confidence_threshold: float = 0.3, iou_threshold: float = 0.7) -> None:
        self.ip_address = ip_address
        self.port = port
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(color=COLORS)

        # Cola para frames
        self.frame_queue = queue.Queue(maxsize=10)
        # Control de hilos
        self.capture_running = True
        self.processing_running = True

    def capture_video(self):
        video_path = f"http://{self.ip_address}:{self.port}/video"
        vid = cv2.VideoCapture(video_path)
        while self.capture_running:
            ret, frame = vid.read()
            if not ret:
                self.capture_running = False
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def process_video(self):
        frame_times = queue.Queue(maxsize=30)  # Para almacenar tiempos de procesamiento y calcular la latencia
        total_frames_processed = 0
        total_time_elapsed = 0

        while self.processing_running or not self.frame_queue.empty():
            if not self.frame_queue.empty():
                frame_start_time = time.time()  # Marca de tiempo al inicio del procesamiento del frame
                frame = self.frame_queue.get()
                
                processed_frame = self.process_frame(frame=frame)
                
                frame_end_time = time.time()  # Marca de tiempo después del procesamiento
                frame_times.put(frame_end_time - frame_start_time)
                if frame_times.full():
                    frame_times.get()  # Mantener el tamaño de la cola manejable
                
                total_frames_processed += 1
                total_time_elapsed += frame_end_time - frame_start_time

                # Calcula y muestra la tasa de FPS y la latencia
                average_fps = total_frames_processed / total_time_elapsed
                average_latency = sum(list(frame_times.queue)) / frame_times.qsize()

                cv2.putText(processed_frame, f"FPS: {average_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)
                cv2.putText(processed_frame, f"Latency: {average_latency*1000:.2f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)
                
                cv2.imshow("Processed Frame", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.processing_running = False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        return self.annotate_frame(frame=frame, detections=detections)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame

    def start(self):
        # Iniciar hilos de captura y procesamiento
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



#Empezando con las simulaciones 2D--------------------------------------------------------------------------------

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

    def capture_video(self):
        #video_path = f"http://{self.ip_address}:{self.port}/video"
        video_path = "data/lento.mp4"
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

        annotated_frame = self.annotate_frame(frame, detections)
        return annotated_frame

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
    
    #-----------------------------------------------------------
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
import pymunk

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
        
        self.taco_positions = []  # Lista para almacenar las posiciones de la punta del taco

        self.average_ball_radius = None  # Inicializar el radio promedio de las bolas como None

    def capture_video(self):
        video_path = "data/lento.mp4"
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
        
        # Ejemplo de cómo podrías llamar a calculate_average_ball_radius
        if self.average_ball_radius is None and len(detections) > 0:
            self.calculate_average_ball_radius(detections)
        
        annotated_frame = frame.copy()

        for i in range(len(detections.xyxy)):
            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]
            tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None
            confidence = detections.confidence[i]

            if class_id == 2:  # Si el objeto detectado es el taco
                # En process_frame, cuando llamas a process_taco, pasa detections también
                self.process_taco(annotated_frame, bbox, detections)
            else:
                # Dibujar todas las detecciones, excepto el taco
                x1, y1, x2, y2 = map(int, bbox)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                radius = int(max(x2 - x1, y2 - y1) / 2)

                # Asegura que class_id sea positivo antes de ajustar por el índice base-0.
                if class_id > 0:
                    color_bgr = COLORS.by_idx(class_id - 1).as_bgr()
                else:
                    # Para class_id = 0 o cualquier otro caso no esperado, usa un color predeterminado (e.g., rojo)
                    color_bgr = (0, 0, 255)  # BGR para rojo

                cv2.circle(annotated_frame, center, radius, color_bgr, 2)
                
        return annotated_frame

    #Función para mejorar el refinamiento de contornos    
    def process_taco(self, frame, bbox, detections):
        x1, y1, x2, y2 = map(int, bbox)
        taco_subimage = frame[y1:y2, x1:x2]

        # Convertir a escala de grises y aplicar detección de bordes
        gray = cv2.cvtColor(taco_subimage, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Encontrar contornos en la subimagen
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lista para almacenar las posiciones de las bolas detectadas (actualizar según tu implementación actual)
        ball_positions = self.get_ball_positions(detections)  # Usa detections directamente

        best_contour = None
        min_distance_to_ball = float('inf')  # Inicializar con un valor alto

        for contour in contours:
            # Evaluar la linealidad y orientación del contorno
            if self.is_contour_linear_and_properly_oriented(contour):
                # Calcular la distancia del contorno a la bola más cercana
                contour_center = self.get_contour_center(contour)
                distance, closest_ball_position = self.get_distance_to_closest_ball(contour_center, ball_positions)

                # Verificar si el contorno está suficientemente cerca de alguna bola
                if distance < min_distance_to_ball:
                    min_distance_to_ball = distance
                    best_contour = contour

        # Actualización para manejar la trayectoria del taco
        if best_contour is not None:
            # Asumiendo que best_contour es la punta del taco
            contour_center = self.get_contour_center(best_contour)
            if contour_center:
                self.update_taco_position(contour_center)
                direction = self.calculate_taco_direction()
                #if direction:
                #    print(f"Taco direction: {direction}")  # Solo para depuración

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def update_taco_position(self, new_position):
        # Asegúrate de que new_position es una tupla o lista con dos elementos
        if new_position is not None and len(new_position) == 2:
            x, y = new_position  # Desempaquetar las coordenadas x e y
            self.taco_positions.append(pymunk.Vec2d(x, y))  # Pasar x e y como argumentos separados
            if len(self.taco_positions) > 10:  # Mantener solo las últimas 10 posiciones
                self.taco_positions.pop(0)
            #print(f"Updated taco positions: {[str(pos) for pos in self.taco_positions]}")  # Imprime las posiciones actuales del taco
        else:
            print("New position is None or not correctly formatted.")

    def calculate_taco_direction(self):
        if len(self.taco_positions) >= 2:
            # Calcular la dirección como la diferencia entre las dos últimas posiciones
            direction = self.taco_positions[-1] - self.taco_positions[-2]
            #print(f"Calculated taco direction: {direction}")  # Imprime la dirección calculada
            return direction
        return None

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

    def is_contour_linear_and_properly_oriented(self, contour, linearity_threshold=5, angle_threshold=30):
        # Ajustar los puntos del contorno para cv2.fitLine
        rows, cols = contour.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calcular la distancia promedio de los puntos del contorno a la línea ajustada
        distances = []
        for point in contour:
            point_x, point_y = point[0][0], point[0][1]
            line_y_at_x = (vy / vx) * (point_x - x) + y
            distance = np.abs(point_y - line_y_at_x)
            distances.append(distance)
        average_distance = np.mean(distances)

        # Evaluar la linealidad basada en la distancia promedio
        is_linear = average_distance <= linearity_threshold

        # Calcular la orientación de la línea y evaluar si está adecuadamente orientada
        angle = np.arctan2(vy, vx) * (180 / np.pi)
        is_properly_oriented = (-angle_threshold <= angle <= angle_threshold) or (180 - angle_threshold <= angle <= 180 + angle_threshold)

        return is_linear and is_properly_oriented
    
    def get_ball_positions(self, detections):
        ball_positions = []
        for i in range(len(detections.xyxy)):
            class_id = detections.class_id[i]
            if class_id in [0, 1]:  # Asumiendo que 0 y 1 son las clases de las bolas
                bbox = detections.xyxy[i]
                x_center = int((bbox[0] + bbox[2]) / 2)
                y_center = int((bbox[1] + bbox[3]) / 2)
                ball_positions.append((x_center, y_center))
        return ball_positions

    def get_contour_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    
    def get_distance_to_closest_ball(self, contour_center, ball_positions):
        if contour_center is None:
            return float('inf'), None  # Retornar un valor alto para la distancia y None para la posición

        min_distance = float('inf')
        closest_ball_position = None
        for ball_position in ball_positions:
            distance = np.sqrt((contour_center[0] - ball_position[0]) ** 2 + (contour_center[1] - ball_position[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_ball_position = ball_position
        return min_distance, closest_ball_position
    
    def calculate_average_ball_radius(self, detections):
        if not detections.xyxy.size:  # Verifica si la lista de detecciones está vacía
            print("No ball detections available. Using default average radius.")
            self.average_ball_radius = 15  # Establece el radio promedio a un valor predeterminado
            return  # Sale de la función

        # Continúa con el cálculo si hay detecciones
        radii = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = bbox
            radius = ((x2 - x1) + (y2 - y1)) / 4  # Calcula el radio para cada detección
            radii.append(radius)
        
        if radii:  # Asegura que la lista de radios no esté vacía
            self.average_ball_radius = np.mean(radii)
            print(f"Average ball radius updated: {self.average_ball_radius}")
        else:
            print("No valid ball detections. Using default average radius.")
            self.average_ball_radius = 15  # Establece el radio promedio a un valor predeterminado si no hay detecciones válidas
    
    #Simulación 2D
    def initialize_simulation_space(self):
        self.space = pymunk.Space()  # Crear espacio de simulación
        self.space.gravity = (0, 0)  # La gravedad puede ser cero para este contexto

        # Preparar bolas de billar como objetos en el espacio de simulación
        self.ball_bodies = []
        for position in self.get_ball_positions():
            ball_body = pymunk.Body(1, pymunk.inf, body_type=pymunk.Body.KINEMATIC)
            ball_body.position = pymunk.Vec2d(position)
            #ball_shape = pymunk.Circle(ball_body, 5)  # Asumiendo un radio de 5 para las bolas
            # Usar self.average_ball_radius para el radio de la bola
            ball_shape = pymunk.Circle(ball_body, self.average_ball_radius)
            self.space.add(ball_body, ball_shape)
            self.ball_bodies.append(ball_body)
    
    def simulate_taco_interaction(self, direction):
        # Asumiendo que direction es un vector pymunk.Vec2d con la dirección del taco
        # Crear un cuerpo temporal para simular el taco
        taco_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        taco_body.position = self.taco_positions[-1]  # La última posición conocida del taco
        taco_shape = pymunk.Segment(taco_body, pymunk.Vec2d(0, 0), direction, 5)
        self.space.add(taco_body, taco_shape)

        # Simular un paso en el espacio para ver si hay interacciones
        self.space.step(1/60.0)

        # Quitar el cuerpo del taco después de la simulación
        self.space.remove(taco_body, taco_shape)

        # Aquí, podrías analizar las posiciones de las bolas después de la simulación para visualizar las trayectorias
    
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