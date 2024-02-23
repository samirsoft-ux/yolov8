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
        video_path = f"http://{self.ip_address}:{self.port}/video"
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
        
        valid_detections_indices = [i for i, bbox in enumerate(detections.xyxy) 
                                    if cv2.pointPolygonTest(self.mesa_corners, ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2), False) >= 0 
                                    and detections.class_id[i] in [0, 1]]
        
        if valid_detections_indices:
            # Asegurar que cada componente requerido está disponible antes de intentar indexarlo
            valid_xyxy = detections.xyxy[valid_detections_indices]
            valid_class_id = detections.class_id[valid_detections_indices]
            valid_confidence = detections.confidence[valid_detections_indices] if detections.confidence is not None else None
            valid_tracker_id = detections.tracker_id[valid_detections_indices] if detections.tracker_id is not None else None

            # Crear un nuevo objeto Detections con los valores filtrados
            valid_detections = sv.Detections(xyxy=valid_xyxy, class_id=valid_class_id, confidence=valid_confidence, tracker_id=valid_tracker_id)
        else:
            # Asegurar que los arrays vacíos tengan la forma adecuada
            valid_detections = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.array([], dtype=int), confidence=np.array([], dtype=float), tracker_id=np.array([], dtype=int))
        
        detections = self.tracker.update_with_detections(valid_detections)
    
        annotated_frame = self.annotate_frame(frame=frame, detections=detections)
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
