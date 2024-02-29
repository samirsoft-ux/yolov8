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
import pygame
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

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
            self.mesa_corners = self.reordenar_esquinas(np.array(data['l_circle_projector']))
        
        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.frame_queue = queue.Queue(maxsize=10)
        self.shutdown_event = threading.Event()
        self.space = self.setup_space()  # Configura el espacio de simulación de Pymunk
        self.historial_lineas = []  # Para almacenar las últimas posiciones y ángulos de la línea
        self.max_historial = 3  # Número máximo de elementos en el historial
        self.ultimo_angulo_valido = None # En tu inicialización de clase, agrega un valor para rastrear el último ángulo válido
    
    # Función para reordenar las esquinas
    def reordenar_esquinas(self, corners):
        # Asumimos que corners es una lista de puntos (x, y) que representan las esquinas de la mesa

        # Paso 1: Encontrar el centroide de todos los puntos
        centroide = np.mean(corners, axis=0)

        # Paso 2: Calcular el ángulo de cada punto respecto al centroide
        # La función atan2 retorna el ángulo entre el eje x positivo y el punto (y, x), siendo y vertical y x horizontal
        angulos = np.arctan2(corners[:, 1] - centroide[1], corners[:, 0] - centroide[0])

        # Paso 3: Ordenar los puntos basados en su ángulo respecto al centroide
        # Esto asegura un ordenamiento consecutivo alrededor del centroide
        orden = np.argsort(angulos)

        # Paso 4: Reordenar los puntos usando el índice obtenido
        corners_ordenados = corners[orden]

        return corners_ordenados
    
    def setup_space(self):
        space = pymunk.Space()
        space.gravity = (0, 0)  # No hay gravedad en una mesa de billar

        # Añadir bordes de la mesa como objetos estáticos
        for i in range(len(self.mesa_corners)):
            # Tomar cada par de puntos consecutivos como un segmento
            start = tuple(self.mesa_corners[i])  # Convertir a tupla
            end = tuple(self.mesa_corners[(i + 1) % len(self.mesa_corners)])  # Convertir a tupla y ciclo al primer punto después del último
            
            shape = pymunk.Segment(space.static_body, start, end, 0.5)  # 0.5 es el grosor del borde
            shape.elasticity = 1.0  # Coeficiente de restitución para simular el rebote de las bolas
            space.add(shape)

        return space
    
    """def visualizar_bordes_mesa(self, dimensiones_pantalla=(1920, 1080)):
        print(self.mesa_corners)
        pygame.init()
        pantalla = pygame.display.set_mode(dimensiones_pantalla)
        reloj = pygame.time.Clock()
        corriendo = True

        draw_options = pymunk.pygame_util.DrawOptions(pantalla)

        while corriendo:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    corriendo = False

            pantalla.fill((255, 255, 255))  # Fondo blanco
            self.space.debug_draw(draw_options)  # Dibuja los objetos de Pymunk

            pygame.display.flip()  # Actualiza la pantalla
            reloj.tick(60)  # Limita a 60 FPS

        pygame.quit()"""
    
    def suavizar_linea(self):
        if len(self.historial_lineas) == 0:
            return None, None
        centro_promedio = np.mean([centro for centro, _ in self.historial_lineas], axis=0)
        angulo_promedio = np.mean([angulo for _, angulo in self.historial_lineas])

        # Limita el cambio de ángulo para evitar saltos bruscos
        if self.ultimo_angulo_valido is not None:
            delta_angulo = min(10, abs(angulo_promedio - self.ultimo_angulo_valido))  # Limita a 10 grados de cambio
            angulo_promedio = self.ultimo_angulo_valido + np.sign(angulo_promedio - self.ultimo_angulo_valido) * delta_angulo

        self.ultimo_angulo_valido = angulo_promedio
        return centro_promedio, angulo_promedio

    def capture_video(self):
        #video_path = f"http://{self.ip_address}:{self.port}/video"
        video_path = "data/len2.mp4"
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
    
    # LA CONDICIONAL ES QUE ESTE MÁS CERCA DEL CENTRO DE UNA BOLA   
    def handle_detections(self, frame, detections):
        centros_bolas = []

        # Primero, recopilar los centros de todas las bolas detectadas
        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]
            if class_id in [0, 1]:  # Bolas
                centro_bola_x = (bbox[0] + bbox[2]) / 2
                centro_bola_y = (bbox[1] + bbox[3]) / 2
                centros_bolas.append((centro_bola_x, centro_bola_y))

        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]

            if class_id == 2:  # Taco
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                    _, binary_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contornos, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contornos:
                        contorno_taco = max(contornos, key=cv2.contourArea)
                        rect = cv2.minAreaRect(contorno_taco)

                        centro, (ancho, alto), angulo = rect
                        if ancho < alto:
                            angulo += 90

                        self.historial_lineas.append(((centro[0] + x1, centro[1] + y1), angulo))
                        if len(self.historial_lineas) > self.max_historial:
                            self.historial_lineas.pop(0)

                        centro_suavizado, angulo_suavizado = self.suavizar_linea()

                        if centro_suavizado is not None and angulo_suavizado is not None:
                            direccion_suavizada = np.array([np.cos(np.deg2rad(angulo_suavizado)), np.sin(np.deg2rad(angulo_suavizado))])
                            longitud_linea = max(frame.shape) * 2
                            punto_inicio_suavizado = np.array(centro_suavizado) - direccion_suavizada * longitud_linea / 2
                            punto_final_suavizado = np.array(centro_suavizado) + direccion_suavizada * longitud_linea / 2
                            
                            # Ahora, determinar cuál extremo de la trayectoria está más cerca a algún centro de bola
                            distancia_minima = float('inf')
                            extremo_cercano = None
                            for centro_bola in centros_bolas:
                                distancia_inicio = np.linalg.norm(punto_inicio_suavizado - centro_bola)
                                distancia_final = np.linalg.norm(punto_final_suavizado - centro_bola)
                                if distancia_inicio < distancia_minima:
                                    distancia_minima = distancia_inicio
                                    extremo_cercano = punto_inicio_suavizado
                                if distancia_final < distancia_minima:
                                    distancia_minima = distancia_final
                                    extremo_cercano = punto_final_suavizado
                            
                            # Dibuja la parte de la trayectoria del taco más cercana a la bola
                            if extremo_cercano is not None:
                                cv2.line(frame, tuple(np.int32(centro_suavizado)), tuple(np.int32(extremo_cercano)), (0, 255, 0), 2)

                else:
                    print("ROI vacío o de tamaño inválido.")
        return frame

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
    #processor.visualizar_bordes_mesa()
    processor.start()