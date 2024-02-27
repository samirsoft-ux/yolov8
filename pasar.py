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
        self.space = pymunk.Space()  # Crear espacio de simulación aquí
        self.space.gravity = (0, 0)  # Definir gravedad, si necesaria
        self.ball_shapes = {}  # Diccionario para mantener las formas de las bolas por tracker_id        

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
        
        # Actualizar detecciones con ByteTrack para obtener tracker_id
        detections = self.tracker.update_with_detections(detections)
        
        # Ejemplo de cómo podrías llamar a calculate_average_ball_radius
        if self.average_ball_radius is None and len(detections) > 0:
            self.calculate_average_ball_radius(detections)
        
        # Imprimir la estructura de detections para depuración
        print(f"Tipo de detections: {type(detections)}")
        print(f"Atributos disponibles en detections: {dir(detections)}")
        if hasattr(detections, 'xyxy'):
            print(f"xyxy: {detections.xyxy}")
        if hasattr(detections, 'tracker_id'):
            print(f"tracker_id: {detections.tracker_id}")
        if hasattr(detections, 'class_id'):
            print(f"class_id: {detections.class_id}")
        if hasattr(detections, 'confidence'):
            print(f"confidence: {detections.confidence}")
        
        annotated_frame = frame.copy()

        # Actualiza las posiciones de las bolas en la simulación basadas en las detecciones actuales
        self.update_simulation_balls(detections)

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

    def update_simulation_balls(self, detections):
        for i, bbox in enumerate(detections.xyxy):
            tracker_id = detections.tracker_id[i]
            position = pymunk.Vec2d((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            if tracker_id not in self.ball_shapes:
                ball_body = pymunk.Body(1, pymunk.inf, body_type=pymunk.Body.KINEMATIC)
                ball_body.position = position
                ball_shape = pymunk.Circle(ball_body, self.average_ball_radius)
                self.space.add(ball_body, ball_shape)
                self.ball_shapes[tracker_id] = ball_shape
            else:
                self.ball_shapes[tracker_id].body.position = position
  
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
                if direction:
                    new_ball_positions = self.simulate_taco_interaction(direction)
                    self.visualize_simulation_result(frame, new_ball_positions)
                    #self.simulate_taco_interaction(direction)  # Simula la interacción del taco
                #    print(f"Taco direction: {direction}")  # Solo para depuración

        # Dibujar el rectángulo del taco para visualización
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
    
    def simulate_taco_interaction(self, direction):
        # Asumiendo que direction es un vector pymunk.Vec2d con la dirección del taco
        taco_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        taco_body.position = self.taco_positions[-1]
        taco_shape = pymunk.Segment(taco_body, pymunk.Vec2d(0, 0), direction, self.average_ball_radius)
        self.space.add(taco_body, taco_shape)

        # Simular múltiples pasos en el espacio para ver cómo evolucionan las interacciones
        for _ in range(10):  # Simula 10 pasos, ajusta según sea necesario
            self.space.step(1/60.0)

        # Recopilar las nuevas posiciones de las bolas
        new_ball_positions = [ball_body.position for ball_body in self.ball_bodies]

        # Quitar el cuerpo del taco después de la simulación
        self.space.remove(taco_body, taco_shape)

        # Devolver las nuevas posiciones para su visualización
        return new_ball_positions

    def visualize_simulation_result(self, frame, new_ball_positions):
        for position in new_ball_positions:
            # Convertir posición pymunk a coordenadas de OpenCV
            x, y = int(position.x), int(position.y)
            # Dibujar un círculo en la nueva posición de la bola
            cv2.circle(frame, (x, y), self.average_ball_radius, (0, 255, 255), 2)  # Color amarillo para la visualización
    
    
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
                if direction:
                    new_ball_positions = self.simulate_taco_interaction(direction)
                    self.visualize_simulation_result(frame, new_ball_positions)

        # Dibujar el rectángulo del taco para visualización
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)