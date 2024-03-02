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
import json
import pymunk
import pygame
import pymunk.pygame_util
from pygame.locals import QUIT
import time

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
    
        #Inicializaciones para simulación 2D
        self.ball_updates = {}  # Para almacenar las actualizaciones de las bolas antes de aplicarlas
        self.balls = {}  # Para almacenar las bolas presentes en el espacio de simulación
        self.active_balls = []  # Lista para mantener los IDs de las bolas activas (detectadas)
        #Calculo Radio 
        self.initial_frames_for_average = 10  # Número de frames para calcular el promedio
        self.detected_radii = []  # Para acumular los radios detectados
        self.frames_processed = 0  # Contador de frames procesados
        #Problemas de seguridad
        self.balls_to_remove = set()
        
        self.cue_removal_needed = False
    
    #BOLAS----------------------------------------------
    def add_ball_to_space(self, position, radius=10, mass=1):
        """
        Añade una bola al espacio de Pymunk y devuelve su objeto Circle.
        """
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, moment)
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.95  # Coeficiente de elasticidad de la bola
        shape.friction = 0.9  # fricción de la bola
        self.space.add(body, shape)
        return shape
    
    #MEJORAR para las bolas
    #Considera la interacción entre la bola y la mesa (por ejemplo, efectos de rotación o "spin") al definir las propiedades físicas, lo que puede requerir ajustes adicionales en las propiedades de las bolas y la mesa.
    #Fricción (shape.friction): Un coeficiente de fricción de 0.9 para la bola podría ser ligeramente alto para una mesa de billar, que generalmente permite que las bolas deslicen suavemente. Las mesas de billar están diseñadas para minimizar la fricción, por lo que un valor más bajo podría ser más realista, alrededor de 0.2 a 0.4, dependiendo del material de la superficie y de cómo quieras que se comporte la bola después del impacto y durante el movimiento.

    def prepare_ball_update(self, tracker_id, position, radius=None):
        if radius is None:
            # Usa el radio promedio si está disponible, de lo contrario usa un valor fijo
            radius = getattr(self, 'average_ball_radius', 18)
        self.ball_updates[tracker_id] = (position, radius)
    
    def process_ball_updates(self, space, data):
        # Añadir o actualizar bolas
        for tracker_id, (position, radius) in self.ball_updates.items():
            if tracker_id in self.balls:
                # Actualiza la posición de la bola existente
                self.balls[tracker_id].body.position = position
            else:
                # Añade una nueva bola al espacio
                self.balls[tracker_id] = self.add_ball_to_space(position, radius, tracker_id)  # Asegúrate de que esta función no modifique directamente el espacio
        self.ball_updates.clear()

        # Eliminar bolas no detectadas
        current_active_balls = self.getCurrentActiveBalls()  # Asumiendo que esta función devuelve los tracker_id de las bolas actualmente activas
        balls_to_remove = [ball_id for ball_id in self.balls if ball_id not in current_active_balls]
        for ball_id in balls_to_remove:
            ball_shape = self.balls[ball_id]
            space.remove(ball_shape.body, ball_shape)  # Elimina la bola del espacio
            del self.balls[ball_id]
    
    def getCurrentActiveBalls(self):
        # Retorna la lista actual de tracker_id de bolas activas
        return self.active_balls
            
    def remove_undetected_balls(self, current_active_balls):
        balls_to_remove = set(self.balls.keys()) - set(current_active_balls)
        self.balls_to_remove.update(balls_to_remove)
    #BOLAS----------------------------------------------
    
    #TACO------------------------------------------------------
    
    def prepare_cue_for_addition(self, start_point, end_point, thickness=5):
        """
        Prepara los datos necesarios para añadir el taco al espacio de Pymunk usando un callback.
        """
        # Almacena los datos del taco para usarlos en el callback
        self.cue_data = (start_point, end_point, thickness)
    
    def cue_callback(self, space, data):
        if hasattr(self, 'cue_data'):
            start_point, end_point, thickness = self.cue_data
            if hasattr(self, 'cue_shape'):
                self.space.remove(self.cue_shape.body, self.cue_shape)
            
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            shape = pymunk.Segment(body, start_point, end_point, thickness)
            shape.elasticity = 1.0
            shape.friction = 0.5
            self.space.add(body, shape)
            self.cue_shape = shape
    
    def prepare_cue_for_removal(self):
        if hasattr(self, 'cue_shape'):
            self.cue_removal_needed = True
            
    def remove_cue_callback(self, space, data):
        if hasattr(self, 'cue_shape'):
            space.remove(self.cue_shape.body, self.cue_shape)
            del self.cue_shape

    
    #TACO------------------------------------------------------
    
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
        
    def handle_detections(self, frame, detections):
        # No se modifican los centros de las bolas para la lógica de la trayectoria del taco
        centros_bolas = []
        current_active_balls = []
        taco_detected = False
        
        if self.frames_processed < self.initial_frames_for_average:
            for i, bbox in enumerate(detections.xyxy):
                class_id = detections.class_id[i]
                if class_id in [0, 1]:  # Bolas
                    radio = ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 4
                    self.detected_radii.append(radio)

            self.frames_processed += 1

            if self.frames_processed == self.initial_frames_for_average:
                if self.detected_radii:
                    self.average_ball_radius = sum(self.detected_radii) / len(self.detected_radii)
                    print (self.average_ball_radius)
                else:
                    self.average_ball_radius = 18  # Valor fijo si no hay detecciones
        
        # Procesamiento de detecciones para bolas y actualización/preparación para Pymunk
        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]
            tracker_id = detections.tracker_id[i]

            if class_id in [0, 1]:  # Bolas
                centro_bola_x = (bbox[0] + bbox[2]) / 2
                centro_bola_y = (bbox[1] + bbox[3]) / 2
                centros_bolas.append((centro_bola_x, centro_bola_y))
                # Actualizar o añadir la bola en el espacio de Pymunk de forma preparatoria
                self.prepare_ball_update(tracker_id, (centro_bola_x, centro_bola_y))
                current_active_balls.append(tracker_id)

        # Actualizar la lista de bolas activas
        self.active_balls = current_active_balls

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
                                # Preparar los datos para el callback
                                self.prepare_cue_for_addition(tuple(np.int32(centro_suavizado)), tuple(np.int32(extremo_cercano)), 5)
                                taco_detected = True

                else:
                    print("ROI vacío o de tamaño inválido.")
        if not taco_detected:
            self.prepare_cue_for_removal()
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
        # Inicializar Pygame y configurar la ventana
        pygame.init()
        pantalla = pygame.display.set_mode((1920, 1080))
        reloj = pygame.time.Clock()
        corriendo = True
        draw_options = pymunk.pygame_util.DrawOptions(pantalla)

        font = pygame.font.Font(None, 36)  # Fuente para el texto de rendimiento

        # Variables para calcular FPS y latencia
        frame_count = 0
        start_time = pygame.time.get_ticks()
        fps = 0  # Inicializa FPS
        latency = 0  # Inicializa la latencia

        # Configuración de colores y visualización
        color_fondo = (0, 0, 0)  # Fondo negro para proyección

        # Iniciar los hilos de captura y procesamiento de video
        capture_thread = threading.Thread(target=self.capture_video)
        processing_thread = threading.Thread(target=self.process_video)
        capture_thread.start()
        processing_thread.start()

        # Antes del bucle while en start()
        self.space.user_data = self  # Esto permite acceder a self dentro de los callbacks de Pymunk

        while corriendo:
            for evento in pygame.event.get():
                if evento.type == QUIT:
                    corriendo = False
                    self.shutdown_event.set()

            frame_start_time = time.time()  # Inicio de procesamiento del frame actual
            
            pantalla.fill(color_fondo)  # Fondo blanco

            # Programar el post-step callback para procesar actualizaciones de bolas
            # Hacerlo aquí asegura que se procesen después del próximo paso de simulación
            self.space.add_post_step_callback(self.process_ball_updates, None)

            # Avanzar la simulación de Pymunk
            self.space.step(1/48.0)

            #if hasattr(self, 'cue_data'):
            #    self.space.add_post_step_callback(self.cue_callback, None)
            #    delattr(self, 'cue_data')  # Elimina cue_data para evitar añadirlo múltiples veces

            if hasattr(self, 'cue_data'):
                self.space.add_post_step_callback(self.cue_callback, self.cue_data)
                delattr(self, 'cue_data')
            if hasattr(self, 'cue_removal_needed') and self.cue_removal_needed:
                self.space.add_post_step_callback(self.remove_cue_callback, None)
                self.cue_removal_needed = False
            
            # Visualizar la simulación de Pymunk
            self.space.debug_draw(draw_options)

            # Cálculo de FPS y latencia (solo se actualiza cada segundo)
            current_time = pygame.time.get_ticks()
            if current_time - start_time > 1000:
                fps = frame_count / ((current_time - start_time) / 1000.0)
                latency = (time.time() - frame_start_time) * 1000  # Latencia en milisegundos
                start_time = current_time
                frame_count = 0

            # Mostrar FPS y latencia constantemente
            fps_text = font.render(f"FPS: {fps:.2f}", True, (0, 255, 0))
            latency_text = font.render(f"Latency: {latency:.2f} ms", True, (0, 255, 0))
            pantalla.blit(fps_text, (50, 50))
            pantalla.blit(latency_text, (50, 90))

            frame_count += 1

            pygame.display.flip()  # Actualizar la pantalla
            reloj.tick(48)  # 60 FPS

        # Esperar a que los hilos de captura y procesamiento finalicen
        capture_thread.join()
        processing_thread.join()
        pygame.quit()
        cv2.destroyAllWindows()
        
    """def start(self):
        # Inicializar Pygame y configurar la ventana
        pygame.init()
        pantalla = pygame.display.set_mode((1920, 1080))
        reloj = pygame.time.Clock()
        corriendo = True
        draw_options = pymunk.pygame_util.DrawOptions(pantalla)
        draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES  # Solo dibujar formas, no conexiones ni colisiones

        font = pygame.font.Font(None, 36)  # Fuente para el texto de rendimiento

        # Configuración de colores y visualización
        color_fondo = (0, 0, 0)  # Fondo negro para proyección
        color_bola = (255, 255, 255)  # Contornos blancos para las bolas

        # Variables para calcular FPS y latencia
        frame_count = 0
        start_time = pygame.time.get_ticks()
        fps = 0  # Inicializa FPS
        latency = 0  # Inicializa la latencia

        # Iniciar los hilos de captura y procesamiento de video
        capture_thread = threading.Thread(target=self.capture_video)
        processing_thread = threading.Thread(target=self.process_video)
        capture_thread.start()
        processing_thread.start()

        while corriendo:
            for evento in pygame.event.get():
                if evento.type == QUIT:
                    corriendo = False
                    self.shutdown_event.set()

            frame_start_time = time.time()  # Inicio de procesamiento del frame actual
            
            pantalla.fill(color_fondo)

            # Avanzar la simulación de Pymunk
            self.space.step(1/48.0)

            # Visualizar la simulación de Pymunk con ajustes personalizados
            for ball_id, ball in self.balls.items():
                # Dibuja solo el contorno de cada bola
                pygame.draw.circle(pantalla, color_bola, (int(ball.body.position.x), int(ball.body.position.y)), ball.shape.radius, 1)

            # Programar el post-step callback para procesar actualizaciones de bolas
            # Hacerlo aquí asegura que se procesen después del próximo paso de simulación
            self.space.add_post_step_callback(self.process_ball_updates, None)
            
            # Visualizar la simulación de Pymunk
            self.space.debug_draw(draw_options)

            # Cálculo de FPS y latencia (solo se actualiza cada segundo)
            current_time = pygame.time.get_ticks()
            if current_time - start_time > 1000:
                fps = frame_count / ((current_time - start_time) / 1000.0)
                latency = (time.time() - frame_start_time) * 1000  # Latencia en milisegundos
                start_time = current_time
                frame_count = 0

            # Mostrar FPS y latencia constantemente
            fps_text = font.render(f"FPS: {fps:.2f}", True, (0, 255, 0))
            latency_text = font.render(f"Latency: {latency:.2f} ms", True, (0, 255, 0))
            pantalla.blit(fps_text, (50, 50))
            pantalla.blit(latency_text, (50, 90))

            frame_count += 1

            pygame.display.flip()  # Actualizar la pantalla
            reloj.tick(48)  # 60 FPS

        # Esperar a que los hilos de captura y procesamiento finalicen
        capture_thread.join()
        processing_thread.join()
        pygame.quit()
        cv2.destroyAllWindows()"""

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