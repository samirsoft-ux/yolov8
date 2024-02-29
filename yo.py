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

def visualizar_trayectoria(frame, inicio, fin, color=(0, 255, 255), grosor=2):
    # Asegurarse de que los puntos sean enteros
    inicio = (int(inicio[0]), int(inicio[1]))
    fin = (int(fin[0]), int(fin[1]))
    cv2.line(frame, inicio, fin, color, grosor)



def visualizar_trayectoria(frame, inicio, fin, color=(0, 255, 255), grosor=2):
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
        self.direcciones_taco = []  # Almacena las últimas direcciones del taco

        self.historial_lineas = []  # Para almacenar las últimas posiciones y ángulos de la línea
        self.max_historial = 48  # Número máximo de elementos en el historial

        # En tu inicialización de clase, agrega un valor para rastrear el último ángulo válido
        self.ultimo_angulo_valido = None

    # Define una función para calcular la dirección del taco suavizada
    """def calcular_direccion_taco_suavizada(self, nuevas_direcciones):
        # Agrega la última dirección calculada a la lista
        self.direcciones_taco.append(nuevas_direcciones)
        # Limita el tamaño de la lista a las últimas N direcciones
        if len(self.direcciones_taco) > 35:  # Por ejemplo, N = 5
            self.direcciones_taco.pop(0)
        # Calcula el promedio de las direcciones
        direccion_promedio = np.mean(self.direcciones_taco, axis=0)

        # Calcula la norma de la dirección promedio
        norma = np.linalg.norm(direccion_promedio)
        if norma == 0:
            return None  # Retorna None o alguna dirección predeterminada si la norma es 0
        return direccion_promedio / norma  # Retorna la dirección normalizada"""

    """def suavizar_linea(self):
        if len(self.historial_lineas) == 0:
            return None, None  # No hay suficiente información para suavizar
        centro_promedio = np.mean([centro for centro, _ in self.historial_lineas], axis=0)
        angulo_promedio = np.mean([angulo for _, angulo in self.historial_lineas])
        return centro_promedio, angulo_promedio"""
    
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

    """def calcular_centro_mesa(self):
        # Asumiendo que self.mesa_corners contiene las coordenadas de las esquinas de la mesa
        # cargadas desde 'l_circle_projector' en data.json
        esquinas = np.array(self.mesa_corners)  # Convertir a numpy array para facilitar los cálculos
        centro_x = np.mean(esquinas[:, 0])
        centro_y = np.mean(esquinas[:, 1])
        return centro_x, centro_y"""

    def calcular_centro_mesa_genérico(self, frame):
        altura, anchura = frame.shape[:2]  # Obtiene las dimensiones del frame
        centro_mesa_x = anchura // 2  # Calcula el centro en X
        centro_mesa_y = altura // 2   # Calcula el centro en Y
        return centro_mesa_x, centro_mesa_y

    """def handle_detections(self, frame, detections):
        # Calcula el centro de la mesa de manera genérica
        centro_mesa_x, centro_mesa_y = self.calcular_centro_mesa_genérico(frame)
        
        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]
            
            if class_id == 2:  # Si es un taco
                centro_taco_x = (bbox[0] + bbox[2]) / 2
                centro_taco_y = (bbox[1] + bbox[3]) / 2
                
                # Determinar la dirección general del taco basada en su posición relativa al centro de la mesa
                if centro_taco_x < centro_mesa_x:
                    # Supongamos que el taco está a la izquierda del centro y se moverá hacia la derecha
                    direccion_taco = [1, 0]  # Derecha
                elif centro_taco_x > centro_mesa_x:
                    # Supongamos que el taco está a la derecha del centro y se moverá hacia la izquierda
                    direccion_taco = [-1, 0]  # Izquierda
                elif centro_taco_y < centro_mesa_y:
                    # Supongamos que el taco está arriba del centro y se moverá hacia abajo
                    direccion_taco = [0, 1]  # Abajo
                elif centro_taco_y > centro_mesa_y:
                    # Supongamos que el taco está abajo del centro y se moverá hacia arriba
                    direccion_taco = [0, -1]  # Arriba

                # Calcula la posición final basada en la dirección y la longitud deseada de la proyección
                longitud_proyeccion = 100  # Longitud arbitraria de la proyección de la dirección
                punto_final = (centro_taco_x + direccion_taco[0] * longitud_proyeccion, centro_taco_y + direccion_taco[1] * longitud_proyeccion)
                
                # Dibuja la trayectoria del taco en el frame
                visualizar_trayectoria(frame, (centro_taco_x, centro_taco_y), punto_final, color=(0, 255, 255), grosor=2)"""
                
    """def handle_detections(self, frame, detections):
        centro_mesa_x, centro_mesa_y = self.calcular_centro_mesa_genérico(frame)
        
        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]
            
            if class_id == 2:  # Taco
                centro_taco_x = (bbox[0] + bbox[2]) / 2
                centro_taco_y = (bbox[1] + bbox[3]) / 2
                
                if not self.ubicaciones_taco:
                    self.ubicaciones_taco.append((centro_taco_x, centro_taco_y))
                    return  # Espera hasta tener al menos dos puntos para determinar la dirección
                    
                ultima_posicion = self.ubicaciones_taco[-1]
                self.ubicaciones_taco.append((centro_taco_x, centro_taco_y))
                
                # Calcular la dirección del movimiento
                delta_x = centro_taco_x - ultima_posicion[0]
                delta_y = centro_taco_y - ultima_posicion[1]
                
                # Determinar la dirección predominante del movimiento
                if abs(delta_x) > abs(delta_y):
                    direccion_taco = [np.sign(delta_x), 0]  # Movimiento predominante horizontal
                else:
                    direccion_taco = [0, np.sign(delta_y)]  # Movimiento predominante vertical
                
                # Calcula la posición final basada en la dirección predominante
                punto_final = (centro_taco_x + direccion_taco[0] * 100, centro_taco_y + direccion_taco[1] * 100)
                
                # Dibuja la trayectoria del taco en el frame
                visualizar_trayectoria(frame, (centro_taco_x, centro_taco_y), punto_final, color=(0, 255, 255), grosor=2)"""
                
    """def handle_detections(self, frame, detections):
        centro_mesa_x, centro_mesa_y = self.calcular_centro_mesa_genérico(frame)
        
        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]
            
            if class_id == 2:  # Taco
                centro_taco_x = (bbox[0] + bbox[2]) / 2
                centro_taco_y = (bbox[1] + bbox[3]) / 2
                
                # Asegurar que tenemos la posición inicial del taco para determinar su lado de entrada
                if len(self.ubicaciones_taco) == 0:
                    self.ubicaciones_taco.append((centro_taco_x, centro_taco_y))
                    # Determina el lado de entrada del taco basado en la posición inicial
                    if centro_taco_x < centro_mesa_x:
                        self.direccion_entrada_taco = "izquierda"
                    elif centro_taco_x > centro_mesa_x:
                        self.direccion_entrada_taco = "derecha"
                    elif centro_taco_y < centro_mesa_y:
                        self.direccion_entrada_taco = "arriba"
                    else:
                        self.direccion_entrada_taco = "abajo"
                else:
                    self.ubicaciones_taco.append((centro_taco_x, centro_taco_y))
                
                # Proyectar la trayectoria basada en el lado de entrada del taco
                if self.direccion_entrada_taco == "izquierda":
                    direccion_taco = [1, 0]  # Derecha
                elif self.direccion_entrada_taco == "derecha":
                    direccion_taco = [-1, 0]  # Izquierda
                elif self.direccion_entrada_taco == "arriba":
                    direccion_taco = [0, 1]  # Abajo
                else:  # "abajo"
                    direccion_taco = [0, -1]  # Arriba
                
                # Calcula la posición final basada en la dirección
                punto_final = (centro_taco_x + direccion_taco[0] * 100, centro_taco_y + direccion_taco[1] * 100)
                
                # Dibuja la trayectoria del taco en el frame
                visualizar_trayectoria(frame, (centro_taco_x, centro_taco_y), punto_final, color=(0, 255, 255), grosor=2)"""

    """def handle_detections(self, frame, detections):
        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]

            if class_id == 2:  # Taco
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                # Extraer la región de interés (ROI) donde se detecta el taco
                roi = frame[y1:y2, x1:x2]

                # Convertir la ROI a escala de grises
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Suavizar la ROI para reducir el ruido
                blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

                # Aplicar umbralización de Otsu para obtener una imagen binaria clara
                _, binary_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Preparar una imagen negra del tamaño del frame original
                black_image = np.zeros_like(frame)

                # Colocar la imagen binaria procesada (ROI) en la imagen negra
                # Asegurarse de que la ROI binaria se convierta de nuevo a BGR para coincidir con el frame
                black_image[y1:y1+binary_roi.shape[0], x1:x1+binary_roi.shape[1]] = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)

                # Devolver la imagen negra con el taco binario resaltado
                return black_image

        # Si no hay detecciones de taco, devolver el frame original
        return frame"""

    ## EL QUE MÁS ME SIRVIO

    """def handle_detections(self, frame, detections):
        centro_mesa_x, centro_mesa_y = self.calcular_centro_mesa_genérico(frame)

        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]

            if class_id == 2:  # Taco
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Asegurar que las coordenadas del recorte estén dentro de los límites de la imagen
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Verificar que el recorte tenga un tamaño válido
                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                    _, binary_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    contornos, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contornos:
                        contorno_taco = max(contornos, key=cv2.contourArea)
                        rect = cv2.minAreaRect(contorno_taco)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        # Ajustar las coordenadas del box al frame completo
                        box_global = box + [x1, y1]
                        cv2.drawContours(frame, [box_global], 0, (0, 0, 255), 2)  # Dibuja el rectángulo en rojo

                        centro, (ancho, alto), angulo = rect
                        if ancho < alto:
                            angulo += 90  # Ajustar el ángulo si el rectángulo está más orientado verticalmente

                        # Agregar el centro y ángulo actual al historial
                        self.historial_lineas.append(((centro[0] + x1, centro[1] + y1), angulo))
                        if len(self.historial_lineas) > self.max_historial:
                            self.historial_lineas.pop(0)

                        # Suavizar la línea usando el historial
                        centro_suavizado, angulo_suavizado = self.suavizar_linea()

                        if centro_suavizado is not None and angulo_suavizado is not None:
                            direccion_suavizada = np.array([np.cos(np.deg2rad(angulo_suavizado)), np.sin(np.deg2rad(angulo_suavizado))])
                            longitud_linea = max(frame.shape) * 2
                            punto_inicio_suavizado = np.array(centro_suavizado) - direccion_suavizada * longitud_linea / 2
                            punto_final_suavizado = np.array(centro_suavizado) + direccion_suavizada * longitud_linea / 2
                            
                            # Determinar cuál de los puntos está más cerca al centro de la mesa
                            if np.linalg.norm(punto_inicio_suavizado - np.array([centro_mesa_x, centro_mesa_y])) < np.linalg.norm(punto_final_suavizado - np.array([centro_mesa_x, centro_mesa_y])):
                                punto_cercano = punto_inicio_suavizado
                                punto_lejano = punto_final_suavizado
                            else:
                                punto_cercano = punto_final_suavizado
                                punto_lejano = punto_inicio_suavizado

                            # Dibujar solo la mitad de la línea más cercana al centro de la mesa
                            cv2.line(frame, tuple(np.int32(centro_suavizado)), tuple(np.int32(punto_cercano)), (0, 255, 0), 2)
                            
                            # Dibujar la línea descartada en otro color
                            cv2.line(frame, tuple(np.int32(centro_suavizado)), tuple(np.int32(punto_lejano)), (0, 0, 255), 2)
                else:
                    print("ROI vacío o de tamaño inválido.")
        return frame"""
        
    def handle_detections(self, frame, detections):
        centro_mesa_x, centro_mesa_y = self.calcular_centro_mesa_genérico(frame)

        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]

            if class_id == 2:  # Taco
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Asegurar que las coordenadas del recorte estén dentro de los límites de la imagen
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Verificar que el recorte tenga un tamaño válido
                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                    _, binary_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    contornos, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contornos:
                        contorno_taco = max(contornos, key=cv2.contourArea)
                        rect = cv2.minAreaRect(contorno_taco)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        # Ajustar las coordenadas del box al frame completo
                        box_global = box + [x1, y1]
                        cv2.drawContours(frame, [box_global], 0, (0, 0, 255), 2)  # Dibuja el rectángulo en rojo con grosor de 20

                        #centro, _, angulo = rect
                        #if angulo < -45:
                        #    angulo += 90

                        centro, (ancho, alto), angulo = rect
                        if ancho < alto:
                            angulo += 90  # Ajustar el ángulo si el rectángulo está más orientado verticalmente

                        
                        # Agregar el centro y angulo actual al historial
                        self.historial_lineas.append(((centro[0] + x1, centro[1] + y1), angulo))
                        if len(self.historial_lineas) > self.max_historial:
                            self.historial_lineas.pop(0)

                        # Suavizar la línea usando el historial
                        centro_suavizado, angulo_suavizado = self.suavizar_linea()

                        if centro_suavizado is not None and angulo_suavizado is not None:
                            direccion_suavizada = np.array([np.cos(np.deg2rad(angulo_suavizado)), np.sin(np.deg2rad(angulo_suavizado))])
                            longitud_linea = max(frame.shape) * 2
                            punto_inicio_suavizado = np.array(centro_suavizado) - direccion_suavizada * longitud_linea / 2
                            punto_final_suavizado = np.array(centro_suavizado) + direccion_suavizada * longitud_linea / 2

                            # Dibujar la línea completa
                            cv2.line(frame, tuple(np.int32(punto_inicio_suavizado)), tuple(np.int32(punto_final_suavizado)), (0, 255, 0), 2)
                else:
                    print("ROI vacío o de tamaño inválido.")
        return frame
        
    """def handle_detections(self, frame, detections):
        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]
            if class_id == 2:  # Taco
                x1, y1, x2, y2 = map(int, [bbox[0], bbox[1], bbox[2], bbox[3]])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

                if x2 <= x1 or y2 <= y1:
                    continue

                roi = frame[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                edges = cv2.Canny(blurred_roi, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Calcular el centroide de la detección del taco
                centroide_taco = ((x1 + x2) / 2, (y1 + y2) / 2)

                if contours:
                    selected_contour = None
                    max_alignment = 0  # Inicializar el máximo alineamiento con un valor bajo

                    for contour in contours:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            centroide_contorno = (cx, cy)

                            # Calcular la alineación respecto a la dirección horizontal
                            # En este ejemplo, simplemente comparamos la coordenada y del centroide del contorno
                            # con la coordenada y del centroide del taco para simular una "dirección deseada"
                            alignment = abs(centroide_contorno[1] - centroide_taco[1])

                            if alignment > max_alignment:
                                max_alignment = alignment
                                selected_contour = contour

                    # Dibujar el contorno seleccionado
                    if selected_contour is not None:
                        cv2.drawContours(roi, [selected_contour], -1, (0, 255, 0), 2)

                frame[y1:y2, x1:x2] = roi"""
        
    """def handle_detections(self, frame, detections):
        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]

            if class_id == 2:  # Taco
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                # Extraer la región de interés (ROI)
                roi = frame[y1:y2, x1:x2]

                # Convertir a escala de grises y umbralizar
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Encontrar contornos en la ROI
                contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Suponiendo que el contorno más grande es el taco
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    topmost = tuple(c[c[:,:,1].argmin()][0])
                    bottommost = tuple(c[c[:,:,1].argmax()][0])
                    
                    # Calcular la dirección de la línea
                    vector_dir = np.array(bottommost) - np.array(topmost)
                    vector_dir_normalized = vector_dir / np.linalg.norm(vector_dir)
                    
                    # Definir el tamaño fijo para la línea a dibujar
                    line_length = 200  # Tamaño fijo de la línea
                    
                    # Calcular los puntos finales de la línea extendida
                    line_start = (np.array(topmost) - vector_dir_normalized * line_length).astype(int)
                    line_end = (np.array(bottommost) + vector_dir_normalized * line_length).astype(int)

                    # Ajustar los puntos para que se dibujen en el frame completo, no solo en la ROI
                    line_start_global = (line_start[0] + x1, line_start[1] + y1)
                    line_end_global = (line_end[0] + x1, line_end[1] + y1)
                    
                    # Dibujar la línea en el frame completo
                    cv2.line(frame, tuple(line_start_global), tuple(line_end_global), (0, 255, 0), 2)

        return frame"""

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

    def verificar_interseccion_y_simular(self, frame, inicio_taco, fin_taco, detections):
        direccion_impacto = np.array(fin_taco) - np.array(inicio_taco)
        direccion_impacto_normalizada = direccion_impacto / np.linalg.norm(direccion_impacto)

        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i]
            if class_id in [0, 1]:  # Bolas
                centro_bola = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                radius = (bbox[2] - bbox[0]) / 2
                if self.intersecta_trayectoria_con_bola(inicio_taco, fin_taco, centro_bola, radius):
                    # Simula la trayectoria de la bola impactada. Necesitas calcular esta parte.
                    # Por ejemplo, asumimos que la bola se moverá en la misma dirección del impacto por simplicidad
                    punto_final_simulado = (centro_bola[0] + direccion_impacto_normalizada[0] * 100, centro_bola[1] + direccion_impacto_normalizada[1] * 100)

                    # Dibuja la trayectoria simulada
                    cv2.line(frame, (int(centro_bola[0]), int(centro_bola[1])), (int(punto_final_simulado[0]), int(punto_final_simulado[1])), (255, 0, 0), 2)

                    # Opcional: Actualiza la posición de la bola en Pymunk si es necesario
                    #self.update_or_add_ball_in_pymunk(detections, i)

    def intersecta_trayectoria_con_bola(self, inicio_taco, fin_taco, centro_bola, radius):
        # Convertir puntos a numpy arrays para facilitar el cálculo
        p1 = np.array(inicio_taco)
        p2 = np.array(fin_taco)
        centro = np.array(centro_bola)

        # Calcular la distancia del centro de la bola a la línea de trayectoria
        d = np.linalg.norm(np.cross(p2-p1, p1-centro)) / np.linalg.norm(p2-p1)

        # Verificar si la distancia es menor que el radio de la bola
        return d <= radius
    
    def simular_trayectoria_bola_impactada(self, centro_bola, radius, direccion_impacto, frame):
        # Asumiendo que tenemos un identificador único para cada bola (podría ser su posición inicial)
        bola_id = f"bola_{centro_bola}"

        # Verificar si la bola ya está en Pymunk, si no, agregarla
        if bola_id not in self.balls:
            self.balls[bola_id] = self.add_ball(centro_bola, radius)
        
        # Aplicar fuerza en la dirección del impacto
        fuerza = 1000 * np.array(direccion_impacto)  # Este valor de fuerza es arbitrario
        self.balls[bola_id].body.apply_impulse_at_local_point(fuerza)

        # Simular por un corto tiempo para calcular la nueva posición
        for _ in range(10):
            self.space.step(1/50.0)

        # Dibujar la nueva posición de la bola
        nueva_posicion = self.balls[bola_id].body.position
        cv2.circle(frame, (int(nueva_posicion.x), int(nueva_posicion.y)), int(radius), (255, 0, 0), 2)  # Dibuja en azul
    
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