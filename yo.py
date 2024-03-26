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
from sklearn.decomposition import PCA

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
        self.historial_lineas = []  # Para almacenar las últimas posiciones y ángulos de la línea
        #####PRUEBA
        ###ESTO PUEDE SER 15 O 10 Y SIRVE PARA PODER HACER MÁS SUAVE EL TACO OSEA QUE NO SEA TAN ERRÁTICO CUANDO ESTÁ ESTÁTICO
        self.max_historial = 15  # Número máximo de elementos en el historial
        self.ultimo_angulo_valido = None # En tu inicialización de clase, agrega un valor para rastrear el último ángulo válido
        
        #RADIO DE LA BOLA
        self.radio_promedio_bolas = 0
        self.radios_acumulados = []
        self.frames_para_calculo = 100  # Número de frames durante los cuales se calculará el radio promedio
        self.frame_actual = 0
        
        #RECOMENDACIONES
        self.ultimos_centros_bolas = {}  # Diccionario para rastrear la última posición conocida de cada bola
        self.umbral_movimiento = 5.0  # Define un umbral de movimiento mínimo para actualizar la posición (puedes ajustarlo según sea necesario)
        #RADIO DE LAS BOLAS
        self.contador_frames = 0  # Añade un contador para los frames
        self.radios_bolas_acumulados = []  # Para almacenar los radios de las bolas
        self.radio_bolas_promedio = None  # Para almacenar el radio promedio
    
    def dibujar_bandas_mesa(self, frame):
        # Asume que self.mesa_corners es un np.array de esquinas ordenadas
        n_corners = len(self.mesa_corners)
        for i in range(n_corners):
            # Dibuja una línea entre cada par de esquinas consecutivas
            cv2.line(frame, tuple(self.mesa_corners[i]), tuple(self.mesa_corners[(i + 1) % n_corners]), (0, 255, 0), 2)    
    
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
    
    """def suavizar_linea(self):
        if len(self.historial_lineas) == 0:
            return None, None
        centro_promedio = np.mean([centro for centro, _ in self.historial_lineas], axis=0)
        angulo_promedio = np.mean([angulo for _, angulo in self.historial_lineas])

        # Limita el cambio de ángulo para evitar saltos bruscos
        if self.ultimo_angulo_valido is not None:
            delta_angulo = min(10, abs(angulo_promedio - self.ultimo_angulo_valido))  # Limita a 10 grados de cambio
            angulo_promedio = self.ultimo_angulo_valido + np.sign(angulo_promedio - self.ultimo_angulo_valido) * delta_angulo

        self.ultimo_angulo_valido = angulo_promedio
        return centro_promedio, angulo_promedio"""
        
    def suavizar_linea(self):
        if len(self.historial_lineas) == 0:
            return None, None
        centro_promedio = np.mean([centro for centro, _ in self.historial_lineas], axis=0)
        angulo_promedio = np.mean([angulo for _, angulo in self.historial_lineas])

        # Ajuste dinámico basado en la variabilidad reciente de los ángulos
        variabilidad_angulo = np.std([angulo for _, angulo in self.historial_lineas])
        limite_cambio = max(5, min(10, 10 - variabilidad_angulo))  # Ejemplo de ajuste dinámico

        if self.ultimo_angulo_valido is not None:
            delta_angulo = min(limite_cambio, abs(angulo_promedio - self.ultimo_angulo_valido))
            angulo_promedio = self.ultimo_angulo_valido + np.sign(angulo_promedio - self.ultimo_angulo_valido) * delta_angulo

        self.ultimo_angulo_valido = angulo_promedio
        return centro_promedio, angulo_promedio

    def capture_video(self):
        video_path = f"http://{self.ip_address}:{self.port}/video"
        #video_path = "data/len2.mp4"
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
                # Dibuja el círculo en el centro de la mesa
                #frame = self.dibujar_circulo_centro_mesa(frame)
                
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

        # Aquí puedes llamar a tu función para hacer una recomendación de tiro
        # basada en las posiciones actuales de las bolas detectadas
        # Necesitarás procesar 'detections' para extraer solo las detecciones de bolas
        #centros_bolas = self.extraer_centros_bolas(detections)
        #recomendacion_tiro = self.recomendar_tiro(centros_bolas)
        #print("Detecciones:", detections)

        self.recomendar_tiro(frame, detections, self.mesa_corners)
        
        # Procesamiento de detecciones de bolas y taco
        self.handle_detections(frame, detections)

        return self.annotate_frame(frame, detections)

    #RECOMENDACIONES
    
    def calcular_radio_promedio(self, detections):
        if self.contador_frames < 30:
            for i in range(len(detections.xyxy)):
                bbox = detections.xyxy[i]
                x1, y1, x2, y2 = bbox
                radio = ((x2 - x1) + (y2 - y1)) / 4  # Calcula el radio como el promedio de ancho y alto
                self.radios_bolas_acumulados.append(radio)
            self.contador_frames += 1
        # Calcula el radio promedio solo si aún no se ha calculado y hay suficientes datos
        if self.contador_frames == 30 and self.radio_bolas_promedio is None:
            self.radio_bolas_promedio = sum(self.radios_bolas_acumulados) / len(self.radios_bolas_acumulados)
            print(f"Radio promedio calculado: {self.radio_bolas_promedio}")

    def extraer_centros_bolas(self, detections):
        centros_bolas = []
        for i in range(len(detections.xyxy)):
            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]
            tracker_id = detections.tracker_id[i]
            
            if class_id in [0, 1]:  # Suponiendo que 0 es la bola blanca y 1 son las otras bolas
                x1, y1, x2, y2 = bbox
                centro_actual = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

                # Verifica si el ID de la bola ya tiene una posición rastreada
                if tracker_id in self.ultimos_centros_bolas:
                    # Calcula la distancia entre la posición actual y la última conocida
                    distancia = np.linalg.norm(centro_actual - self.ultimos_centros_bolas[tracker_id])

                    # Si la distancia es menor que el umbral, usa la última posición conocida
                    if distancia < self.umbral_movimiento:
                        centro_actual = self.ultimos_centros_bolas[tracker_id]
                    else:
                        # Si el movimiento es significativo, actualiza la última posición conocida
                        self.ultimos_centros_bolas[tracker_id] = centro_actual
                else:
                    # Si es la primera vez que se detecta la bola, añade su centro al diccionario
                    self.ultimos_centros_bolas[tracker_id] = centro_actual

                centros_bolas.append((tracker_id, class_id, centro_actual))

        return centros_bolas
    
    """def recomendar_tiro(self, frame, detections, mesa_corners):
        if self.radio_bolas_promedio is None:
            self.calcular_radio_promedio(detections)
        
        if self.radio_bolas_promedio is None:
            print("El radio promedio de las bolas aún no ha sido calculado.")
            return frame
        
        centros_bolas = self.extraer_centros_bolas(detections)
        centro_bola_blanca = None
        # Encuentra el centro de la bola blanca
        for tracker_id, class_id, centro_bola in centros_bolas:
            if class_id == 0:
                centro_bola_blanca = centro_bola
                break
        
        if centro_bola_blanca is None:
            print("Bola blanca no encontrada.")
            return frame
        
        # Trayectorias hacia las demás bolas
        for tracker_id, class_id, centro_bola in centros_bolas:
            if class_id == 0:
                continue  # Omitir la bola blanca
            
            distancia_minima = np.inf
            buchaca_cercana = None
            # Encuentra la buchaca más cercana a la bola actual
            for esquina in mesa_corners:
                distancia = np.linalg.norm(centro_bola - esquina)
                if distancia < distancia_minima:
                    distancia_minima = distancia
                    buchaca_cercana = esquina

            # Calcula la dirección hacia la buchaca más cercana
            direccion_hacia_buchaca = (buchaca_cercana - centro_bola) / np.linalg.norm(buchaca_cercana - centro_bola)
            
            # Calcula el punto objetivo "atrás" desde el centro de la bola en dirección hacia la buchaca
            punto_objetivo = centro_bola - direccion_hacia_buchaca * (self.radio_bolas_promedio * 2)
            punto_objetivo2 = centro_bola - direccion_hacia_buchaca * (self.radio_bolas_promedio)
            
            # Dibuja un círculo en el punto_objetivo como visualización
            cv2.circle(frame, (int(punto_objetivo[0]), int(punto_objetivo[1])), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(punto_objetivo2[0]), int(punto_objetivo2[1])), 5, (255, 0, 0), -1)
            
            # Verifica si la trayectoria está despejada entre punto_objetivo y punto_objetivo2
            if self.trayectoria_despejada(frame, punto_objetivo2, punto_objetivo, [cb[2] for cb in centros_bolas if cb[0] != tracker_id]):
                # Trayectoria despejada, dibuja en verde
                cv2.line(frame, (int(punto_objetivo2[0]), int(punto_objetivo2[1])), (int(punto_objetivo[0]), int(punto_objetivo[1])), (0, 255, 0), 2)
            else:
                # Trayectoria obstruida, dibuja en rojo para el debug
                cv2.line(frame, (int(punto_objetivo2[0]), int(punto_objetivo2[1])), (int(punto_objetivo[0]), int(punto_objetivo[1])), (0, 0, 255), 2)
            ###CREO QUE ACÁ NO ES NECESARIO ESTO PERO PODRÍAMOS REVISAR LOS CASOS
            
            if self.trayectoria_despejada(frame, centro_bola_blanca, punto_objetivo, [cb[2] for cb in centros_bolas if cb[0] != tracker_id]):
                # Trayectoria despejada, dibuja en verde
                cv2.line(frame, (int(centro_bola_blanca[0]), int(centro_bola_blanca[1])), (int(punto_objetivo[0]), int(punto_objetivo[1])), (0, 255, 0), 2)
            else:
                # Trayectoria obstruida, dibuja en rojo para el debug
                cv2.line(frame, (int(centro_bola_blanca[0]), int(centro_bola_blanca[1])), (int(punto_objetivo[0]), int(punto_objetivo[1])), (0, 0, 255), 2)

            if self.trayectoria_despejada(frame, centro_bola, buchaca_cercana, [cb[2] for cb in centros_bolas if cb[0] != tracker_id]):
                # Trayectoria despejada, dibuja en verde
                cv2.line(frame, (int(centro_bola[0]), int(centro_bola[1])), (int(buchaca_cercana[0]), int(buchaca_cercana[1])), (0, 255, 0), 2)
            else:
                # Trayectoria obstruida, dibuja en rojo para el debug
                cv2.line(frame, (int(centro_bola[0]), int(centro_bola[1])), (int(buchaca_cercana[0]), int(buchaca_cercana[1])), (0, 0, 255), 2)
        return frame"""

    def recomendar_tiro(self, frame, detections, mesa_corners):
        if self.radio_bolas_promedio is None:
            self.calcular_radio_promedio(detections)
        
        if self.radio_bolas_promedio is None:
            print("El radio promedio de las bolas aún no ha sido calculado.")
            return frame
        
        centros_bolas = self.extraer_centros_bolas(detections)
        centro_bola_blanca = None
        # Encuentra el centro de la bola blanca
        for tracker_id, class_id, centro_bola in centros_bolas:
            if class_id == 0:
                centro_bola_blanca = centro_bola
                break
        
        if centro_bola_blanca is None:
            print("Bola blanca no encontrada.")
            return frame
        
        angulos_y_trayectorias = []
        # Trayectorias hacia las demás bolas
        for tracker_id, class_id, centro_bola in centros_bolas:
            if class_id == 0:
                continue  # Omitir la bola blanca
            
            # Luego, dentro de tu lógica de recomendación de tiro, reemplaza el cálculo de la buchaca más cercana
            buchacas = self.encontrar_buchacas(mesa_corners)  # Obtiene todas las ubicaciones de las buchacas
            
            distancia_minima = np.inf        
            
            buchaca_cercana = None
            # Itera sobre todas las buchacas para encontrar la más cercana
            for buchaca in buchacas:
                distancia = np.linalg.norm(centro_bola - buchaca)
                if distancia < distancia_minima:
                    distancia_minima = distancia
                    buchaca_cercana = buchaca

            # Calcula la dirección hacia la buchaca más cercana
            direccion_hacia_buchaca = (buchaca_cercana - centro_bola) / np.linalg.norm(buchaca_cercana - centro_bola)
            # Calcula los puntos objetivo
            punto_objetivo = centro_bola - direccion_hacia_buchaca * (self.radio_bolas_promedio * 2)
            punto_objetivo2 = centro_bola - direccion_hacia_buchaca * self.radio_bolas_promedio
            # Dibuja un círculo en el punto_objetivo como visualización
            cv2.circle(frame, (int(punto_objetivo[0]), int(punto_objetivo[1])), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(punto_objetivo2[0]), int(punto_objetivo2[1])), 5, (255, 0, 0), -1)
            
            
            # Verifica las trayectorias sin dibujarlas aún
            trayectoria1_despejada = self.trayectoria_despejada(frame, punto_objetivo2, punto_objetivo, [cb[2] for cb in centros_bolas if cb[0] != tracker_id])
            trayectoria2_despejada = self.trayectoria_despejada(frame, centro_bola_blanca, punto_objetivo, [cb[2] for cb in centros_bolas if cb[0] != tracker_id])
            trayectoria3_despejada = self.trayectoria_despejada(frame, centro_bola, buchaca_cercana, [cb[2] for cb in centros_bolas if cb[0] != tracker_id])
            
            # Solo si las tres trayectorias están despejadas, calcula el ángulo y decide si dibujar
            if trayectoria1_despejada and trayectoria2_despejada and trayectoria3_despejada:
                if class_id != 0:  # Asegura no calcular ángulos para la bola blanca
                    vector_trayectoria1 = punto_objetivo2 - punto_objetivo
                    vector_trayectoria2 = centro_bola_blanca - punto_objetivo
                    angulo = self.calcular_angulo(vector_trayectoria1, vector_trayectoria2)
                    angulos_y_trayectorias.append((angulo, tracker_id, punto_objetivo2, punto_objetivo, centro_bola_blanca, buchaca_cercana, centro_bola))

        # Ordena y selecciona la trayectoria con el ángulo más cercano a 180 grados
        angulos_y_trayectorias.sort(key=lambda x: abs(180 - x[0]))
        
        # Dibuja las trayectorias no seleccionadas en rojo
        for i, (_, _, punto_objetivo2, punto_objetivo, centro_bola_blanca, _, _) in enumerate(angulos_y_trayectorias[1:], start=1):  # Omitir la primera que es la seleccionada
            cv2.line(frame, (int(punto_objetivo2[0]), int(punto_objetivo2[1])), (int(punto_objetivo[0]), int(punto_objetivo[1])), (0, 0, 255), 2)
            cv2.line(frame, (int(centro_bola_blanca[0]), int(centro_bola_blanca[1])), (int(punto_objetivo[0]), int(punto_objetivo[1])), (0, 0, 255), 2)
        
        if angulos_y_trayectorias:
            for i, (angulo, tracker_id, punto_objetivo2, punto_objetivo, centro_bola_blanca, buchaca_cercana, centro_bola) in enumerate(angulos_y_trayectorias):
                color_linea = (0, 255, 0) if i == 0 else (0, 0, 255)  # Verde para la seleccionada, rojo para las demás
                # Dibuja las líneas
                cv2.line(frame, (int(punto_objetivo2[0]), int(punto_objetivo2[1])), (int(punto_objetivo[0]), int(punto_objetivo[1])), color_linea, 2)
                cv2.line(frame, (int(centro_bola_blanca[0]), int(centro_bola_blanca[1])), (int(punto_objetivo[0]), int(punto_objetivo[1])), color_linea, 2)
                cv2.line(frame, (int(centro_bola[0]), int(centro_bola[1])), (int(buchaca_cercana[0]), int(buchaca_cercana[1])), color_linea, 2)
                # Dibuja el ángulo como texto
                punto_medio = ((int(punto_objetivo[0]) + int(punto_objetivo2[0])) // 2, (int(punto_objetivo[1]) + int(punto_objetivo2[1])) // 2)
                cv2.putText(frame, f"{angulo:.2f} grados", punto_medio, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_linea, 2)
        else:
            # Si no hay trayectorias que cumplen con la condición o están todas obstruidas, puedes decidir no dibujar nada
            # o manejar de otra manera, como mostrar un mensaje de debug.
            print("No se encontraron trayectorias óptimas")

        return frame

    def encontrar_buchacas(self, mesa_corners):
        # Asegurarse de que mesa_corners sea una lista para poder usar append
        buchacas = mesa_corners.tolist() if isinstance(mesa_corners, np.ndarray) else mesa_corners.copy()

        # Calcula las buchacas centrales en los bordes largos de la mesa
        buchaca_central_superior = ((mesa_corners[0] + mesa_corners[1]) / 2).tolist()
        buchaca_central_inferior = ((mesa_corners[2] + mesa_corners[3]) / 2).tolist()

        # Añade las buchacas centrales a la lista
        buchacas.extend([buchaca_central_superior, buchaca_central_inferior])

        return np.array(buchacas)  # Opcionalmente, convierte la lista final a np.ndarray si es necesario para el resto de tu código
    
    def calcular_angulo(self, v1, v2):
        # Normalizar los vectores
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        # Calcular el producto punto entre los vectores normalizados
        producto_punto = np.dot(v1_norm, v2_norm)
        # Calcular el ángulo en radianes y luego convertirlo a grados
        angulo_radianes = np.arccos(np.clip(producto_punto, -1.0, 1.0))
        angulo_grados = np.degrees(angulo_radianes)
        return angulo_grados
    
    def trayectoria_despejada(self, frame, centro_bola, buchaca, centros_bolas):
        # Asignar un color único para el debug visual de cada trayectoria
        color_debug = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        
        for centro_otra_bola in centros_bolas:
            if np.array_equal(centro_bola, centro_otra_bola):
                continue  # Ignora la bola de inicio
                
            # Calcular la distancia al punto más cercano en la línea y obtener dicho punto
            distancia, punto_mas_cercano = self.calcular_distancia_y_punto_cercano(centro_bola, buchaca, centro_otra_bola)
            
            if distancia <= (self.radio_bolas_promedio*2):
                # Dibuja un círculo en la bola que obstruye
                cv2.circle(frame, tuple(np.int32(centro_otra_bola)), int(self.radio_bolas_promedio), color_debug, 2)
                
                # Dibuja una línea desde el centro de la bola obstructora hasta el punto de obstrucción
                cv2.line(frame, tuple(np.int32(centro_otra_bola)), tuple(np.int32(punto_mas_cercano)), color_debug, 2)
                
                return False  # Trayectoria no despejada debido a la intersección
        return True  # Trayectoria despejada

    def calcular_distancia_y_punto_cercano(self, A, B, P):
        """Calcula la distancia de un punto P a un segmento de línea AB y devuelve el punto más cercano en AB a P."""
        # Convertir a np.array para facilitar cálculos
        A = np.array(A)
        B = np.array(B)
        P = np.array(P)

        AB = B - A
        AP = P - A
        t = np.dot(AP, AB) / np.dot(AB, AB)
        t = max(0, min(1, t))  # Limita t para que el punto esté dentro del segmento AB
        
        punto_mas_cercano = A + t * AB
        distancia = np.linalg.norm(punto_mas_cercano - P)

        return distancia, punto_mas_cercano

    ##Mejorar la detección del centro de las bolas ya que esta va cambiando
    ##Mejorar los calculos de las bolas ya que estos van cambiando

    #RECOMENDACIONES

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
    
    """##PRUEBA
    def calcular_centro_mesa(self):
        if self.mesa_corners is not None and len(self.mesa_corners) > 0:
            centro_mesa = np.mean(self.mesa_corners, axis=0)
            return centro_mesa
        return None
    def dibujar_circulo_centro_mesa(self, frame):
        centro_mesa = self.calcular_centro_mesa()
        if centro_mesa is not None and self.radio_promedio_bolas > 0:
            # Convierte el centro de la mesa a una tupla de enteros para que OpenCV lo acepte
            centro_mesa = tuple(np.round(centro_mesa).astype(int))
            # Dibuja el círculo rojo en el centro de la mesa
            cv2.circle(frame, centro_mesa, int(self.radio_promedio_bolas), (0, 0, 255), 2)
        return frame
    ##"""
    
    #CALCULAR EL CHOQUE DEL TACO CON UNA BOLA
    def colision_trayectoria(self, frame, centro_suavizado, direccion_suavizada, centros_bolas, radio_bolas, grosor_taco):
        punto_colision_cercano = None
        distancia_minima = float('inf')
        # Ajusta el radio de las bolas para considerar el grosor del taco
        radio_efectivo = radio_bolas + grosor_taco / 2
        
        for centro_bola in centros_bolas:
            # Calcula el punto en la trayectoria del taco más cercano al centro de la bola
            punto_mas_cercano, distancia_al_centro = self.calcular_punto_mas_cercano_y_distancia(centro_suavizado, direccion_suavizada, centro_bola)
            
            if distancia_al_centro <= radio_efectivo:
                # Verifica si este es el punto de colisión más cercano encontrado hasta ahora
                if distancia_al_centro < distancia_minima:
                    distancia_minima = distancia_al_centro
                    # Aquí, en lugar de simplemente seleccionar el punto_mas_cercano,
                    punto_interseccion_int = tuple(np.int32(punto_mas_cercano))
                    
                    ##PRUEBA PARA VER EL PUNTO DE CHOQUE PERO DE LA LINEA PERPENDICULAR
                    #cv2.circle(frame, punto_interseccion_int, 20, (0, 255, 255), -1)  # Dibuja un círculo amarillo
        
                    # utilizamos la función encontrar_punto_interseccion para obtener una posición de colisión más precisa.
                    punto_interseccion_exacto = self.encontrar_punto_interseccion(frame, centro_suavizado, direccion_suavizada, centro_bola, radio_bolas)
                    
                    # Es posible que punto_interseccion_exacto sea None si no hay intersección real debido al discriminante negativo.
                    if punto_interseccion_exacto is not None:
                        punto_colision_cercano = punto_interseccion_exacto
                        # Actualiza la distancia mínima con la distancia al punto de intersección exacto
                        distancia_minima = np.linalg.norm(centro_suavizado - punto_interseccion_exacto)

        if punto_colision_cercano is not None:
            # Verifica si el punto de colisión está suficientemente cerca del centro de la bola
            umbral_proximidad = 5  # Ajusta este valor según sea necesario
            distancia_al_centro_bola = np.linalg.norm(np.array(centro_bola) - np.array(punto_interseccion_int))
            
            if distancia_al_centro_bola <= umbral_proximidad:
                # Calcular la dirección y velocidad de la bola post-impacto
                vector_velocidad_bola = self.calcular_vector_velocidad_bola_post_impacto(direccion_suavizada, punto_colision_cercano, centros_bolas[0])
                
                # Calcular el punto opuesto al punto de colisión, a través del centro de la bola
                # Esto crea un vector desde el punto de colisión hacia el centro de la bola
                vector_al_centro = np.array(centro_bola) - np.array(punto_colision_cercano)
                # Normalizar este vector
                vector_al_centro_normalizado = vector_al_centro / np.linalg.norm(vector_al_centro)
                # Calcular el punto inicial y final de la trayectoria proyectada
                punto_inicio_trayectoria = np.array(centro_bola) + vector_al_centro_normalizado * radio_bolas
                
                # Configuraciones iniciales para la simulación de rebotes
                velocidad_fija_bola = 1500  # Velocidad inicial arbitraria
                coeficiente_friccion = 0.9  # Factor de reducción de velocidad por rebote
                rebotes_contador = 0  # Contador de rebotes
                
                # Inicia la simulación de la trayectoria de la bola desde el punto de impacto
                punto_actual = punto_inicio_trayectoria
                direccion_actual = vector_velocidad_bola / np.linalg.norm(vector_velocidad_bola)

                # Mientras la "velocidad" de la bola no sea insignificante
                while velocidad_fija_bola > 1.0 and rebotes_contador < 4:
                    # Calcula el punto final proyectado
                    punto_final_proyectado = punto_actual + direccion_actual * velocidad_fija_bola

                    # Primero, verifica si hay colisión con otra bola en la trayectoria hacia el punto proyectado
                    """punto_colision_otra_bola, direccion_nueva = self.detectar_colision_con_bolas(punto_actual, direccion_actual, centros_bolas, radio_bolas)
                    if punto_colision_otra_bola is not None:
                        # Aquí manejas la colisión detectada con otra bola
                        # Por ejemplo, actualizando la dirección actual con la nueva dirección calculada
                        direccion_actual = direccion_nueva
                        # Puedes elegir romper el ciclo o ajustar el punto actual para la próxima iteración
                        punto_actual = punto_colision_otra_bola
                        # Si decides continuar con más lógica después de la colisión, asegúrate de ajustar la velocidad y la dirección adecuadamente
                        velocidad_fija_bola *= coeficiente_friccion
                        # Incrementar el contador de rebotes si es necesario
                        rebotes_contador += 1
                        # Para simplificar, podrías romper el ciclo después de manejar la primera colisión con otra bola
                        break"""
                    
                    # Intenta detectar una colisión con las bandas de la mesa
                    punto_colision_banda, segmento_colision = self.detectar_colision_trayectoria_con_banda(punto_actual, punto_final_proyectado, self.mesa_corners)

                    if punto_colision_banda:
                        punto_interseccion_int2 = tuple(np.int32(punto_colision_banda))
                        cv2.circle(frame, punto_interseccion_int2, 30, (0, 255, 255), -1)  # Dibuja un círculo amarillo
        
                        # Si detecta una colisión, calcula la normal de la banda
                        normal_banda = self.calcular_normal_banda(segmento_colision[0], segmento_colision[1])
                                                
                        # Refleja la dirección de la trayectoria basándose en la normal de la banda
                        direccion_actual = self.calcular_vector_reflexion(direccion_actual, normal_banda)
                        
                        # Reduce la velocidad debido al rebote
                        velocidad_fija_bola *= coeficiente_friccion
                        
                        # Dibuja la trayectoria desde el punto actual hasta el punto de colisión
                        cv2.line(frame, tuple(np.int32(punto_actual)), tuple(np.int32(punto_colision_banda)), (0, 255, 0), 2)

                        # Actualiza el punto actual para el siguiente cálculo
                        punto_actual = punto_colision_banda
                        
                        rebotes_contador += 1  # Incrementa el contador de rebotes
                    else:
                        break  # Termina el bucle
            
            return True, punto_colision_cercano
        else:
            return False, None
    
    def detectar_colision_con_bolas(self, punto_actual, direccion_actual, centros_bolas, radio_bolas):
        for centro_bola in centros_bolas:
            distancia_entre_centros = np.linalg.norm(punto_actual - centro_bola)
            if distancia_entre_centros <= 2 * radio_bolas:  # Suponiendo que todas las bolas tienen el mismo radio
                # Calcula el punto de colisión como el punto medio entre los centros en el momento del impacto
                punto_colision = (punto_actual + centro_bola) / 2
                
                # La nueva dirección se puede calcular como la diferencia entre el punto actual y el centro de la bola golpeada,
                # normalizada para que represente solo la dirección.
                direccion_nueva = centro_bola - punto_actual
                direccion_nueva_normalizada = direccion_nueva / np.linalg.norm(direccion_nueva)
                
                # Asegura devolver el punto de colisión y la nueva dirección normalizada
                return punto_colision, direccion_nueva_normalizada

        # Si no se detecta colisión con ninguna bola, devuelve None para ambos valores
        return None, None
    
    def calcular_normal_banda(self, p0, p1):
        # Calcula el vector director del segmento de banda
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        # Calcula el vector normal (perpendicular)
        normal = np.array([-dy, dx])
        # Normaliza el vector normal
        norma = np.linalg.norm(normal)
        if norma == 0:
            return None  # Evita la división por cero
        normal_normalizada = normal / norma
        return normal_normalizada
    
    def calcular_vector_reflexion(self, vector_incidencia, normal_banda):
        # Calcula el vector de reflexión
        reflejo = vector_incidencia - 2 * np.dot(vector_incidencia, normal_banda) * normal_banda
        return reflejo
     
    def detectar_colision_trayectoria_con_banda(self, punto_inicio_trayectoria, punto_final_trayectoria, mesa_corners):
        for i in range(len(mesa_corners)):
            p0 = mesa_corners[i]
            p1 = mesa_corners[(i + 1) % len(mesa_corners)]  # Asegura que el último punto se conecte con el primero

            punto_interseccion = self.calcular_interseccion(punto_inicio_trayectoria, punto_final_trayectoria, p0, p1)
            if punto_interseccion:
                return punto_interseccion, (p0, p1)  # Retorna el punto de intersección y el segmento de banda con el que intersecta

        return None, None  # No hubo colisión con bandas
    
    
    def calcular_interseccion(self, p0, p1, p2, p3):
        
        #Calcula el punto de intersección entre dos segmentos de línea definidos por los puntos p0, p1 y p2, p3.
        #Retorna el punto de intersección como (x, y) o None si no hay intersección.
        
        s1_x = p1[0] - p0[0]
        s1_y = p1[1] - p0[1]
        s2_x = p3[0] - p2[0]
        s2_y = p3[1] - p2[1]

        denom = s1_x * s2_y - s2_x * s1_y
        if denom == 0:
            return None  # Las líneas son paralelas o coincidentes

        s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / denom
        t = ( s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / denom

        if 0 <= s <= 1 and 0 <= t <= 1:  # Si s y t están entre 0 y 1, hay intersección
            # El punto de intersección está dentro de los segmentos de línea
            interseccion_x = p0[0] + (t * s1_x)
            interseccion_y = p0[1] + (t * s1_y)
            return (interseccion_x, interseccion_y)

        return None  # No hay intersección dentro de los segmentos de línea
        
    def calcular_vector_velocidad_bola_post_impacto(self, direccion_taco, punto_impacto, centro_bola):
        # Definir una velocidad fija para la bola después del impacto
        velocidad_fija_bola = 2.0  # Esta es una velocidad fija arbitraria
        
        # Normalizar la dirección del taco para usarla como la dirección de la bola
        direccion_norm = direccion_taco / np.linalg.norm(direccion_taco)
        
        # Determinar si el impacto es por arriba o por debajo de la bola
        vector_impacto = np.array(punto_impacto) - np.array(centro_bola)
        vector_impacto_norm = vector_impacto / np.linalg.norm(vector_impacto)
        
        # Determinar la orientación del impacto comparando los vectores
        # Esta condición es un ejemplo genérico y debe ser ajustada a tu lógica específica
        es_impacto_inferior = np.dot(vector_impacto_norm, direccion_norm) > 0
        
        # Invertir la dirección si el impacto es por debajo de la bola
        if es_impacto_inferior:
            vector_velocidad_bola_post_impacto = (direccion_norm * velocidad_fija_bola) * (-1)
        else:
            vector_velocidad_bola_post_impacto = direccion_norm * velocidad_fija_bola
            
        return vector_velocidad_bola_post_impacto
    
    def encontrar_punto_interseccion(self, frame, centro_suavizado, direccion_suavizada, centro_bola, radio_bola):
        p0 = np.array(centro_suavizado)  # Punto inicial de la trayectoria del taco
        d = direccion_suavizada  # Dirección normalizada de la trayectoria del taco
        c = np.array(centro_bola)  # Centro de la bola
        r = radio_bola  # Radio de la bola
        
        # Vector desde el inicio de la línea al centro de la circunferencia
        p0_c = c - p0
        
        # Coeficientes de la ecuación cuadrática at^2 + bt + c = 0
        a = np.dot(d, d)
        b = 2 * np.dot(d, p0_c)
        c = np.dot(p0_c, p0_c) - r**2
        
        discriminante = b**2 - 4*a*c
        
        if discriminante < 0:
            # No hay intersección si el discriminante es negativo
            return None
        
        # Calcular las soluciones de la ecuación cuadrática (los valores de t)
        t1 = (-b + np.sqrt(discriminante)) / (2*a)
        t2 = (-b - np.sqrt(discriminante)) / (2*a)
        
        # Elegir el t que corresponda al punto de intersección más cercano al inicio de la trayectoria
        t = min(t for t in [t1, t2] if t >= 0) if min(t1, t2) >= 0 else max(t1, t2)
        
        # Calcular el punto de intersección exacto usando el valor de t encontrado
        punto_interseccion = p0 + t * (d * -1)
        
        #print(punto_interseccion)
        #print(type(punto_interseccion))
        # Visualización del punto de intersección en el frame
        #punto_interseccion_int = tuple(np.int32(punto_interseccion))
        #cv2.circle(frame, punto_interseccion_int, 30, (0, 255, 255), -1)  # Dibuja un círculo amarillo
        
        return punto_interseccion
    
    def calcular_punto_mas_cercano_y_distancia(self, linea_inicio, direccion, punto):
        # Normaliza la dirección
        direccion_norm = direccion / np.linalg.norm(direccion)
        
        # Vector desde el inicio de la línea al punto
        inicio_a_punto = punto - linea_inicio
        
        # Proyección escalar del vector inicio_a_punto en la dirección de la línea
        t = np.dot(inicio_a_punto, direccion_norm)
        
        # Calcula el punto más cercano
        punto_mas_cercano = linea_inicio + t * direccion_norm
        
        # Calcula la distancia desde el punto más cercano al centro de la bola
        distancia = np.linalg.norm(punto - punto_mas_cercano)
        
        return punto_mas_cercano, distancia
    
    # LA CONDICIONAL ES QUE ESTE MÁS CERCA DEL CENTRO DE UNA BOLA   
    def handle_detections(self, frame, detections):
        centros_bolas = []

        # Aquí dibujas las bandas de la mesa en el frame
        #self.dibujar_bandas_mesa(frame)
        
        #CALCULAR EL RADIO DE LAS BOLAS O BOLA
        if self.frame_actual < self.frames_para_calculo:
            radios_bolas = []

            for i, bbox in enumerate(detections.xyxy):
                class_id = detections.class_id[i]
                if class_id in [0, 1]:  # Suponiendo que 0 y 1 son IDs para las bolas
                    x1, y1, x2, y2 = bbox
                    radio = ((x2 - x1) + (y2 - y1)) / 4
                    self.radios_acumulados.append(radio)
            
            self.frame_actual += 1

        # Calcula el radio promedio solo después de acumular datos de los primeros N frames
        if self.frame_actual == self.frames_para_calculo and self.radios_acumulados:
            self.radio_promedio_bolas = sum(self.radios_acumulados) / len(self.radios_acumulados)
            #print(f"Radio promedio calculado después de {self.frames_para_calculo} frames: {self.radio_promedio_bolas}")

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
                    ####ERROR########
                    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                    _, binary_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contornos, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    ####ERROR######## AQUÍ HAY UN PROBLEMA Y ES QUE SI NO HAY BUENA ILUMINACIÓNLA UMBRALIZACIÓN DE ARRIBA SE PIERDE Y YA NO SE DETECTA BIEN LOS CONTORNOS (TENER EN CUENTA)
                    ####################TAREA#######################

                    """###PRUEBA PARA VER CÓMO ESTÁ LA UMBRALIZACIÓN
                    #Convertir binary_roi a BGR para dibujar contornos en color si es necesario
                    if len(binary_roi.shape) < 3:
                        binary_roi_color = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
                    else:
                        binary_roi_color = binary_roi.copy()

                    # Dibuja todos los contornos encontrados
                    # -1 significa dibujar todos los contornos
                    # (0, 255, 0) es el color del contorno (verde en este caso)
                    # 2 es el grosor de la línea del contorno
                    cv2.drawContours(binary_roi_color, contornos, -1, (0, 255, 0), 2)

                    # Mostrar la imagen con los contornos dibujados
                    cv2.imshow('Contornos Detectados', binary_roi_color)"""
                    
                    if contornos:
                        contorno_taco = max(contornos, key=cv2.contourArea)
                        rect = cv2.minAreaRect(contorno_taco)

                        centro, (ancho, alto), angulo = rect
                        if ancho < alto:
                            angulo += 90

                        grosor_taco = min(ancho, alto)  # Consideramos el menor de los lados como el grosor
                        
                        self.historial_lineas.append(((centro[0] + x1, centro[1] + y1), angulo))
                        if len(self.historial_lineas) > self.max_historial:
                            self.historial_lineas.pop(0)

                        centro_suavizado, angulo_suavizado = self.suavizar_linea()

                        """#####PRUEBA SIRVE PARA VER CÓMO ES QUE SE ESTÁ CALCULANDO LA DIRECCIÓN DEL TACO
                        # Dibujar el rectángulo que encierra el taco
                        box = cv2.boxPoints(((centro[0] + x1, centro[1] + y1), (ancho, alto), angulo))
                        box = np.int0(box)
                        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)  # Dibuja con contorno rojo

                        # Calcular la dirección y dibujarla
                        direccion = np.array([np.cos(np.deg2rad(angulo_suavizado)), np.sin(np.deg2rad(angulo_suavizado))])
                        punto_inicio = np.array(centro_suavizado)
                        punto_final = punto_inicio + direccion * 100  # Ajusta la longitud de la línea de dirección según sea necesario

                        # Dibujar la línea de dirección
                        cv2.line(frame, tuple(np.int32(punto_inicio)), tuple(np.int32(punto_final)), (255, 255, 0), 2)  # Línea azul claro

                        # Opcionalmente, puedes dibujar un círculo en el centro_suavizado para marcar el punto de inicio
                        cv2.circle(frame, tuple(np.int32(centro_suavizado)), 5, (0, 255, 0), -1)  # Punto verde"""
                        
                        if centro_suavizado is not None and angulo_suavizado is not None:
                            direccion_suavizada = np.array([np.cos(np.deg2rad(angulo_suavizado)), np.sin(np.deg2rad(angulo_suavizado))])
                            
                            # Encuentra la primera bola que choca basado en la distancia al centro de la trayectoria del taco
                            bola_primera_colision = None
                            #distancia_minima_colision = float('inf')
                            distancia_minima_al_inicio = float('inf')  # Nuevo criterio basado en la distancia al inicio
                            for centro_bola in centros_bolas:
                                """punto_mas_cercano, distancia = self.calcular_punto_mas_cercano_y_distancia(centro_suavizado, direccion_suavizada, centro_bola)
                                if distancia < distancia_minima_colision:
                                    distancia_minima_colision = distancia
                                    bola_primera_colision = centro_bola"""
                                    
                                punto_mas_cercano, distancia_al_centro = self.calcular_punto_mas_cercano_y_distancia(centro_suavizado, direccion_suavizada, centro_bola)
                                #print(f"Centro Bola: {centro_bola}, Distancia: {distancia_al_centro}, Punto Más Cercano: {punto_mas_cercano}")
                                # Calcula la distancia desde el punto de inicio de la trayectoria hasta el punto más cercano
                                distancia_al_inicio = np.linalg.norm(punto_mas_cercano - centro_suavizado)
                                radio_efectivo = self.radio_promedio_bolas + grosor_taco / 2  # Define radio_efectivo aquí
                                if distancia_al_centro <= radio_efectivo:
                                    # Verifica si esta bola está más cerca del inicio de la trayectoria comparado con las anteriores
                                    if distancia_al_inicio < distancia_minima_al_inicio:
                                        distancia_minima_al_inicio = distancia_al_inicio
                                        bola_primera_colision = centro_bola
                            
                            if bola_primera_colision:
                                #print(f"Bola Primera Colisión: {bola_primera_colision}, Distancia Mínima: {distancia_minima_al_inicio}")
                                # Ahora consideramos el grosor del taco en la verificación de colisión
                                colision, punto_colision = self.colision_trayectoria(frame, centro_suavizado, direccion_suavizada, [bola_primera_colision], self.radio_promedio_bolas, grosor_taco)
                                if colision:
                                    cv2.circle(frame, tuple(np.int32(punto_colision)), 10, (0, 0, 255), -1)
                                    # Dibuja la trayectoria hasta el punto de colisión
                                    cv2.line(frame, tuple(np.int32(centro_suavizado)), tuple(np.int32(punto_colision)), (0, 255, 0), 2)
                                else:
                                    # Si no hay colisión directa, considera la trayectoria opuesta
                                    direccion_opuesta = -direccion_suavizada
                                    colision_opuesta, punto_colision_opuesto = self.colision_trayectoria(frame, centro_suavizado, direccion_opuesta, [bola_primera_colision], self.radio_promedio_bolas, grosor_taco)
                                    
                                    if colision_opuesta:
                                        # Dibuja la trayectoria opuesta hasta el punto de colisión
                                        cv2.line(frame, tuple(np.int32(centro_suavizado)), tuple(np.int32(punto_colision_opuesto)), (0, 255, 0), 2)                                        
                                    else:
                                        # Si ninguna trayectoria resulta en colisión, dibuja la trayectoria original completa
                                        longitud_linea = max(frame.shape) * 2
                                        punto_final_original = centro_suavizado + direccion_suavizada * longitud_linea
                                        cv2.line(frame, tuple(np.int32(centro_suavizado)), tuple(np.int32(punto_final_original)), (0, 255, 0), 2)
                            
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
                #cv2.circle(annotated_frame, center, radius, (255, 255, 255), 2)  # Blanco
                label = f"Bola blanca #{tracker_id}"
                #cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Otras bolas: dibujar un círculo azul
            elif class_id == 1:  # Otras bolas
                #cv2.circle(annotated_frame, center, radius, (255, 0, 0), 2)  # Azul
                label = f"Bola #{tracker_id}"
                #cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
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