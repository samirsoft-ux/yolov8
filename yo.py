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
    def colision_trayectoria(self, centro_suavizado, direccion_suavizada, centros_bolas, radio_bolas, grosor_taco):
        punto_colision_cercano = None
        distancia_minima = float('inf')
        
        # Ajusta el radio de las bolas para considerar el grosor del taco
        radio_efectivo = radio_bolas + grosor_taco / 2
        
        for centro_bola in centros_bolas:
            # Calcula el punto en la trayectoria del taco más cercano al centro de la bola
            punto_mas_cercano, distancia_al_centro = self.calcular_punto_mas_cercano_y_distancia(centro_suavizado, direccion_suavizada, centro_bola)
            
            # Si el punto más cercano está dentro del radio efectivo de la bola, hay una colisión
            if distancia_al_centro <= radio_efectivo:
                # Verifica si este es el punto de colisión más cercano encontrado hasta ahora
                if distancia_al_centro < distancia_minima:
                    distancia_minima = distancia_al_centro
                    punto_colision_cercano = punto_mas_cercano

        # Si encontramos un punto de colisión, retorna verdadero y el punto
        if punto_colision_cercano is not None:
            return True, punto_colision_cercano
        else:
            return False, None
    
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
                            distancia_minima_colision = float('inf')
                            for centro_bola in centros_bolas:
                                punto_mas_cercano, distancia = self.calcular_punto_mas_cercano_y_distancia(centro_suavizado, direccion_suavizada, centro_bola)
                                if distancia < distancia_minima_colision:
                                    distancia_minima_colision = distancia
                                    bola_primera_colision = centro_bola
                            
                            if bola_primera_colision:
                                # Ahora consideramos el grosor del taco en la verificación de colisión
                                colision, punto_colision = self.colision_trayectoria(centro_suavizado, direccion_suavizada, [bola_primera_colision], self.radio_promedio_bolas, grosor_taco)
                                if colision:
                                    # Dibuja la trayectoria hasta el punto de colisión
                                    cv2.line(frame, tuple(np.int32(centro_suavizado)), tuple(np.int32(punto_colision)), (0, 255, 0), 2)
                                else:
                                    # Si no hay colisión directa, considera la trayectoria opuesta
                                    direccion_opuesta = -direccion_suavizada
                                    colision_opuesta, punto_colision_opuesto = self.colision_trayectoria(centro_suavizado, direccion_opuesta, [bola_primera_colision], self.radio_promedio_bolas, grosor_taco)
                                    
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

    processor.start()