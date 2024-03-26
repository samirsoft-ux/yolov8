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
        detections = self.tracker.update_with_detections(detections)
        return self.annotate_frame(frame=frame, detections=detections)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
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
    
    
    def colision_trayectoria(self, centro_suavizado, direccion_suavizada, centros_bolas, radio_bolas, grosor_taco):
        punto_colision_cercano = None
        distancia_minima = float('inf')
        
        radio_efectivo = radio_bolas + grosor_taco / 2
        
        for centro_bola in centros_bolas:
            punto_mas_cercano, distancia_al_centro = self.calcular_punto_mas_cercano_y_distancia(centro_suavizado, direccion_suavizada, centro_bola)
            if distancia_al_centro <= radio_efectivo:
                if distancia_al_centro < distancia_minima:
                    distancia_minima = distancia_al_centro
                    punto_colision_cercano = punto_mas_cercano

        if punto_colision_cercano is not None:
            return True, punto_colision_cercano
        else:
            return False, None
    
    def calcular_punto_mas_cercano_y_distancia(self, linea_inicio, direccion, punto):
        direccion_norm = direccion / np.linalg.norm(direccion)
        inicio_a_punto = punto - linea_inicio
        t = np.dot(inicio_a_punto, direccion_norm)
        punto_mas_cercano = linea_inicio + t * direccion_norm
        distancia = np.linalg.norm(punto - punto_mas_cercano)
        return punto_mas_cercano, distancia
    
    def handle_detections(self, frame, detections):
        centros_bolas = []
        if self.frame_actual < self.frames_para_calculo:
            radios_bolas = []
            for i, bbox in enumerate(detections.xyxy):
                class_id = detections.class_id[i]
                if class_id in [0, 1]:  # Suponiendo que 0 y 1 son IDs para las bolas
                    x1, y1, x2, y2 = bbox
                    radio = ((x2 - x1) + (y2 - y1)) / 4
                    self.radios_acumulados.append(radio)
            self.frame_actual += 1
        if self.frame_actual == self.frames_para_calculo and self.radios_acumulados:
            self.radio_promedio_bolas = sum(self.radios_acumulados) / len(self.radios_acumulados)
            
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
                        grosor_taco = min(ancho, alto)  # Consideramos el menor de los lados como el grosor
                        self.historial_lineas.append(((centro[0] + x1, centro[1] + y1), angulo))
                        if len(self.historial_lineas) > self.max_historial:
                            self.historial_lineas.pop(0)
                        centro_suavizado, angulo_suavizado = self.suavizar_linea()
                        if centro_suavizado is not None and angulo_suavizado is not None:
                            direccion_suavizada = np.array([np.cos(np.deg2rad(angulo_suavizado)), np.sin(np.deg2rad(angulo_suavizado))])
                            bola_primera_colision = None
                            distancia_minima_colision = float('inf')
                            for centro_bola in centros_bolas:
                                punto_mas_cercano, distancia = self.calcular_punto_mas_cercano_y_distancia(centro_suavizado, direccion_suavizada, centro_bola)
                                if distancia < distancia_minima_colision:
                                    distancia_minima_colision = distancia
                                    bola_primera_colision = centro_bola
                            
                            if bola_primera_colision:
                                colision, punto_colision = self.colision_trayectoria(centro_suavizado, direccion_suavizada, [bola_primera_colision], self.radio_promedio_bolas, grosor_taco)
                                if colision:
                                    cv2.line(frame, tuple(np.int32(centro_suavizado)), tuple(np.int32(punto_colision)), (0, 255, 0), 2)
                                else:
                                    direccion_opuesta = -direccion_suavizada
                                    colision_opuesta, punto_colision_opuesto = self.colision_trayectoria(centro_suavizado, direccion_opuesta, [bola_primera_colision], self.radio_promedio_bolas, grosor_taco)
                                    if colision_opuesta:
                                        cv2.line(frame, tuple(np.int32(centro_suavizado)), tuple(np.int32(punto_colision_opuesto)), (0, 255, 0), 2)                                        
                                    else:
                                        longitud_linea = max(frame.shape) * 2
                                        punto_final_original = centro_suavizado + direccion_suavizada * longitud_linea
                                        cv2.line(frame, tuple(np.int32(centro_suavizado)), tuple(np.int32(punto_final_original)), (0, 255, 0), 2)
                            
                else:
                    print("ROI vacío o de tamaño inválido.")
        return frame
    
    
    
ya el problema está aca y es que este cálculo contempla varios puntos representados por circulos amarillos y cuando hay varias bolas en la trayectoria del taco y por ende varios circulos amarillos están chocando con varias bolas a veces toma la más cercana a veces la más lejana a veces la del medio, en conclusión creo que en este punto es donde se necesita diferenciar y seleccionar sola la bola más cercana al punto de incio de la trayectoria del taco
ChatGPT
Para asegurarte de que siempre se seleccione la bola más cercana al punto de inicio de la trayectoria del taco, necesitas un enfoque que te permita no solo encontrar el punto más cercano a lo largo de la trayectoria para cada bola, sino también evaluar qué bola se encuentra efectivamente más cerca en términos de la distancia a lo largo de esa trayectoria desde el punto de inicio del taco. Esto implica dos pasos principales:

Calcular el punto más cercano en la trayectoria para cada bola, como ya estás haciendo.
Determinar cuál de estos puntos está efectivamente más cerca del punto de inicio de la trayectoria del taco en términos de distancia a lo largo de esa trayectoria.
Mejora en el Proceso de Selección
Para implementar esto, puedes modificar el proceso de selección en tu método handle_detections para incluir no solo la distancia del centro de la bola al punto más cercano en la trayectoria, sino también la distancia a lo largo de la trayectoria desde el punto de inicio del taco hasta ese punto más cercano. Esto te permitirá identificar y seleccionar consistentemente la bola que está efectivamente más cerca del inicio de la trayectoria del taco.

Aquí te muestro cómo podrías ajustar tu código para lograr esto:

python
Copy code
def handle_detections(self, frame, detections):
    # Tus cálculos previos para establecer centro_suavizado, angulo_suavizado, etc.
    ...
    bola_primera_colision = None
    distancia_minima_al_inicio = float('inf')  # Nuevo criterio basado en la distancia al inicio

    for centro_bola in centros_bolas:
        punto_mas_cercano, distancia_al_centro = self.calcular_punto_mas_cercano_y_distancia(centro_suavizado, direccion_suavizada, centro_bola)
        # Calcula la distancia desde el punto de inicio de la trayectoria hasta el punto más cercano
        distancia_al_inicio = np.linalg.norm(punto_mas_cercano - centro_suavizado)
        
        if distancia_al_centro <= radio_efectivo:
            # Verifica si esta bola está más cerca del inicio de la trayectoria comparado con las anteriores
            if distancia_al_inicio < distancia_minima_al_inicio:
                distancia_minima_al_inicio = distancia_al_inicio
                bola_primera_colision = centro_bola

    # El resto de tu lógica para manejar la colisión
    ...
Explicación del Cambio
Distancia al Inicio de la Trayectoria: Al introducir distancia_al_inicio como criterio, te estás asegurando de que la selección de la bola para la colisión se base en qué bola se encuentra más cerca del punto de inicio del movimiento del taco, no solo en términos de proximidad perpendicular a la trayectoria, sino realmente en términos de quién sería golpeada primero según la dirección del taco.
Este enfoque te ayudará a solucionar el problema de inconsistencias al seleccionar la bola con la que colisiona cuando hay múltiples bolas cerca de la trayectoria del taco. Al centrarte en la distancia a lo largo de la trayectoria desde el punto de inicio, deberías poder identificar de manera más confiable y coherente cuál es la primera bola que el taco golpearía.























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
        # Calcular el punto de inicio de la trayectoria en el lado opuesto
        punto_inicio_trayectoria = np.array(centro_bola) + vector_al_centro_normalizado * radio_bolas
        
        # Dibujar la trayectoria de la bola post-impacto desde el punto opuesto
        longitud_trayectoria = 1000  # Define qué tan largo quieres que sea el trazo de la trayectoria
        punto_final_trayectoria = punto_inicio_trayectoria + vector_velocidad_bola * longitud_trayectoria
        cv2.line(frame, tuple(np.int32(punto_inicio_trayectoria)), tuple(np.int32(punto_final_trayectoria)), (0, 255, 0), 2)
    
    return True, punto_colision_cercano
else:
    return False, None


for segmento in segmentos_banda:
    cv2.line(frame, tuple(np.int32(segmento[0])), tuple(np.int32(segmento[1])), (255, 0, 0), 20)  # Bandas en rojo
    punto_interseccion = self.calcular_interseccion(punto_inicio_trayectoria, punto_final_trayectoria, segmento[0], segmento[1])
    #print(segmento[0])
    #print(segmento[1])
    if punto_interseccion:
        # Si hay intersección, ajustar punto_final_trayectoria al punto de intersección
        #punto_final_trayectoria = punto_interseccion
        
        # Visualización del punto de intersección en el frame
        punto_interseccion_int = tuple(np.int32(punto_interseccion))
        cv2.circle(frame, punto_interseccion_int, 30, (0, 255, 255), -1)  # Dibuja un círculo amarillo

        break  # Salir del bucle después de encontrar la primera intersección

cv2.line(frame, tuple(np.int32(punto_inicio_trayectoria)), tuple(np.int32(punto_final_trayectoria)), (0, 255, 0), 2)
                
                
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
                # Calcular el punto de inicio de la trayectoria en el lado opuesto
                punto_inicio_trayectoria = np.array(centro_bola) + vector_al_centro_normalizado * radio_bolas
                
                # Dibujar la trayectoria de la bola post-impacto desde el punto opuesto
                longitud_trayectoria = 1000  # Define qué tan largo quieres que sea el trazo de la trayectoria
                # Suponiendo que tienes una lista de segmentos de banda
                segmentos_banda = self.generar_segmentos_banda()
                for segmento in segmentos_banda:
                    print("Segmento de banda:", segmento[0], "a", segmento[1])
                punto_final_trayectoria = punto_inicio_trayectoria + vector_velocidad_bola * longitud_trayectoria
                
                # Verificar la intersección con cada segmento de banda
                for segmento in segmentos_banda:
                    cv2.line(frame, tuple(np.int32(segmento[0])), tuple(np.int32(segmento[1])), (255, 0, 0), 20)  # Bandas en rojo
                    cv2.line(frame, tuple(np.int32(punto_inicio_trayectoria)), tuple(np.int32(punto_final_trayectoria)), (0, 0, 255), 2)  # Trayectoria en azul
                    punto_interseccion = self.calcular_interseccion(punto_inicio_trayectoria, punto_final_trayectoria, segmento[0], segmento[1])
                    #print(segmento[0])
                    #print(segmento[1])
                    if punto_interseccion:
                        # Si hay intersección, ajustar punto_final_trayectoria al punto de intersección
                        #punto_final_trayectoria = punto_interseccion
                        
                        # Visualización del punto de intersección en el frame
                        punto_interseccion_int = tuple(np.int32(punto_interseccion))
                        cv2.circle(frame, punto_interseccion_int, 30, (0, 255, 255), -1)  # Dibuja un círculo amarillo
        
                        break  # Salir del bucle después de encontrar la primera intersección
                
                cv2.line(frame, tuple(np.int32(punto_inicio_trayectoria)), tuple(np.int32(punto_final_trayectoria)), (0, 255, 0), 2)
            
            return True, punto_colision_cercano
        else:
            return False, None
        
        
        
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
        segmentos_banda = self.generar_segmentos_banda()
        intersecciones_potenciales = []
        if distancia_al_centro_bola <= umbral_proximidad:
            # Calcular la dirección y velocidad de la bola post-impacto
            vector_velocidad_bola = self.calcular_vector_velocidad_bola_post_impacto(direccion_suavizada, punto_colision_cercano, centros_bolas[0])
            
            # Calcular el punto opuesto al punto de colisión, a través del centro de la bola
            # Esto crea un vector desde el punto de colisión hacia el centro de la bola
            vector_al_centro = np.array(centro_bola) - np.array(punto_colision_cercano)
            # Normalizar este vector
            vector_al_centro_normalizado = vector_al_centro / np.linalg.norm(vector_al_centro)
            # Calcular el punto de inicio de la trayectoria en el lado opuesto
            punto_inicio_trayectoria = np.array(centro_bola) + vector_al_centro_normalizado * radio_bolas
            
            # Dibujar la trayectoria de la bola post-impacto desde el punto opuesto
            longitud_trayectoria = 1000  # Define qué tan largo quieres que sea el trazo de la trayectoria
            # Suponiendo que tienes una lista de segmentos de banda
            
            for segmento in segmentos_banda:
                print("Segmento de banda:", segmento[0], "a", segmento[1])
            punto_final_trayectoria = punto_inicio_trayectoria + vector_velocidad_bola * longitud_trayectoria
            
            for segmento in segmentos_banda:
                cv2.line(frame, tuple(np.int32(segmento[0])), tuple(np.int32(segmento[1])), (255, 0, 0), 20)  # Bandas en rojo
                #cv2.line(frame, tuple(np.int32(punto_inicio_trayectoria)), tuple(np.int32(punto_final_trayectoria)), (0, 0, 255), 2)  # Trayectoria en azul
                punto_interseccion = self.calcular_interseccion(punto_inicio_trayectoria, punto_final_trayectoria, segmento[0], segmento[1])
                if punto_interseccion:
                    cv2.circle(frame, tuple(np.int32(punto_interseccion)), 10, (0, 255, 255), -1)
                    distancia = np.linalg.norm(punto_inicio_trayectoria - np.array(punto_interseccion))
                    intersecciones_potenciales.append((punto_interseccion, distancia))

            # Evalúa las intersecciones potenciales para encontrar la mejor candidata
            if intersecciones_potenciales:
                # Selecciona la intersección más cercana (o cualquier otro criterio que prefieras)
                interseccion_seleccionada = min(intersecciones_potenciales, key=lambda x: x[1])
                punto_final_trayectoria = interseccion_seleccionada[0]

            # Dibuja la trayectoria ajustada de la bola hasta el punto seleccionado
            cv2.line(frame, tuple(np.int32(punto_inicio_trayectoria)), tuple(np.int32(punto_final_trayectoria)), (0, 255, 0), 2)
        
        return True, punto_colision_cercano
    else:
        return False, None
    
    #### CALCULOS DE LA BOLA QUE HA CHOCADO EL TACO CON ALGUNA BANDA 
    def generar_segmentos_banda(self):
        # Asumiendo que mesa_corners ya está ordenada y contiene 4 esquinas
        segmentos = []
        for i in range(len(self.mesa_corners)):
            punto_inicio = self.mesa_corners[i]
            # El operador % len(self.mesa_corners) asegura que el índice vuelve a 0 después de alcanzar el último punto
            punto_final = self.mesa_corners[(i + 1) % len(self.mesa_corners)]
            segmentos.append((punto_inicio, punto_final))
        return segmentos
    
    def calcular_interseccion(self, p1, p2, p3, p4):
        # Convertir puntos a 'numpy.array' para facilitar los cálculos
        p1 = np.array(p1); p2 = np.array(p2)
        p3 = np.array(p3); p4 = np.array(p4)
        
        # Calcular partes del sistema de ecuaciones para encontrar el punto de intersección
        s = np.vstack([p1, p2, p3, p4])  # Pila vertical de puntos
        h = np.hstack((s, np.ones((4, 1))))  # Añadir una columna de unos
        l1 = np.cross(h[0], h[1])  # Encontrar los vectores perpendiculares a los puntos
        l2 = np.cross(h[2], h[3])  
        x, y, z = np.cross(l1, l2)  # El punto de intersección es perpendicular a ambos vectores perpendiculares
        
        if z == 0:  # Si z es 0, las líneas son paralelas o coincidentes
            return None
        return (x/z, y/z)
    
    
    
def colision_trayectoria(self, frame, centro_suavizado, direccion_suavizada, centros_bolas, radio_bolas, grosor_taco):
    punto_colision_cercano = None
    distancia_minima = float('inf')
    # Ajusta el radio de las bolas para considerar el grosor del taco
    radio_efectivo = radio_bolas + grosor_taco / 2
    
    for centro_bola in centros_bolas:
        # Calcula el punto en la trayectoria del taco más cercano al centro de la bola
        punto_mas_cercano, distancia_al_centro = self.calcular_punto_mas_cercano_y_distancia(centro_suavizado, direccion_suavizada, centro_bola)
        
        if distancia_al_centro <= radio_efectivo:
            if distancia_al_centro < distancia_minima:
                distancia_minima = distancia_al_centro
                punto_interseccion_exacto = self.encontrar_punto_interseccion(frame, centro_suavizado, direccion_suavizada, centro_bola, radio_bolas)
                if punto_interseccion_exacto is not None:
                    punto_colision_cercano = punto_interseccion_exacto
                    distancia_minima = np.linalg.norm(centro_suavizado - punto_interseccion_exacto)

    if punto_colision_cercano is not None:
        punto_inicio_trayectoria = np.array(centro_bola) + (np.array(punto_colision_cercano) - np.array(centro_bola)) * 2  # Asumiendo centro_bola es el centro de la última bola colisionada
        vector_velocidad_bola = self.calcular_vector_velocidad_bola_post_impacto(direccion_suavizada, punto_colision_cercano, centro_bola)
        longitud_trayectoria = 1000
        punto_final_trayectoria_proyectado = punto_inicio_trayectoria + vector_velocidad_bola * longitud_trayectoria
        
        # Ciclo para manejar múltiples rebotes
        while True:
            punto_colision_banda, segmento_colision = self.detectar_colision_trayectoria_con_banda(punto_inicio_trayectoria, punto_final_trayectoria_proyectado, self.mesa_corners)
            if punto_colision_banda:
                normal_banda = self.calcular_normal_banda(segmento_colision)
                vector_reflexion = self.calcular_vector_reflexion(vector_velocidad_bola, normal_banda)
                punto_inicio_trayectoria = punto_colision_banda  # Actualiza el punto de inicio para el próximo rebote
                vector_velocidad_bola = vector_reflexion  # Actualiza la dirección de la bola post-rebote
                punto_final_trayectoria_proyectado = punto_inicio_trayectoria + vector_reflexion * longitud_trayectoria
                # Dibuja el rebote
                cv2.circle(frame, tuple(np.int32(punto_colision_banda)), 10, (255, 0, 0), -1)  # Punto de colisión con la banda
            else:
                # No más colisiones, dibuja la trayectoria final
                cv2.line(frame, tuple(np.int32(punto_inicio_trayectoria)), tuple(np.int32(punto_final_trayectoria_proyectado)), (0, 255, 0), 2)
                break

    return True, punto_colision_cercano if punto_colision_cercano else None



    if punto_colision_cercano is not None and distancia_al_centro_bola <= umbral_proximidad:
        velocidad_fija_bola = 1500
        coeficiente_friccion = 0.9
        trayectoria_actual = {'puntos': [(punto_inicio_trayectoria,)], 'rebotes': 0}
        
        punto_actual = punto_inicio_trayectoria
        direccion_actual = vector_velocidad_bola / np.linalg.norm(vector_velocidad_bola)

        while velocidad_fija_bola > 1.0:
            punto_final_proyectado = punto_actual + direccion_actual * velocidad_fija_bola
            punto_colision_banda, segmento_colision = self.detectar_colision_trayectoria_con_banda(punto_actual, punto_final_proyectado, self.mesa_corners)

            if punto_colision_banda:
                normal_banda = self.calcular_normal_banda(segmento_colision[0], segmento_colision[1])
                direccion_actual = self.calcular_vector_reflexion(direccion_actual, normal_banda)
                velocidad_fija_bola *= coeficiente_friccion
                
                # Aquí actualizamos la trayectoria actual con el nuevo punto de colisión y aumentamos el contador de rebotes
                trayectoria_actual['puntos'].append(punto_colision_banda)
                trayectoria_actual['rebotes'] += 1
                
                punto_actual = punto_colision_banda
            else:
                break
        
        # Guardamos la trayectoria actual
        todas_las_trayectorias.append(trayectoria_actual)
    
    # Seleccionamos la trayectoria con más rebotes
    trayectoria_con_mas_rebotes = max(todas_las_trayectorias, key=lambda x: x['rebotes'], default=None)
    
    # Dibujamos la trayectoria seleccionada
    if trayectoria_con_mas_rebotes:
        for i in range(len(trayectoria_con_mas_rebotes['puntos']) - 1):
            cv2.line(frame, tuple(np.int32(trayectoria_con_mas_rebotes['puntos'][i])), tuple(np.int32(trayectoria_con_mas_rebotes['puntos'][i + 1])), (0, 255, 0), 2)
            
            
            
































if punto_colision_cercano is not None:
    # Verifica si el punto de colisión está suficientemente cerca del centro de la bola
    umbral_proximidad = 5  # Ajusta este valor según sea necesario
    distancia_al_centro_bola = np.linalg.norm(np.array(centro_bola) - np.array(punto_interseccion_int))
    
    if distancia_al_centro_bola <= umbral_proximidad:
        # Calcular la dirección y velocidad de la bola post-impacto
        vector_velocidad_bola = self.calcular_vector_velocidad_bola_post_impacto(direccion_suavizada, punto_colision_cercano, centros_bolas[0])
        
        vector_al_centro = np.array(centro_bola) - np.array(punto_colision_cercano)
        # Normalizar este vector
        vector_al_centro_normalizado = vector_al_centro / np.linalg.norm(vector_al_centro)
        # Calcular el punto inicial y final de la trayectoria proyectada
        punto_inicio_trayectoria = np.array(centro_bola) + vector_al_centro_normalizado * radio_bolas
        
        # Configuraciones iniciales para la simulación de rebotes
        velocidad_fija_bola = 1500  # Velocidad inicial arbitraria
        coeficiente_friccion = 0.9  # Factor de reducción de velocidad por rebote
        
        trayectoria_actual = {'puntos': [(punto_inicio_trayectoria,)], 'rebotes': 0}

        
        # Inicia la simulación de la trayectoria de la bola desde el punto de impacto
        punto_actual = punto_inicio_trayectoria
        direccion_actual = vector_velocidad_bola / np.linalg.norm(vector_velocidad_bola)

        while velocidad_fija_bola > 1.0:
            # Calcula el punto final proyectado
            punto_final_proyectado = punto_actual + direccion_actual * velocidad_fija_bola

            # Intenta detectar una colisión con las bandas de la mesa
            punto_colision_banda, segmento_colision = self.detectar_colision_trayectoria_con_banda(punto_actual, punto_final_proyectado, self.mesa_corners)

            if punto_colision_banda:
                # Si detecta una colisión, calcula la normal de la banda
                normal_banda = self.calcular_normal_banda(segmento_colision[0], segmento_colision[1])
                                        
                # Refleja la dirección de la trayectoria basándose en la normal de la banda
                direccion_actual = self.calcular_vector_reflexion(direccion_actual, normal_banda)
                
                # Reduce la velocidad debido al rebote
                velocidad_fija_bola *= coeficiente_friccion
                
                # Aquí actualizamos la trayectoria actual con el nuevo punto de colisión y aumentamos el contador de rebotes
                trayectoria_actual['puntos'].append(punto_colision_banda)
                trayectoria_actual['rebotes'] += 1

                # Actualiza el punto actual para el siguiente cálculo
                punto_actual = punto_colision_banda
                
            else:
                break  # Termina el bucle
            
        # Guardamos la trayectoria actual
        todas_las_trayectorias.append(trayectoria_actual)
        
    # Seleccionamos la trayectoria con más rebotes
    trayectoria_con_mas_rebotes = max(todas_las_trayectorias, key=lambda x: x['rebotes'], default=None)
    
    # Dibujamos la trayectoria seleccionada
    if trayectoria_con_mas_rebotes:
        for i in range(len(trayectoria_con_mas_rebotes['puntos']) - 1):
            cv2.line(frame, tuple(np.int32(trayectoria_con_mas_rebotes['puntos'][i])), tuple(np.int32(trayectoria_con_mas_rebotes['puntos'][i + 1])), (0, 255, 0), 2)
    
    return True, punto_colision_cercano
else:
    return False, None


imprimiednio todas las trayectorias
[{'puntos': [[604.6160955863497, 503.24922882462613], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
             [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195]], 'rebotes': 70}]
imprimiednio la trayectoria mas grande
{'puntos': [[604.6160955863497, 503.24922882462613], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], 
            [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195], [395.3497546611093, 257.73567616356195]], 'rebotes': 70}
imprimiednio todas las trayectorias
[{'puntos': [[605.2338417337363, 503.3160568345733], [395.7040693381995, 257.7631497615182], [234.5888976078241, 396.1399125821547], [746.3510931665044, 961.1631600623798], [1444.3319795424482, 339.07387007147327], [1719.3827300722146, 673.6564058851276], [1375.1058500349477, 994.7864090927781], [1375.1058500349477, 994.786409092778], [1375.1058500349477, 994.786409092778]], 'rebotes': 8}]
imprimiednio la trayectoria mas grande
{'puntos': [[605.2338417337363, 503.3160568345733], [395.7040693381995, 257.7631497615182], [234.5888976078241, 396.1399125821547], [746.3510931665044, 961.1631600623798], [1444.3319795424482, 339.07387007147327], [1719.3827300722146, 673.6564058851276], [1375.1058500349477, 994.7864090927781], [1375.1058500349477, 994.786409092778], [1375.1058500349477, 994.786409092778]], 'rebotes': 8}