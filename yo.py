import argparse

import torch

import time

import cv2
import numpy as np

import supervision as sv

import subprocess

from ultralytics import YOLO

ip_address = "192.168.137.83"
port = 8080
ip = "http://{}:{}/video".format(ip_address, port)

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
    def __init__(
        self,
        source_weights_path: str,
        ip_address: str,
        port: int,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.ip_address = ip_address
        self.port = port
        self.target_video_path=args.target_video_path
        self.confidence_threshold=args.confidence_threshold
        self.iou_threshold=args.iou_threshold
    
        self.model = YOLO(source_weights_path)
        
        self.tracker = sv.ByteTrack()
        
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        
    def process_video(self):
        video_path = f"http://{self.ip_address}:{self.port}/video"
        vid = cv2.VideoCapture(video_path)
    
        # Inicializa una variable para almacenar el tiempo del último frame procesado
        prev_frame_time = 0
        # Inicializa una variable para almacenar el tiempo del frame actual
        new_frame_time = 0
    
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            
            new_frame_time = time.time()
            processed_frame = self.process_frame(frame=frame)
            
            fps = 1/(new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        cv2.destroyAllWindows()
    
        #for frame in frame_generator:
            #processed_frame = self.process_frame(frame=frame)
            #cv2.imshow("frame", processed_frame)
            #if cv2.waitKey(1) & 0xFF == ord("q"):
                #break
        #cv2.destroyAllWindows()
    
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