from ultralytics import YOLO
import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class ASLPredictor:
    def __init__(self, weights_path="models/best.pt"):
        if not os.path.exists(weights_path):
            # Try absolute path relative to this file if models/ is not found
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(base_dir, weights_path)
            
        logger.info(f"Loading YOLO model from {weights_path}")
        self.model = YOLO(weights_path)
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}

    def predict(self, image_content):
        """
        Predicts ASL letters from image bytes.
        Returns: (result_dict, error_message)
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None, "Invalid image data"

            # Run inference
            results = self.model(img)
            
            predictions = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    label = self.class_names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    
                    predictions.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": xyxy
                    })
            
            return {
                "predictions": predictions,
                "count": len(predictions)
            }, None

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, str(e)
