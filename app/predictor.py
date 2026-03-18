import cv2
import numpy as np
import onnxruntime as ort
import os
import logging
import ast

logger = logging.getLogger(__name__)

class ASLPredictor:
    def __init__(self, weights_path="models/best.onnx"):
        if not os.path.exists(weights_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(base_dir, weights_path)
            
        logger.info(f"Loading ONNX model from {weights_path}")
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(weights_path, providers=['CPUExecutionProvider'])
        
        # Extract class names from ONNX metadata (Ultralytics embeds this)
        meta = self.session.get_modelmeta().custom_metadata_map
        if 'names' in meta:
            try:
                self.class_names = ast.literal_eval(meta['names'])
            except Exception as e:
                logger.warning(f"Could not parse class names: {e}")
                self.class_names = {i: str(i) for i in range(30)}
        else:
            self.class_names = {i: str(i) for i in range(30)}

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get expected input shape (usually [1, 3, 640, 640])
        shape = self.session.get_inputs()[0].shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640
        self.input_type = self.session.get_inputs()[0].type

    def predict(self, image_content):
        """
        Predicts ASL letters from image bytes using pure ONNXRuntime and OpenCV.
        Returns: (result_dict, error_message)
        """
        try:
            # 1. Decode image
            nparr = np.frombuffer(image_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None, "Invalid image data"

            orig_h, orig_w = img.shape[:2]

            # 2. Preprocess (Letterbox resize)
            r = min(self.input_w / orig_w, self.input_h / orig_h)
            new_unpad = int(round(orig_w * r)), int(round(orig_h * r))
            dw, dh = self.input_w - new_unpad[0], self.input_h - new_unpad[1]
            dw, dh = dw / 2, dh / 2  # divide padding into 2 sides
            
            if (orig_w, orig_h) != new_unpad:
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            
            # Convert to CHW, RGB, and normalize (0-1)
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (self.input_w, self.input_h), swapRB=True, crop=False)
            
            # Cast to FP16 if the model was exported with half=True
            if 'float16' in self.input_type:
                blob = blob.astype(np.float16)
                
            # 3. Inference
            outputs = self.session.run([self.output_name], {self.input_name: blob})
            out = outputs[0][0] # Shape: [num_classes + 4, 8400]
            out = out.T # Shape: [8400, num_classes + 4]
            
            # 4. Postprocess & NMS
            scores = np.max(out[:, 4:], axis=1)
            class_ids = np.argmax(out[:, 4:], axis=1)
            
            conf_threshold = 0.25
            mask = scores > conf_threshold
            
            out = out[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            
            predictions = []
            if len(out) > 0:
                # Convert xc, yc, w, h -> x1, y1, x2, y2
                boxes = out[:, :4]
                boxes_xyxy = np.empty_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2 # x1
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2 # y1
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2 # x2
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2 # y2
                
                # Rescale boxes back to original image size
                boxes_xyxy[:, [0, 2]] -= dw
                boxes_xyxy[:, [1, 3]] -= dh
                boxes_xyxy /= r
                
                # Clip boxes to boundaries
                boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
                boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
                
                # Non-Maximum Suppression (NMS)
                indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_threshold, 0.45)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        predictions.append({
                            "label": self.class_names.get(int(class_ids[i]), str(class_ids[i])),
                            "confidence": float(scores[i]),
                            "bbox": boxes_xyxy[i].tolist()
                        })
                        
            return {
                "predictions": predictions,
                "count": len(predictions)
            }, None

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, str(e)
