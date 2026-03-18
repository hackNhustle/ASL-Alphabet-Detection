import os
import time
import numpy as np
import cv2
from app.predictor import ASLPredictor

def main():
    print("Loading ONNX model...")
    start_load = time.time()
    try:
        predictor = ASLPredictor("models/best.onnx")
    except Exception as e:
        print("Failed to load:", e)
        return
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.")

    # Create dummy image to test memory overhead properly
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', dummy_img)
    image_bytes = buffer.tobytes()

    print("Running warmup inference...")
    res, err = predictor.predict(image_bytes)
    if err:
        print("Error during warmup:", err)

    print("Running 10 inferences to measure speed...")
    start_infer = time.time()
    for _ in range(10):
        _, _ = predictor.predict(image_bytes)
    infer_time = (time.time() - start_infer) / 10
    print(f"Average ONNX inference time: {infer_time*1000:.2f} ms")

if __name__ == "__main__":
    main()
