from ultralytics import YOLO

def main():
    print("Loading YOLO model...")
    model = YOLO("models/best.pt")
    
    # Export to ONNX format (half=True reduces precision to FP16 for less RAM)
    print("Exporting to ONNX...")
    model.export(format="onnx", half=True, simplify=True)
    print("Export complete.")

if __name__ == "__main__":
    main()
