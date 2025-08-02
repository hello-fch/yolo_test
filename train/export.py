from ultralytics import YOLO

def export_onnx(model_path, output_path):
    """
    导出YOLOv8分割模型为ONNX格式
    """

    # 加载训练好的模型
    model = YOLO(model_path)

    # 导出为ONNX格式
    model.export(format='onnx', name=output_path)
