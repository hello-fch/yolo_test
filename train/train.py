from ultralytics import YOLO
import os
import argparse

def train_yolo_segmentation(model_path,train_args=None):
    """
    训练YOLOv8分割模型
    :param model_path: 预训练权重路径
    :param train_args: 训练参数配置字典 (可选)
    :return:
    """
    # 创建YOLO模型实例 (加载预训练权重)
    model = YOLO(model_path)  # 使用分割模型
    if train_args is None:
        print("未提供训练参数，使用默认配置。")
        # 训练参数配置 (对应命令行参数)
        train_args = {
            'data': 'tray_seg.yaml',  # 数据集配置文件路径
            'epochs': 1000,  # 训练轮数
            'imgsz': 640,  # 输入图像尺寸
            'batch': 16,  # 批次大小
            # 'optimizer': 'AdamW',  # 优化器
            'lr0': 0.001,  # 初始学习率
            'device': '0',  # 使用GPU (如果是CPU设为'cpu')
            'workers': 4,  # 数据加载线程数
            'save_period': 20,  # 每20轮保存一次模型

            'verbose': True,  # 显示详细输出
            'patience': 50,  # 早停等待轮数
            'seed': 42,  # 随机种子
            'close_mosaic': 10,  # 最后10轮关闭mosaic增强
            'copy_paste': 0.5,  # 复制粘贴增强概率
            'hsv_h': 0.015,  # 色调增强系数
            'hsv_s': 0.7,  # 饱和度增强系数
            'hsv_v': 0.4,  # 明度增强系数
            'degrees': 15,  # 旋转角度范围
            'flipud': 0.5,  # 上下翻转概率
            'fliplr': 0.5,  # 左右翻转概率
        }

    # 启动训练
    results = model.train(**train_args)

    return results

if __name__ == '__main__':
    print("开始训练YOLOv8分割模型...")
    # results = train_yolo_segmentation('model/paperBin_dec_48_500.pt')  # 替换为你的模型路径