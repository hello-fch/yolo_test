from ruamel.yaml import YAML
from export import export_onnx
from train import train_yolo_segmentation
from split_dataset import split_dataset_main

# 数据集配置
path= 'E:/其他/Python/测试yolo训练和使用/train/train/dataset'  # 数据集根目录：根据自己实际情况修改
data = {
    'path': path,  # 数据集根目录：根据自己实际情况修改
    'train': 'images/train',          # 训练集路径
    'val': 'images/val',              # 验证集路径
    'names': {
        0: 'tray',  # 托盘
        1: 'package'  # 包裹
    },
    'colors': [[0, 0, 255],[255, 0, 0]]  # 托盘蓝色,包裹红色
}

# 训练参数配置 (对应命令行参数)
train_args = {
    'data': data,  # 数据集配置文件路径
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

def save_data_config(output_path):
    """
    保存数据集配置到YAML文件
    :param output_path: 输出的YAML文件路径
    """
    # 创建 YAML 对象并设置格式
    yaml = YAML()
    yaml.default_flow_style = False  # 自动选择最佳格式
    yaml.representer.add_representer(
        list,
        lambda rep, data: rep.represent_sequence('tag:yaml.org,2002:seq', data,
                                                 flow_style=all(isinstance(i, list) for i in data))
    )

    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)
    print(f"数据集配置已保存到 {output_path}")

def main():
    #先划分数据集
    image_dir = "train/images"  # 替换为你的图像目录
    label_dir = "train/labels"  # 替换为你的标签目录
    output_dir = "train/dataset"  # 替换为你的输出目录
    split_dataset_main(image_dir=image_dir,label_dir=label_dir,output_dir=output_dir,train=0.8,val=0.2,test=0.0)

    # 保存数据集配置到YAML文件
    yaml_path = 'dataset.yaml'  # 替换为你想要保存的路径
    save_data_config(output_path=yaml_path)

    # 然后训练模型
    model_path = '../model/paperBin_dec_48_500.pt'  # 替换为你的模型路径
    train_args['data'] = yaml_path  # 使用刚刚保存的YAML配置文件路径
    results = train_yolo_segmentation(model_path=model_path, train_args=train_args)
    return  results

if __name__ == '__main__':
    main()
    model_path= '../model/paperBin_dec_48_500.pt'  # 替换为你的模型路径
    output_onnx_path= '../model/exported_model.onnx'  # 替换为你想要保存的ONNX模型路径
    # export_onnx(model_path,output_onnx_path)
