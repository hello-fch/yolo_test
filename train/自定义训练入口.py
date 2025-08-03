from pathlib import Path

from ruamel.yaml import YAML
from export import export_onnx
from train import train_yolo_segmentation
from split_dataset import split_dataset_main

# 数据集配置
path= f'{Path(__file__).resolve().parent}/train/dataset'  # 数据集根目录：根据自己实际情况修改
data = {
    'path': path,  # 数据集根目录：根据自己实际情况修改
    'train': 'images/train',          # 训练集路径
    'val': 'images/val',              # 验证集路径
    'names': {},
    'colors': [[0, 0, 255],[255, 0, 0]]  # 托盘蓝色,包裹红色
}

# 训练参数配置 (对应命令行参数)
train_args = {
    'data': '',  # 数据集配置文件路径
    'epochs': 1000,  # 训练轮数
    'imgsz': 640,  # 输入图像尺寸
    'batch': 16,  # 批次大小
    # 'optimizer': 'AdamW',  # 优化器
    'lr0': 0.001,  # 初始学习率
    'device': 'cpu',  # 使用GPU (如果是CPU设为'cpu')
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

def save_data_config(input_path,output_path):
    """
    保存数据集配置到YAML文件
    :param output_path: 输出的YAML文件路径
    """
    # 创建 YAML 对象并设置格式
    yaml = YAML()
    yaml.default_flow_style = False  # 自动选择最佳格式

    #检查输入路径是否存在
    if Path(input_path).exists():
        # 读取输入的YAML配置文件中的names,然后将自定义yaml的names替换为这个
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = yaml.load(f)
            # 将 names 列表转为字典格式
            names_list = input_data.get('names', [])
            if isinstance(names_list, list):
                data['names'] = {i: name for i, name in enumerate(names_list)}

    #判断 list 中的元素是否都是 list，如果是，则使用 flow style 格式输出（例如 [ [1, 2], [3, 4] ]）；否则使用 block style（换行缩进格式）输出
    yaml.representer.add_representer(
        list,
        lambda rep, data: rep.represent_sequence(
            'tag:yaml.org,2002:seq',
            data,
            flow_style=all(isinstance(i, list) for i in data))
    )

    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)
    print(f"数据集配置已保存到 {output_path}")

def main(is_export_onnx=False):
    #先划分数据集
    image_dir = "train/images"  # 替换为你的图像目录
    label_dir = "train/labels"  # 替换为你的标签目录
    output_dir = "train/dataset"  # 替换为你的输出目录
    split_dataset_main(image_dir=image_dir,label_dir=label_dir,output_dir=output_dir,train=0.8,val=0.2,test=0.0)

    # 保存数据集配置到YAML文件
    download_data_yaml_path = 'data_config/data.yaml'  #这里的data.yaml是标注数据自带的配置路径，需要读取它的names替换custom_data.yaml中的names
    yaml_path = 'data_config/custom_data.yaml'  # 替换为你想要保存的路径
    save_data_config(input_path=download_data_yaml_path,output_path=yaml_path)

    # 然后训练模型
    model_path = '../model/fox.pt'  # 替换为你的模型路径
    train_args['data'] = yaml_path  # 使用刚刚保存的YAML配置文件路径
    results = train_yolo_segmentation(model_path=model_path, train_args=train_args)

    #训练后导出ONNX模型
    if is_export_onnx:
        output_onnx_path= '../model/exported_model.onnx'  # 替换为你想要保存的ONNX模型路径
        print("开始导出ONNX模型...")
        export_onnx(model_path,output_onnx_path)
        print("ONNX模型导出完成！")

    return  results

if __name__ == '__main__':
    main()
