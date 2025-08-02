import os
import shutil
import argparse
import random
from tqdm import tqdm


def split_dataset(image_dir, label_dir, output_dir, ratios):
    """
    划分YOLO格式数据集

    参数:
    image_dir: 原始图像目录路径
    label_dir: 原始标签目录路径
    output_dir: 输出目录路径
    ratios: 划分比例字典，例如 {'train': 0.7, 'val': 0.2, 'test': 0.1}
    """
    # 验证比例总和是否为1
    if abs(sum(ratios.values()) - 1.0) > 0.001:
        raise ValueError("比例总和必须等于1")

    # 获取所有图像文件名（不带扩展名）
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    base_names = [os.path.splitext(f)[0] for f in image_files]
    random.shuffle(base_names)  # 随机打乱

    # 计算各集合的数量
    total = len(base_names)
    train_count = int(total * ratios['train'])
    val_count = int(total * ratios['val'])
    test_count = total - train_count - val_count

    # 创建输出目录结构
    datasets = {
        'train': base_names[:train_count],
        'val': base_names[train_count:train_count + val_count],
        'test': base_names[train_count + val_count:]
    }

    print(f"数据集划分结果:")
    print(f"总样本数: {total}")
    print(f"训练集: {len(datasets['train'])} ({len(datasets['train']) / total:.1%})")
    print(f"验证集: {len(datasets['val'])} ({len(datasets['val']) / total:.1%})")
    print(f"测试集: {len(datasets['test'])} ({len(datasets['test']) / total:.1%})")

    # 创建输出目录
    for set_name in datasets:
        os.makedirs(os.path.join(output_dir, 'images', set_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', set_name), exist_ok=True)

    # 复制文件到对应目录
    for set_name, names in datasets.items():
        print(f"\n正在处理 {set_name} 集...")
        for name in tqdm(names):
            # 查找匹配的图像文件
            for ext in ['.jpg', '.jpeg', '.png']:
                img_src = os.path.join(image_dir, name + ext)
                if os.path.exists(img_src):
                    img_dest = os.path.join(output_dir, 'images', set_name, os.path.basename(img_src))
                    shutil.copy2(img_src, img_dest)
                    break

            # 复制标签文件
            for ext in ['.txt']:
                label_src = os.path.join(label_dir, name + ext)
                if os.path.exists(label_src):
                    label_dest = os.path.join(output_dir, 'labels', set_name, os.path.basename(label_src))
                    shutil.copy2(label_src, label_dest)
                    break

    print("\n数据集划分完成！")

def split_dataset_main(
        image_dir="train/train/images",
        label_dir="train/train/labels",
        output_dir="train/train/dataset",
        train=0.8,
        val=0.2,
        test=0.0):
    """
    主函数，用于解析命令行参数并调用数据集划分函数
    :param image_dir: 原始图像目录路径
    :param label_dir: 原始标签目录路径
    :param output_dir: 输出目录路径
    :param train: 训练集比例 (0-1)
    :param val:验证集比例 (0-1)
    :param test:测试集比例 (0-1)
    :return:
    """
    parser = argparse.ArgumentParser(description="YOLO数据集划分工具")
    parser.add_argument("--image_dir", type=str, default=image_dir, help="原始图像目录路径")
    parser.add_argument("--label_dir", type=str, default=label_dir, help="原始标签目录路径")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="输出目录路径")
    parser.add_argument("--train", type=float, default=train, help="训练集比例 (0-1)")
    parser.add_argument("--val", type=float, default=val, help="验证集比例 (0-1)")
    parser.add_argument("--test", type=float, default=test, help="测试集比例 (0-1)")

    args = parser.parse_args()

    # 设置划分比例
    ratios = {
        'train': args.train,
        'val': args.val,
        'test': args.test
    }

    # 执行数据集划分
    split_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        ratios=ratios
    )

if __name__ == "__main__":
    print("开始划分数据集...")
    # split_dataset_main()