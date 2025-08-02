from jmespath.ast import projection
from sympy import false
import cv2
import time
from ultralytics import YOLO
from collections import Counter

class YOLOModel:
    def __init__(
            self,
            model_path='paperBin_dec_48_500.pt',
            project='return',
            save=True):
        """ YOLO模型类初始化函数
        :param model_path: 模型文件路径
        :param project: 预测结果保存的项目名称
        :param save: 是否保存预测结果
        """
        # 初始化YOLO模型相关参数
        self.model = None
        self.project = project
        self.save = save
        self.model_path = model_path
        self.label_counter = Counter()

    def load_model(self):
        """ 加载YOLO模型函数
        """
        # 加载模型
        self.model = YOLO(self.model_path)

    def predict(self,source, is_show=False):
        '''
        预测函数
        :param source: 输入源路径（单个图片或视频，或者是文件夹）
        :param is_show: 是否显示预测结果
        :return:
        '''
        # 预测图片
        results = self.model.predict(source=source,project='return', save=True)

        # 遍历每张图片的预测结果
        for result in results:
            # 获取当前图片所有检测到的类别ID（整数形式）
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            # 统计当前图片的类别
            for class_id in class_ids:
                # 将类别ID转换为实际标签名
                label_name = self.model.names[class_id]
                self.label_counter[label_name] += 1

            if is_show:
                # 可视化：获取绘制后的图像（带框等），是 numpy 格式
                plotted_img = result.plot()
                # 显示图片
                cv2.imshow("YOLOv8 Result", plotted_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # 打印结果
        print("识别标签统计:")
        for label, count in self.label_counter.items():
            print(f"{label}: {count}个")

        # 获取总检测对象数
        total_objects = sum(self.label_counter.values())
        print(f"\n总共检测到 {total_objects} 个对象")


if __name__ == "__main__":
    # 创建YOLO模型实例
    model_path = 'model/fox.pt'  # 模型文件路径
    project_path = 'return'  # 预测结果保存的项目名称
    yolo_model = YOLOModel(model_path=model_path, project=project_path, save=False)

    # 加载模型
    yolo_model.load_model()

    # 进行预测
    yolo_model.predict(source='测试图片/测试', is_show=True)
