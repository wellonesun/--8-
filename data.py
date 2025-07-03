import os
import xml.etree.ElementTree as ET
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# 数据集路径配置 - 根据您的实际目录结构修改
DATA_DIR = 'D:\\BaiduNetdiskDownload\\train\\train'
IMAGE_DIR = os.path.join(DATA_DIR, 'image')
BOX_DIR = os.path.join(DATA_DIR, 'box')
CLEAN_DATA_DIR = 'clean_data'

# 创建清洗后数据目录
os.makedirs(os.path.join(CLEAN_DATA_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(CLEAN_DATA_DIR, 'labels'), exist_ok=True)

# 目标类别映射
CLASS_MAP = {
    'holothurian': 0,  # 海参
    'echinus': 1,      # 海胆
    'scallop': 2,      # 扇贝
    'starfish': 3      # 海星
}

def parse_xml_annotation(xml_path):
    """解析XML标注文件"""
    if not os.path.exists(xml_path):
        print(f"警告：标注文件 {xml_path} 不存在")
        return []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            
            # 忽略水草类别
            if cls_name == 'waterweeds':
                continue
                
            # 只处理四类目标生物
            if cls_name not in CLASS_MAP:
                print(f"警告：发现未知类别 '{cls_name}'，跳过")
                continue
                
            bbox = obj.find('bndbox')
            try:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
            except (AttributeError, ValueError):
                print(f"警告：标注文件 {xml_path} 中的边界框格式错误，跳过")
                continue
            
            annotations.append({
                'class': cls_name,
                'class_id': CLASS_MAP[cls_name],
                'bbox': [xmin, ymin, xmax, ymax]
            })
        
        return annotations
    except ET.ParseError:
        print(f"错误：无法解析XML文件 {xml_path}")
        return []

def visualize_annotation(image_path, annotations):
    """可视化标注结果（用于调试）"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for ann in annotations:
        xmin, ymin, xmax, ymax = ann['bbox']
        class_name = ann['class']
        color = {
            'holothurian': (255, 0, 0),    # 红色
            'echinus': (0, 255, 0),        # 绿色
            'scallop': (0, 0, 255),        # 蓝色
            'starfish': (255, 255, 0)      # 黄色
        }.get(class_name, (128, 128, 128))  # 默认灰色
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, class_name, (xmin, ymin-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(os.path.basename(image_path))
    plt.axis('off')
    plt.show()

def clean_noisy_data(image_name):
    """清洗噪声标注数据（u前缀文件）"""
    img_path = os.path.join(IMAGE_DIR, image_name)
    xml_path = os.path.join(BOX_DIR, image_name.replace('.jpg', '.xml'))
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图像 {img_path}，跳过")
        return None, []
    
    # 解析原始标注
    orig_annotations = parse_xml_annotation(xml_path)
    
    # 噪声数据处理策略
    cleaned_annotations = []
    for ann in orig_annotations:
        xmin, ymin, xmax, ymax = ann['bbox']
        
        # 确保坐标在图像范围内
        height, width = img.shape[:2]
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(1, min(xmax, width))
        ymax = max(1, min(ymax, height))
        
        # 确保边界框有效
        if xmin >= xmax or ymin >= ymax:
            print(f"警告：无效边界框 ({xmin},{ymin},{xmax},{ymax})，跳过")
            continue
            
        # 计算边界框面积
        bbox_area = (xmax - xmin) * (ymax - ymin)
        img_area = width * height
        
        # 过滤过小或过大的边界框
        if bbox_area < 100:  # 小于100像素
            print(f"警告：跳过过小目标 ({bbox_area}像素)")
            continue
        if bbox_area > img_area * 0.5:  # 大于图像面积的50%
            print(f"警告：跳过过大目标 ({bbox_area}像素)")
            continue
            
        # 更新边界框
        ann['bbox'] = [xmin, ymin, xmax, ymax]
        cleaned_annotations.append(ann)
    
    # 调试：可视化清洗结果
    if cleaned_annotations:
        print(f"图像 {image_name} 清洗后保留 {len(cleaned_annotations)} 个标注")
        visualize_annotation(img_path, cleaned_annotations)
    
    return img, cleaned_annotations

def save_yolo_annotation(image_path, image_name, annotations):
    """保存YOLO格式的标注文件"""
    base_name = os.path.splitext(image_name)[0]
    label_path = os.path.join(CLEAN_DATA_DIR, 'labels', f"{base_name}.txt")
    
    # 获取图像尺寸
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图像 {image_path} 以获取尺寸")
        return
    
    height, width = img.shape[:2]
    
    with open(label_path, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            xmin, ymin, xmax, ymax = ann['bbox']
            
            # 转换为YOLO格式 (归一化中心坐标和宽高)
            cx = (xmin + xmax) / 2 / width
            cy = (ymin + ymax) / 2 / height
            nw = (xmax - xmin) / width
            nh = (ymax - ymin) / height
            
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

def prepare_dataset():
    """准备训练数据集"""
    # 获取所有图片文件（支持.jpg和.png）
    all_images = [f for f in os.listdir(IMAGE_DIR) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not all_images:
        print(f"错误：在 {IMAGE_DIR} 中没有找到任何图片文件！")
        return 0
    
    print(f"找到 {len(all_images)} 张图片")
    
    clean_data = []
    noisy_data = []
    
    # 分类干净数据和噪声数据
    for img_name in all_images:
        if img_name.startswith('c'):
            clean_data.append(img_name)
        elif img_name.startswith('u'):
            noisy_data.append(img_name)
        else:
            print(f"警告：图片 {img_name} 没有c或u前缀，将作为干净数据处理")
            clean_data.append(img_name)
    
    print(f"干净数据: {len(clean_data)} 张, 噪声数据: {len(noisy_data)} 张")
    
    processed_count = 0
    
    # 处理干净数据
    for img_name in tqdm(clean_data, desc="处理干净数据"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        xml_path = os.path.join(BOX_DIR, img_name.replace('.jpg', '.xml')
                                 .replace('.jpeg', '.xml')
                                 .replace('.png', '.xml'))
        
        if not os.path.exists(xml_path):
            print(f"警告：干净数据 {img_name} 的标注文件不存在，跳过")
            continue
            
        annotations = parse_xml_annotation(xml_path)
        if not annotations:
            print(f"警告：干净数据 {img_name} 没有有效标注，跳过")
            continue
            
        # 复制图像到清洗目录
        shutil.copy(img_path, os.path.join(CLEAN_DATA_DIR, 'images', img_name))
        
        # 保存YOLO格式标注
        save_yolo_annotation(img_path, img_name, annotations)
        processed_count += 1
    
    # 处理噪声数据
    for img_name in tqdm(noisy_data, desc="处理噪声数据"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img, annotations = clean_noisy_data(img_name)
        
        if img is None or not annotations:
            print(f"警告：噪声数据 {img_name} 没有有效数据，跳过")
            continue
            
        # 保存处理后的图像
        cv2.imwrite(os.path.join(CLEAN_DATA_DIR, 'images', img_name), img)
        
        # 保存清洗后的标注
        save_yolo_annotation(img_path, img_name, annotations)
        processed_count += 1
    
    print(f"成功处理 {processed_count} 张图片")
    return processed_count

def split_dataset(test_size=0.1, val_size=0.1):
    """划分训练集、验证集、测试集"""
    images_dir = os.path.join(CLEAN_DATA_DIR, 'images')
    labels_dir = os.path.join(CLEAN_DATA_DIR, 'labels')
    
    all_images = [f for f in os.listdir(images_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not all_images:
        print("错误：清洗后的数据目录中没有图片，请先运行 prepare_dataset()")
        return
    
    print(f"清洗后数据集包含 {len(all_images)} 张图片")
    
    # 确保有足够的数据进行划分
    if len(all_images) < 10:
        test_size = 0
        val_size = 0
        print("警告：数据量太少，不进行划分")
    
    # 第一次划分：分离测试集
    if test_size > 0:
        train_val, test = train_test_split(all_images, test_size=test_size, random_state=42)
    else:
        train_val = all_images
        test = []
    
    # 第二次划分：分离验证集
    if val_size > 0:
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
    else:
        train = train_val
        val = []
    
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(CLEAN_DATA_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(CLEAN_DATA_DIR, split, 'labels'), exist_ok=True)
    
    # 移动文件到对应目录
    for img_name in train:
        move_to_split(img_name, 'train')
    
    for img_name in val:
        move_to_split(img_name, 'val')
    
    for img_name in test:
        move_to_split(img_name, 'test')
    
    print(f"数据集划分完成: 训练集 {len(train)}, 验证集 {len(val)}, 测试集 {len(test)}")

def move_to_split(image_name, split):
    """移动文件到指定分割目录"""
    base_name = os.path.splitext(image_name)[0]
    
    # 移动图像
    src_img = os.path.join(CLEAN_DATA_DIR, 'images', image_name)
    dst_img = os.path.join(CLEAN_DATA_DIR, split, 'images', image_name)
    shutil.move(src_img, dst_img)
    
    # 移动标注
    label_name = f"{base_name}.txt"
    src_label = os.path.join(CLEAN_DATA_DIR, 'labels', label_name)
    if os.path.exists(src_label):
        dst_label = os.path.join(CLEAN_DATA_DIR, split, 'labels', label_name)
        shutil.move(src_label, dst_label)

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    
    # 准备数据集
    processed_count = prepare_dataset()
    
    # 如果有足够数据，进行划分
    if processed_count > 0:
        split_dataset()
    else:
        print("没有处理任何数据，请检查输入路径和文件")
