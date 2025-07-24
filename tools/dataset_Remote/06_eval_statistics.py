import json
import os
import glob
from PIL import Image

def get_image_size(image_path):
    """获取图像尺寸"""
    try:
        with Image.open(image_path) as img:
            return img.size  # 返回 (width, height)
    except Exception as e:
        print(f"无法获取图像尺寸: {image_path}, 错误: {e}")
        return None

def convert_json_to_coco_txt(json_file_path, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历所有JSON文件
    for json_file in glob.glob(os.path.join(json_file_path, '*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 判断数据来源
        image_path = data['image_path']
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在: {image_path}")
            continue
            
        if 'xview' in image_path.lower():
            # xView类别映射
            class_mapping = {'Car': 5, 'Bus': 6, 'Truck': 9,'Building': 48}
        else:
            # VisDrone类别映射
            class_mapping = {'Car': 3, 'Bus': 8, 'Truck': 5}
        
        # 获取图像尺寸
        img_size = get_image_size(image_path)
        if img_size is None:
            print(f"跳过文件 {json_file} 因为无法获取图像尺寸")
            continue
            
        img_width, img_height = img_size
        
        # 准备输出内容
        output_lines = []
        detections = data['detections']
        
        for class_name, bboxes in detections.items():
            if class_name not in class_mapping:
                continue
                
            class_id =  class_mapping[class_name]
            
            for bbox in bboxes:
                if len(bbox) != 4:
                    print(f"警告: {json_file} 中的 {class_name} 有无效的bbox: {bbox}")
                    continue
                    
                x1, y1, x2, y2 = bbox
                
                # 计算中心点和宽高（归一化）
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # 确保坐标在0-1范围内
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                
                # 添加到输出
                output_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 写入TXT文件
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_file = os.path.join(output_folder, f"{base_name}.txt")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        
        print(f"成功转换: {json_file} -> {output_file} (图像尺寸: {img_width}x{img_height})")

# 使用示例
file_name = 'ms-swift/output/export_v3_23936/fixed' 
# 'ms-swift/output/export_v3_23936/fixed' 
#'Qwen/Qwen2.5-VL-7B-Instruct/fixed' 
json_folder = f'./results/eval/labels/{file_name}'  # JSON文件所在文件夹
output_folder = f'./results/eval/coco_labels/{file_name}'  # 输出COCO格式TXT文件的文件夹

convert_json_to_coco_txt(json_folder, output_folder)