import json
import os
import glob
import argparse
from PIL import Image

def get_image_size(image_path):
    """获取图像尺寸"""
    try:
        with Image.open(image_path) as img:
            return img.size  # 返回 (width, height)
    except Exception as e:
        print(f"无法获取图像尺寸: {image_path}, 错误: {e}")
        return None

def get_class_mappings(task_type):
    """获取所有类别映射"""
    if task_type == 'fixed':
        return {
            'default': {
                'xview': {'Car': 5, 'Bus': 6, 'Truck': 9, 'Building': 48},
                'visdrone': {'Car': 3, 'Bus': 8, 'Truck': 5}
            }
        }
    elif task_type == 'open':
        return {
            'mapping1': {  # 第一种映射
                'xview': {
                    'facility': 48,
                    'passenger vehicle': 5,
                    'cargo truck': 9
                },
                'visdrone': {
                    'passenger vehicle': 3,
                    'cargo truck': 5
                }
            },
            'mapping2': {  # 第二种映射
                'xview': {
                    'structure': 48,
                    'transportation': 5,
                    'heavy vehicle': 9
                },
                'visdrone': {
                    'transportation': 3,
                    'heavy vehicle': 5
                }
            }
        }

def convert_json_to_coco_txt(json_file_path, output_folder, task_type='fixed'):
    # 创建主输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有类别映射
    all_mappings = get_class_mappings(task_type)
    
    # 遍历所有JSON文件
    for json_file in glob.glob(os.path.join(json_file_path, '*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 判断数据来源
        image_path = data['image_path']
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在: {image_path}")
            continue
            
        # 获取图像尺寸
        img_size = get_image_size(image_path)
        if img_size is None:
            print(f"跳过文件 {json_file} 因为无法获取图像尺寸")
            continue
            
        img_width, img_height = img_size
        
        # 确定数据集类型
        dataset_type = 'xview' if 'xview' in image_path.lower() else 'visdrone'
        
        # 处理每种映射
        for mapping_name, mappings in all_mappings.items():
            # 创建子文件夹
            mapping_folder = os.path.join(output_folder, mapping_name)
            os.makedirs(mapping_folder, exist_ok=True)
            
            # 获取当前数据集类型的映射
            class_mapping = mappings[dataset_type]
            
            # 准备输出内容
            output_lines = []
            detections = data['detections']
            
            for class_name, bboxes in detections.items():
                # 查找匹配的类别ID
                matched_class_id = None
                for key in class_mapping:
                    if key.lower() in class_name.lower():
                        matched_class_id = class_mapping[key]
                        break
                        
                if matched_class_id is None:
                    continue
                    
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
                    output_lines.append(f"{matched_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # 写入TXT文件
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            output_file = os.path.join(mapping_folder, f"{base_name}.txt")
            
            with open(output_file, 'w') as f:
                f.write('\n'.join(output_lines))
            
            print(f"成功转换: {json_file} -> {output_file} (映射: {mapping_name}, 图像尺寸: {img_width}x{img_height})")

def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='将JSON标注转换为COCO格式的TXT文件')
    parser.add_argument('--file_name', type=str, required=True,
                        help='输入文件路径，如 ms-swift/output/export_v5_11968/fixed')
    parser.add_argument('--task_type', type=str, choices=['fixed', 'open'], required=True,
                        help='任务类型: fixed 或 open')
    
    args = parser.parse_args()
    
    # 设置输入输出路径
    json_folder = f'./results/eval/labels/{args.file_name}/{args.task_type}'
    output_folder = f'./results/eval/coco_labels/{args.file_name}/{args.task_type}'
    
    # 执行转换
    convert_json_to_coco_txt(json_folder, output_folder, args.task_type)

if __name__ == '__main__':
    main()

# # 处理fixed任务
# python ./tools/dataset_Remote/06_eval_statistics.py --file_name ms-swift/output/export_v5_11968 --task_type fixed

# # 处理open任务
# python ./tools/dataset_Remote/06_eval_statistics.py --file_name ms-swift/output/export_v5_11968 --task_type open

# # 处理fixed任务
# python ./tools/dataset_Remote/06_eval_statistics.py --file_name Qwen/Qwen2.5-VL-7B-Instruct --task_type fixed

# # 处理open任务
# python ./tools/dataset_Remote/06_eval_statistics.py --file_name Qwen/Qwen2.5-VL-7B-Instruct --task_type open