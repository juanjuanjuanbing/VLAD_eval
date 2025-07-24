import os
import numpy as np
from collections import defaultdict

def parse_label_file(label_path):
    """解析标签文件，返回格式为[class_id, x_center, y_center, width, height]的列表"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # 至少包含class_id和4个坐标
                    class_id = int(parts[0])
                    box = list(map(float, parts[1:5]))
                    boxes.append([class_id] + box)
    return boxes

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 转换为中心坐标到角坐标
    box1 = [box1[0] - box1[2]/2, box1[1] - box1[3]/2, 
            box1[0] + box1[2]/2, box1[1] + box1[3]/2]
    box2 = [box2[0] - box2[2]/2, box2[1] - box2[3]/2, 
            box2[0] + box2[2]/2, box2[1] + box2[3]/2]
    
    # 计算交集区域
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算并集区域
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou

def calculate_ap(recalls, precisions):
    """计算AP (11点插值法)"""
    interp_precisions = []
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if np.any(mask):
            interp_precisions.append(np.max(precisions[mask]))
        else:
            interp_precisions.append(0)
    return np.mean(interp_precisions)

def evaluate_dataset(image_paths, class_ids):
    """评估指定数据集和类别"""
    gt_boxes = []
    pred_boxes = []
    
    for img_path in image_paths:
        # 获取图片文件名（不带路径）
        img_filename = os.path.basename(img_path)
        label_filename = img_filename.replace('.jpg', '.txt')
        
        # GT标签路径
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        # 预测结果路径统一在 ./results/labels/ 下
        file_name = 'ms-swift/output/export_v3_23936/fixed' 
        # 'ms-swift/output/export_v3_23936/fixed' 
        #'Qwen/Qwen2.5-VL-7B-Instruct/fixed' 
        pred_path = os.path.join(f'./results/eval/coco_labels/{file_name}', label_filename)
        print(pred_path)
        
        gt_boxes.append(parse_label_file(label_path))
        pred_boxes.append(parse_label_file(pred_path))
    
    # 存储每个类别的结果
    class_results = {class_id: {'tp': [], 'fp': [], 'scores': [], 'n_gt': 0} 
                     for class_id in class_ids}
    
    # 遍历每张图片
    for gt_img, pred_img in zip(gt_boxes, pred_boxes):
        # 按类别分组
        gt_by_class = defaultdict(list)
        for gt in gt_img:
            if gt[0] in class_ids:
                gt_by_class[gt[0]].append(gt[1:])
        
        pred_by_class = defaultdict(list)
        for pred in pred_img:
            if pred[0] in class_ids:
                pred_by_class[pred[0]].append({'box': pred[1:], 'score': 1.0})  # 假设置信度为1.0
        
        # 对每个类别处理
        for class_id in class_ids:
            class_gt = gt_by_class.get(class_id, [])
            class_pred = pred_by_class.get(class_id, [])
            
            class_results[class_id]['n_gt'] += len(class_gt)
            
            if not class_pred:
                continue
                
            # 按置信度排序预测框
            class_pred_sorted = sorted(class_pred, key=lambda x: x['score'], reverse=True)
            
            # 初始化匹配状态
            gt_matched = [False] * len(class_gt)
            
            for pred in class_pred_sorted:
                pred_box = pred['box']
                max_iou = 0
                best_gt_idx = -1
                
                # 计算与所有GT框的IOU
                for gt_idx, gt_box in enumerate(class_gt):
                    if not gt_matched[gt_idx]:
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            best_gt_idx = gt_idx
                
                # 判断是否匹配
                if max_iou >= 0.5:  # IoU阈值为0.5
                    gt_matched[best_gt_idx] = True
                    class_results[class_id]['tp'].append(1)
                    class_results[class_id]['fp'].append(0)
                else:
                    class_results[class_id]['tp'].append(0)
                    class_results[class_id]['fp'].append(1)
                
                class_results[class_id]['scores'].append(pred['score'])
    
    # 计算每个类别的AP和F1
    aps = []
    f1s = []
    
    for class_id in class_ids:
        tp = np.array(class_results[class_id]['tp'])
        fp = np.array(class_results[class_id]['fp'])
        scores = np.array(class_results[class_id]['scores'])
        n_gt = class_results[class_id]['n_gt']
        
        if len(tp) == 0:
            aps.append(0)
            f1s.append(0)
            continue
            
        # 按分数排序
        sort_idx = np.argsort(-scores)
        tp = tp[sort_idx]
        fp = fp[sort_idx]
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算召回率和精确率
        recalls = tp_cumsum / max(1, n_gt)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
        
        # 计算AP
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
        
        # 计算F1
        f1 = 2 * precisions[-1] * recalls[-1] / max(precisions[-1] + recalls[-1], 1e-6)
        f1s.append(f1)
    
    # 计算mAP和mF1
    map_nc = np.mean(aps)
    mf1 = np.mean(f1s)
    
    return map_nc, mf1

def main():
    # 定义要计算的类别

    visdrone_classes = [3,8,5]  # 示例: VisDrone类别ID
    xview_classes = [5,6,9,48]     # 示例: xView类别ID
    
    # 读取测试图片列表
    with open('./datasets/VLAD_Remote/test_image_list.txt', 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    # 分离VisDrone和xView图片路径
    visdrone_paths = [p for p in image_paths if 'VisDrone' in p]
    xview_paths = [p for p in image_paths if 'xView' in p]
    
    # 计算VisDrone指标
    visdrone_map, visdrone_mf1 = evaluate_dataset(visdrone_paths, visdrone_classes)
    
    # 计算xView指标
    xview_map, xview_mf1 = evaluate_dataset(xview_paths, xview_classes)
    
    # 输出结果
    print("VisDrone Results (Classes: {}):".format(visdrone_classes))
    print("- mAPnc: {:.4f}".format(visdrone_map))
    print("- mF1: {:.4f}".format(visdrone_mf1))
    
    print("\nxView Results (Classes: {}):".format(xview_classes))
    print("- mAPnc: {:.4f}".format(xview_map))
    print("- mF1: {:.4f}".format(xview_mf1))

if __name__ == '__main__':
    main()


