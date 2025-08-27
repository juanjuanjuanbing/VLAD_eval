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

def evaluate_dataset(image_paths, class_ids, model_path):
    """评估指定数据集和类别（仅计算IoU≥0.5的匹配）"""
    gt_boxes = []
    pred_boxes = []

    # 初始化统计变量
    total_gt = 0      # 所有类别的GT总数
    total_matched = 0  # 所有类别的成功匹配数（TP）

    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        label_filename = img_filename.replace('.jpg', '.txt')
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        pred_path = os.path.join(f'./results/eval/coco_labels/{model_path}', label_filename)        
        gt_boxes.append(parse_label_file(label_path))
        pred_boxes.append(parse_label_file(pred_path))
    
    class_results = {class_id: {'tp': [], 'fp': [], 'scores': [], 'n_gt': 0, 'ious': []} 
                     for class_id in class_ids}
    
    for gt_img, pred_img in zip(gt_boxes, pred_boxes):
        gt_by_class = defaultdict(list)
        for gt in gt_img:
            if gt[0] in class_ids:
                gt_by_class[gt[0]].append(gt[1:])
        
        pred_by_class = defaultdict(list)
        for pred in pred_img:
            if pred[0] in class_ids:
                pred_by_class[pred[0]].append({'box': pred[1:], 'score': 1.0})
        
        for class_id in class_ids:
            class_gt = gt_by_class.get(class_id, [])
            class_pred = pred_by_class.get(class_id, [])
            class_results[class_id]['n_gt'] += len(class_gt)
            total_gt += len(class_gt)  # 累加到全局GT总数
            
            if not class_pred:
                continue
                
            class_pred_sorted = sorted(class_pred, key=lambda x: x['score'], reverse=True)
            gt_matched = [False] * len(class_gt)
            
            for pred in class_pred_sorted:
                pred_box = pred['box']
                max_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(class_gt):
                    if not gt_matched[gt_idx]:
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            best_gt_idx = gt_idx
                
                # 关键修改：仅记录IoU≥0.5的匹配
                if max_iou >= 0.5:
                    class_results[class_id]['ious'].append(max_iou)
                
                if max_iou >= 0.5:
                    gt_matched[best_gt_idx] = True
                    class_results[class_id]['tp'].append(1)
                    class_results[class_id]['fp'].append(0)
                    total_matched += 1  # 成功匹配数+1
                else:
                    class_results[class_id]['tp'].append(0)
                    class_results[class_id]['fp'].append(1)
                
                class_results[class_id]['scores'].append(pred['score'])
    
    aps = []
    f1s = []
    ious = []
    
    for class_id in class_ids:
        tp = np.array(class_results[class_id]['tp'])
        fp = np.array(class_results[class_id]['fp'])
        scores = np.array(class_results[class_id]['scores'])
        n_gt = class_results[class_id]['n_gt']
        class_ious = class_results[class_id]['ious']
        
        if len(tp) == 0:
            aps.append(0)
            f1s.append(0)
            ious.append(0)
            continue
            
        sort_idx = np.argsort(-scores)
        tp = tp[sort_idx]
        fp = fp[sort_idx]
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / max(1, n_gt)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
        
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
        
        f1 = 2 * precisions[-1] * recalls[-1] / max(precisions[-1] + recalls[-1], 1e-6)
        f1s.append(f1)
        
        mean_iou = np.mean(class_ious) if class_ious else 0
        ious.append(mean_iou)
    
    map_nc = np.mean(aps)
    mf1 = np.mean(f1s)
    miou = np.mean(ious)

    # 打印统计信息
    print(f"\n[统计] GT总数: {total_gt}, 成功匹配数(TP): {total_matched}")
    print(f"[统计] 匹配率: {total_matched / max(1, total_gt):.2%}")
    
    return map_nc, mf1, miou

def main():
    # 定义候选模型列表
    candidate_models = [
        'ms-swift/output/export_v5_11968/fixed/default',
        'ms-swift/output/export_v5_11968/open/mapping1',
        'ms-swift/output/export_v5_11968/open/mapping2',
        'Qwen/Qwen2.5-VL-7B-Instruct/fixed/default',
        'Qwen/Qwen2.5-VL-7B-Instruct/open/mapping1',
        'Qwen/Qwen2.5-VL-7B-Instruct/open/mapping2',
        'Falcon/fixed/default',
        'Falcon/open/mapping1',
        'Falcon/open/mapping2',
        'llava-hf/llava-v1.6-vicuna-7b-hf/fixed/default'
    ]
    
    # 定义要计算的类别
    visdrone_classes = [3, 8, 5]  # VisDrone类别ID
    xview_classes = [48]          # xView类别ID
    
    # 读取测试图片列表
    with open('./datasets/VLAD_Remote/test_image_list.txt', 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    # 分离VisDrone和xView图片路径，并统计样本量
    visdrone_paths = [p for p in image_paths if 'VisDrone' in p]
    xview_paths = [p for p in image_paths if 'xView' in p]
    n_visdrone = len(visdrone_paths)
    n_xview = len(xview_paths)
    total_samples = n_visdrone + n_xview
    
    # 计算权重
    weight_visdrone = n_visdrone / total_samples
    weight_xview = n_xview / total_samples
    
    # 准备结果文件
    result_file = './results/eval/model_comparison.txt'
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    with open(result_file, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"VisDrone样本数: {n_visdrone}, xView样本数: {n_xview}\n")
        f.write(f"权重分配: VisDrone={weight_visdrone:.2f}, xView={weight_xview:.2f}\n")
        f.write("="*50 + "\n\n")
        
        for model in candidate_models:
            f.write(f"Evaluating model: {model}\n")
            
            # 计算VisDrone指标
            visdrone_map, visdrone_mf1, visdrone_miou = evaluate_dataset(
                visdrone_paths, visdrone_classes, model)
            
            # 计算xView指标
            xview_map, xview_mf1, xview_miou = evaluate_dataset(
                xview_paths, xview_classes, model)
            
            # 按样本量加权平均
            avg_map = (visdrone_map * weight_visdrone + xview_map * weight_xview)
            avg_mf1 = (visdrone_mf1 * weight_visdrone + xview_mf1 * weight_xview)
            avg_miou = (visdrone_miou * weight_visdrone + xview_miou * weight_xview)
            
            # 写入结果
            f.write("VisDrone Results (Classes: {}):\n".format(visdrone_classes))
            f.write("- mAPnc: {:.4f}\n".format(visdrone_map))
            f.write("- mF1: {:.4f}\n".format(visdrone_mf1))
            f.write("- mIoU: {:.4f}\n".format(visdrone_miou))
            
            f.write("\nxView Results (Classes: {}):\n".format(xview_classes))
            f.write("- mAPnc: {:.4f}\n".format(xview_map))
            f.write("- mF1: {:.4f}\n".format(xview_mf1))
            f.write("- mIoU: {:.4f}\n".format(xview_miou))
            
            f.write("\nWeighted Average (by sample size):\n")
            f.write("- mAPnc: {:.4f}\n".format(avg_map))
            f.write("- mF1: {:.4f}\n".format(avg_mf1))
            f.write("- mIoU: {:.4f}\n".format(avg_miou))
            
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"Evaluation completed. Results saved to {result_file}")

if __name__ == '__main__':
    main()