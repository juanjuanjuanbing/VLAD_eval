from typing import List, Dict, Any
import os
import json
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI
import re
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class BBoxDetector:
    def __init__(self):
        # self.config = {
        #     "api_key": "EMPTY",
        #     "api_base": "http://localhost:8001/v1",
        #     "model_name": "./ms-swift/output/export_v5_11968",
        #     "temperature": 0.1,
        #     "top_p": 0.7,
        #     "timeout": 60
        # }

        self.config = {
            "api_key": "EMPTY",
            "api_base": "http://localhost:8002/v1",
            "model_name": "llava-hf/llava-v1.6-vicuna-7b-hf",
            "temperature": 0.1,
            "top_p": 0.7,
            "timeout": 60
        }
        # self.config = {
        #     "api_key": "EMPTY",
        #     "api_base": "http://localhost:8000/v1",
        #     "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        #     "temperature": 0.1,
        #     "top_p": 0.7,
        #     "timeout": 60
        # }
        self.client = None

    def __enter__(self):
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base"]
        )
        return self

    def __exit__(self, *args):
        if self.client:
            self.client.close()

    def encode_image(self, image: Image.Image) -> str:
        """将PIL图像编码为base64字符串"""
        with BytesIO() as buffer:
            image.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode()

    def parse_bbox_data(self, bbox_str: str) -> Dict[str, Any]:
        result = {
            "object": None,
            "bboxes": []
        }
        
        if "Qwen2.5" in self.config["model_name"]:
            # 原有Qwen2.5模型的解析逻辑保持不变
            stack = []
            current_num = ""
            in_outer = False
            in_inner = False
            current_bbox = []

            object_match = re.search(r'"label":\s*"([^"]+)"', bbox_str)
            if object_match:
                result["object"] = object_match.group(1)

            for char in bbox_str:
                if char == '[':
                    if not in_outer:
                        in_outer = True
                    elif not in_inner:
                        in_inner = True
                    continue
                
                if char == ']':
                    if in_inner:
                        if current_num:
                            current_bbox.append(int(current_num))
                            current_num = ""
                        
                        if len(current_bbox) == 4:
                            result["bboxes"].append(current_bbox)
                        current_bbox = []
                        in_inner = False
                    elif in_outer:
                        in_outer = False
                    continue
                
                if in_inner:
                    if char == ',':
                        if current_num:
                            current_bbox.append(int(current_num))
                            current_num = ""
                    elif char.isdigit() or char == '-':
                        current_num += char
        
        elif "export" in self.config["model_name"]:
            try:
                # 尝试解析object字段
                object_match = re.search(r'"object":\s*"([^"]+)"', bbox_str)
                if object_match:
                    result["object"] = object_match.group(1)
                
                # 使用正则表达式提取所有 (x,y) 坐标点
                pattern = r"\((\d+),\s*(\d+)\)"
                matches = re.findall(pattern, bbox_str)
                
                # 限制最多处理100个点（25个bbox）
                max_points = 100
                if len(matches) > max_points:
                    matches = matches[:max_points]
                    print(f"Warning: Only processing first {max_points} points (total: {len(matches)})")
                
                # 每两个点组成一个 bbox: [(x1,y1), (x2,y2)] → [x1,y1,x2,y2]
                for i in range(0, len(matches) - 1, 2):
                    x1, y1 = map(int, matches[i])
                    x2, y2 = map(int, matches[i + 1])
                    result["bboxes"].append([x1, y1, x2, y2])
                    
            except Exception as e:
                print(f"Error parsing export model bbox data: {e}")
                print(f"Raw bbox string: {bbox_str}")
        
        elif "llava" in self.config["model_name"]:
            # 使用正则表达式解析（处理非JSON格式的情况）
            try:
                # 尝试解析object字段（如果有的话）
                if result["object"] is None:
                    object_match = re.search(r'"label":\s*"([^"]+)"', bbox_str)
                    if object_match:
                        result["object"] = object_match.group(1)
                
                # 使用正则表达式提取bbox数组（支持浮点数）
                bbox_pattern = r'\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]'
                matches = re.findall(bbox_pattern, bbox_str)
                
                for match in matches:
                    if len(match) == 4:
                        # 解析为浮点数
                        x1, y1, x2, y2 = map(float, match)
                        
                        # 检查是否为归一化坐标（0-1范围）
                        if all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
                            # 转换为像素坐标（需要图像尺寸）
                            x1 = int(x1 * self.image_width)
                            y1 = int(y1 * self.image_height)
                            x2 = int(x2 * self.image_width)
                            y2 = int(y2 * self.image_height)
                        
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        # 检查重复
                        if bbox not in result["bboxes"]:
                            result["bboxes"].append(bbox)         
            except Exception as e:
                print(f"Error parsing llava model bbox data: {e}")
                print(f"Raw bbox string: {bbox_str}")
        elif "llava" in self.config["model_name"]:
            # TODO: 其他模型的解析逻辑
            pass
        
        return result

    def detect(self, image_path: str, prompt: str) -> Dict:
        """
        处理单张图片并返回JSON结果
        :param image_path: 图片路径
        :param prompt: 检测提示文本
        :return: 包含检测结果的字典
        """
        try:
            # 打开并编码图像
            image = Image.open(image_path)
            encoded_image = self.encode_image(image)
            
            # 存储图像尺寸用于后续坐标转换
            self.image_width, self.image_height = image.size
            
            # 发送请求到模型
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                # timeout=self.config["timeout"],
                # max_tokens=8192
            )
            
            # 解析结果
            content = response.choices[0].message.content
            # print(content)
            parsed_data = self.parse_bbox_data(content)
            print(parsed_data)
            
            # 返回结果字典（包含图像尺寸信息）
            result = {
                "image_path": image_path,
                "image_width": self.image_width,
                "image_height": self.image_height,
                "prompt": prompt,
                "object": parsed_data["object"],
                "bboxes": parsed_data["bboxes"]
            }
            return result
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {"error": str(e)}

# 使用示例
if __name__ == "__main__":
    detector = BBoxDetector()
    
    with detector:
        result = detector.detect(
            image_path="example.jpg",
            prompt="Detect all cars in the image and output bounding boxes in format [x1,y1,x2,y2]"
        )
        
        print("Detection result:")
        print(json.dumps(result, indent=2))



            