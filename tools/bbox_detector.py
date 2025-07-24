from typing import List, Dict
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
        self.config = {
            "api_key": "EMPTY",
            "api_base": "http://localhost:8001/v1",
            "model_name": "./ms-swift/output/export_v3_23936",
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

    def parse_bbox_data(self, bbox_str: str) -> List[List[int]]:
        bboxes = []
        if "Qwen2.5" in self.config["model_name"]:
            # 原有Qwen2.5模型的解析逻辑保持不变
            stack = []
            current_num = ""
            in_outer = False
            in_inner = False
            current_bbox = []
            
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
                            bboxes.append(current_bbox)
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
                # 使用正则表达式提取所有 (x,y) 坐标点
                pattern = r"\((\d+),\s*(\d+)\)"
                matches = re.findall(pattern, bbox_str)
                
                # 每两个点组成一个 bbox: [(x1,y1), (x2,y2)] → [x1,y1,x2,y2]
                for i in range(0, len(matches) - 1, 2):
                    x1, y1 = map(int, matches[i])
                    x2, y2 = map(int, matches[i + 1])
                    bboxes.append([x1, y1, x2, y2])
                    
            except Exception as e:
                print(f"Error parsing export model bbox data: {e}")
                print(f"Raw bbox string: {bbox_str}")
        
        elif "model_3" in self.config["model_name"]:
            # TODO: 其他模型的解析逻辑
            pass
        
        return bboxes
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
                max_tokens=8192
            )
            
            # 解析结果
            content = response.choices[0].message.content
            # print(f"Model response: {content}")
            bboxes = self.parse_bbox_data(content)
            # print(content)
            # 返回结果字典
            result = {
                "image_path": image_path,
                "prompt": prompt,
                "bboxes": bboxes
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {"error": str(e)}
        
class BBoxDetectorTransformers:
    def __init__(self):
        self.config = {
            "model_name": 'MBZUAI/GeoPixel-7B',
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "temperature": 0.1,
            "top_p": 0.7,
            "max_new_tokens": 512
        }
        self.model = None
        self.processor = None

    def __enter__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True  # 关键参数
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config["model_name"],
            trust_remote_code=True  # 处理器也需要
        )
        return self

    def __exit__(self, *args):
        if self.model:
            del self.model
            torch.cuda.empty_cache()

    def encode_image(self, image: Image.Image) -> Image.Image:
        """直接返回PIL图像，Transformers处理器会处理图像"""
        return image

    def parse_bbox_data(self, bbox_str: str) -> List[List[int]]:
        bboxes = []
        if "Qwen2.5" in self.config["model_name"]:
            # 原有Qwen2.5模型的解析逻辑保持不变
            stack = []
            current_num = ""
            in_outer = False
            in_inner = False
            current_bbox = []
            
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
                            bboxes.append(current_bbox)
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
                # 使用正则表达式提取所有 (x,y) 坐标点
                pattern = r"\((\d+),\s*(\d+)\)"
                matches = re.findall(pattern, bbox_str)
                
                # 每两个点组成一个 bbox: [(x1,y1), (x2,y2)] → [x1,y1,x2,y2]
                for i in range(0, len(matches) - 1, 2):
                    x1, y1 = map(int, matches[i])
                    x2, y2 = map(int, matches[i + 1])
                    bboxes.append([x1, y1, x2, y2])
                    
            except Exception as e:
                print(f"Error parsing export model bbox data: {e}")
                print(f"Raw bbox string: {bbox_str}")
        
        elif "model_3" in self.config["model_name"]:
            # TODO: 其他模型的解析逻辑
            pass
        
        return bboxes

    def detect(self, image_path: str, prompt: str) -> Dict:
        """
        处理单张图片并返回JSON结果
        :param image_path: 图片路径
        :param prompt: 检测提示文本
        :return: 包含检测结果的字典
        """
        try:
            # 打开图像
            image = Image.open(image_path)
            
            # 准备输入
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.config["device"])
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    max_new_tokens=self.config["max_new_tokens"]
                )
            
            # 解码输出
            content = self.processor.decode(outputs[0], skip_special_tokens=True)
            bboxes = self.parse_bbox_data(content)
            
            # 返回结果字典
            result = {
                "image_path": image_path,
                "prompt": prompt,
                "bboxes": bboxes
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



            