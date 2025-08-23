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

class BBoxDetectorTransformers:
    def __init__(self):
        self.config = {
            "model_name": "ll-13/SkySenseGPT-7B-CLIP-ViT",  # 更换为目标模型
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "temperature": 0.1,
            "top_p": 0.7,
            "max_new_tokens": 512,
            "torch_dtype": torch.float16  # 适配7B模型的显存需求
        }
        self.model = None
        self.processor = None

    def __enter__(self):
        # 加载模型（保留信任远程代码参数，解决架构识别问题）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            torch_dtype=self.config["torch_dtype"],
            device_map="auto"
        )
        # 加载处理器（适配图文输入）
        self.processor = AutoProcessor.from_pretrained(
            self.config["model_name"],
            trust_remote_code=True,
            do_resize=True,
            size={"height": 512, "width": 512}  # CLIP模型常用尺寸
        )
        return self

    def __exit__(self, *args):
        if self.model:
            del self.model
            torch.cuda.empty_cache()

    def encode_image(self, image: Image.Image) -> Image.Image:
        """保持接口不变，返回PIL图像供处理器处理"""
        return image.convert("RGB")  # 确保图像为RGB格式

    def parse_bbox_data(self, bbox_str: str) -> List[List[int]]:
        """增强解析逻辑，适配SkySenseGPT可能的输出格式"""
        bboxes = []
        # 支持格式：[x1,y1,x2,y2] 或 多个边界框用逗号分隔
        import re
        # 匹配类似 [12,34,56,78] 的模式
        pattern = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
        matches = re.findall(pattern, bbox_str)
        for match in matches:
            try:
                bbox = [int(coord) for coord in match]
                bboxes.append(bbox)
            except ValueError:
                continue
        return bboxes

    def detect(self, image_path: str, prompt: str) -> Dict:
        """保持接口不变，优化图文输入处理"""
        try:
            # 打开图像并预处理
            image = Image.open(image_path)
            image = self.encode_image(image)  # 调用内部图像编码方法

            # 准备输入（文本+图像）
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
                    max_new_tokens=self.config["max_new_tokens"],
                    do_sample=True
                )

            # 解码输出
            content = self.processor.decode(outputs[0], skip_special_tokens=True)
            bboxes = self.parse_bbox_data(content)

            # 返回结果字典
            result = {
                "image_path": image_path,
                "prompt": prompt,
                "bboxes": bboxes,
                "raw_response": content  # 新增原始响应，方便调试
            }

            return result

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {"error": str(e), "image_path": image_path}


# 使用示例（保持原有调用方式）
if __name__ == "__main__":
    detector = BBoxDetectorTransformers()
    
    with detector:
        result = detector.detect(
            image_path="/home/zhiwei/VLAD_test/datasets/VLAD_Remote/VisDrone/VisDrone2019-DET-train/images/0000068_02104_d_0000006.jpg",
            prompt="Detect all cars in the image and output bounding boxes in [x1,y1,x2,y2] format with no other outputs."
        )
        
        print("Detection result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))