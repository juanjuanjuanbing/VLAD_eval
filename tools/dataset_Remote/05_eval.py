import os
import sys
import json
from typing import List
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import os
import json
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI
from tools.bbox_detector import BBoxDetector
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class BBoxEvaluator:
    def __init__(
        self, 
        image_list_path: str, 
        class_list: List[str], 
        output_dir: str = "./results/eval/labels",
        task_config: str = "fixed",  # "fixed" or "open" or "open_ended"
        instructions_path: str = None  # Path to instructions JSON file
    ):
        """
        Initialize the evaluator
        
        :param image_list_path: Path to text file containing image paths (one per line)
        :param class_list: List of classes to detect (used when task_config is "fixed")
        :param output_dir: Directory to save JSON results (default: ./results/eval/labels)
        :param task_config: "fixed" for fixed classes or "open" for open classes
        :param instructions_path: Path to JSON file containing instructions (for open_ended task)
        """
        self.image_list_path = image_list_path
        self.class_list = class_list
        self.task_config = task_config.lower()
        self.instructions_path = instructions_path
        
        if self.task_config not in ["fixed", "open", "open_ended"]:
            raise ValueError("task_config must be either 'fixed', 'open', or 'open_ended'")
            
        # Define open classes if needed
        if self.task_config == "open":
            self.class_list = [
                "facility",
                "structure",
                "passenger vehicle",
                "transportation",
                "cargo truck ",
                "heavy vehicle"
            ]
        
        self.detector = BBoxDetector()
        self.output_dir = f"{output_dir}/{self.detector.config['model_name']}/{self.task_config}"
        
        # Load instructions if open_ended task
        if self.task_config == "open_ended":
            if not instructions_path:
                raise ValueError("instructions_path must be provided for open_ended task")
            with open(instructions_path) as f:
                self.instructions = json.load(f)
            # Create mapping from image path to instruction for faster lookup
            self.instruction_map = {item['image_path']: item for item in self.instructions}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_prompt(self, class_name: str = None, instruction_data: dict = None) -> str:
        """Generate the detection prompt for a given class or for open detection"""
        if self.task_config == "fixed":
            return (
                f"""This is aerial image, detect {class_name} in the image and return bounding boxes in JSON format and no other outputs."""
                """e.g.[
                    {"bbox_2d": [x1, y1, x2, y2], "label": "object_type"},   
                    {"bbox_2d": [x1, y1, x2, y2], "label": "object_type"}, ]"""
            )
        elif self.task_config == "open":
            return (
                f"""This is aerial image, detect {class_name} in the image and return bounding boxes in JSON format and no other outputs."""
                """e.g.[
                    {"bbox_2d": [x1, y1, x2, y2], "label": "object_type"},   
                    {"bbox_2d": [x1, y1, x2, y2], "label": "object_type"}, ]"""
            )
        elif self.task_config == "open_ended":
            if not instruction_data:
                raise ValueError("instruction_data must be provided for open_ended task")
            
            instruction = instruction_data.get('instruction', '')
            objects = instruction_data.get('objects', [])
            dataset_type = instruction_data.get('dataset_type', '')
            if self.detector.config["model_name"] == "Qwen/Qwen2.5-VL-7B-Instruct":
                return f"""
                Mission Instruction: {instruction}            
                Analyze this aerial image and detect all objects relevant to the mission instruction.
                Return the bounding boxes in JSON format with labels and no other outputs. 
                Example format:
                [
                    {{"bbox_2d": [x1, y1, x2, y2], "label": "object_type"}},
                    {{"bbox_2d": [x1, y1, x2, y2], "label": "object_type"}}
                ]
                """
            else:
                return f"""
                Mission Instruction: {instruction}            
                Analyze this aerial image and detect all objects relevant to the mission instruction.
                Return the bounding boxes in JSON format with labels and no other outputs.
                Example format:
                [
                    {{"bbox_2d": [x1, y1, x2, y2], "label": "object_type"}},
                    {{"bbox_2d": [x1, y1, x2, y2], "label": "object_type"}}
                ]
                """
    
    def process_image(self, image_path: str) -> dict:
        """
        Process a single image and detect all classes
        
        :param image_path: Path to the image file
        :return: Dictionary containing all detection results for the image
        """
        results = {
            "image_path": image_path, 
            "detections": {}, 
            "task_config": self.task_config
        }
        
        if self.task_config == "fixed":
            for class_name in self.class_list:
                prompt = self.generate_prompt(class_name)
                detection_result = self.detector.detect(image_path, prompt)
                
                if "bboxes" in detection_result:
                    results["detections"][class_name] = detection_result["bboxes"]
                else:
                    results["detections"][class_name] = []
                    print(f"Warning: Failed to detect {class_name} in {image_path}")
                    
        elif self.task_config == "open":
            for class_name in self.class_list:
                prompt = self.generate_prompt(class_name)
                detection_result = self.detector.detect(image_path, prompt)
                
                if "bboxes" in detection_result:
                    results["detections"][class_name] = detection_result["bboxes"]
                else:
                    results["detections"][class_name] = []
                    print(f"Warning: Failed to detect {class_name} in {image_path}")
                    
        elif self.task_config == "open_ended":
            # Get instruction data for this image
            instruction_data = self.instruction_map.get(image_path)
            if not instruction_data:
                print(f"Warning: No instruction found for image {image_path}")
                return results
                
            # Add instruction metadata to results
            results.update({
                "instruction": instruction_data.get('instruction'),
                "dataset_type": instruction_data.get('dataset_type'),
                "gt_object": instruction_data.get('objects')
            })

            # Generate and run prompt
            prompt = self.generate_prompt(instruction_data=instruction_data)
            detection_result = self.detector.detect(image_path, prompt)
            
            if "bboxes" in detection_result:
                results["detections"] = detection_result["bboxes"]
                if "object" in detection_result:
                    results["object"] = detection_result["object"]
            else:
                results["detections"] = []
                print(f"Warning: Failed to detect objects in {image_path}")
            
        print(results)
        return results
    
    def save_results(self, image_path: str, results: dict):
        """
        Save detection results to JSON file
        
        :param image_path: Original image path
        :param results: Detection results dictionary
        """
        # Get image name without extension
        image_name = Path(image_path).stem
        output_path = os.path.join(self.output_dir, f"{image_name}.json")
        
        # Custom JSON formatting function
        def format_json(data, indent=2):
            if isinstance(data, list) and all(isinstance(x, list) and len(x) == 4 for x in data):
                # If it's a bbox coordinate list, output compactly
                return json.dumps(data, separators=(',', ':'))
            elif isinstance(data, dict):
                # If it's a dictionary, process recursively
                return "{\n" + ",\n".join(
                    f'{" "*indent}"{k}": {format_json(v, indent+2)}'
                    for k, v in data.items()
                ) + "\n" + " "*(indent-2) + "}"
            else:
                # Normal output for other cases
                return json.dumps(data)
        
        with open(output_path, 'w') as f:
            formatted_json = format_json(results)
            f.write(formatted_json)
    
    def run_evaluation(self):
        """Run evaluation on all images in the list"""
        # Read image list
        with open(self.image_list_path) as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        # Process each image
        for image_path in tqdm(image_paths, desc="Processing images"):
            if not os.path.exists(image_path):
                print(f"Warning: Image not found - {image_path}")
                continue
            
            try:
                with self.detector:
                    results = self.process_image(image_path)
                    self.save_results(image_path, results)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

if __name__ == "__main__":
    # Configuration
    IMAGE_LIST = "./datasets/VLAD_Remote/test_image_list.txt"  # Path to text file with image paths
    INSTRUCTIONS_PATH = "./tools/dataset_Remote/instructions_with_objects.json"  # Path to instructions JSON file
    
    # Example usage for fixed classes
    fixed_classes = ["Car", "Bus", "Truck", "Building"]
    fixed_evaluator = BBoxEvaluator(IMAGE_LIST, fixed_classes, task_config="fixed")
    fixed_evaluator.run_evaluation()
    
    # Example usage for open classes
    open_evaluator = BBoxEvaluator(IMAGE_LIST, [], task_config="open")
    open_evaluator.run_evaluation()

    open_evaluator = BBoxEvaluator(
        IMAGE_LIST, 
        [], 
        task_config="open_ended",
        instructions_path=INSTRUCTIONS_PATH
    )
    open_evaluator.run_evaluation()