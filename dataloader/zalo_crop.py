import os
import json
from PIL import Image

class ZaloTrafficSignCropper:
    def __init__(self, dataset_dir, output_dir):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.json_path = os.path.join(dataset_dir, "train_traffic_sign_dataset.json")

        with open(self.json_path, "r") as f:
            self.annotations = json.load(f)

    def crop_and_save(self):
        os.makedirs(self.output_dir, exist_ok=True)

        for item in self.annotations["annotations"]:  
            image_filename = f"{str(item['image_id'])}.png"
            label = item["category_id"]  
            bbox = item["bbox"]  

            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height

            image_path = os.path.join(self.images_dir, image_filename)

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue

            cropped_image = image.crop((x_min, y_min, x_max, y_max))

            label_dir = os.path.join(self.output_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)

            output_path = os.path.join(label_dir, f"{os.path.splitext(image_filename)[0]}_{x_min}_{y_min}.png")
            try:
                cropped_image.save(output_path)
                print(f"Saved cropped image to {output_path}")
            except Exception as e:
                print(f"Error saving cropped image {output_path}: {e}")

if __name__ == "__main__":
    dataset_dir = r"D:\USTH_SUBJECTS\B3\Thesis\AdvancedTrafficSignRecognition\datasets\raw\ZAC_Traffic_Sign\traffic_train\traffic_train"
    output_dir = r"D:\USTH_SUBJECTS\B3\Thesis\AdvancedTrafficSignRecognition\datasets\zalo_process"

    cropper = ZaloTrafficSignCropper(dataset_dir, output_dir)
    cropper.crop_and_save()
