import os
import shutil
from pathlib import Path
from glob import glob
import random
from definitions import ROOT_DIR, TRAIN_IMAGE_FORMAT


class DatasetManager:
    def __init__(self, train_size=0.7, val_size=0.2, test_size=0.1, image_format=TRAIN_IMAGE_FORMAT):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.image_format = image_format
        self.data_path = Path(ROOT_DIR).parent / 'data'
        self.train_path = Path(ROOT_DIR).parent / 'train'
        self.val_path = Path(ROOT_DIR).parent / 'valid'
        self.test_path = Path(ROOT_DIR).parent / 'test'
        
        for path in [self.train_path, self.val_path, self.test_path]:
            os.makedirs(path / 'images', exist_ok=True)
            os.makedirs(path / 'labels', exist_ok=True)
    
    def get_labeled_image_paths(self):
        """Get paths of all images that have corresponding label files"""
        label_root = str(self.data_path / 'labels')
        image_root = str(self.data_path / 'images')
        
        label_files = [Path(elem).stem for elem in glob(label_root + '/*.txt')]
        image_files = [Path(elem).stem for elem in glob(image_root + f'/*.{self.image_format}')]
        
        # Find images that have labels
        labeled_images = list(set(image_files).intersection(set(label_files)))
        return labeled_images
    
    def split_dataset(self):
        """Split the dataset into train, validation and test sets"""
        labeled_images = self.get_labeled_image_paths()
        random.shuffle(labeled_images)
        
        train_count = int(len(labeled_images) * self.train_size)
        val_count = int(len(labeled_images) * self.val_size)
        
        train_images = labeled_images[:train_count]
        val_images = labeled_images[train_count:train_count + val_count]
        test_images = labeled_images[train_count + val_count:]
        
        return {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
    
    def organize_dataset(self):
        """Organize the dataset into train, validation and test directories"""
        split_data = self.split_dataset()
        
        # Clear existing files in train/val/test directories
        for path in [self.train_path, self.val_path, self.test_path]:
            for file in glob(str(path / 'images' / '*')):
                os.remove(file)
            for file in glob(str(path / 'labels' / '*')):
                os.remove(file)
        
        # Copy files to their respective directories
        for split_name, image_list in split_data.items():
            target_path = getattr(self, f"{split_name}_path")
            
            for image_name in image_list:
                # Copy image
                src_img = self.data_path / 'images' / f"{image_name}.{self.image_format}"
                dst_img = target_path / 'images' / f"{image_name}.{self.image_format}"
                if src_img.exists():
                    shutil.copy(str(src_img), str(dst_img))

                src_label = self.data_path / 'labels' / f"{image_name}.txt"
                dst_label = target_path / 'labels' / f"{image_name}.txt"
                if src_label.exists():
                    shutil.copy(str(src_label), str(dst_label))
        
        # Return statistics
        return {
            'train': len(split_data['train']),
            'val': len(split_data['val']),
            'test': len(split_data['test'])
        }
    
    def generate_data_yaml(self):
        """Generate data.yaml file for YOLOv10 training"""
        yaml_content = f"""train: ../train/images
                            val: ../valid/images
                            test: ../test/images

                            nc: 3
                            names: ['male_cat', 'female_cat', 'other']
                            """
        yaml_path = self.data_path / 'data.yaml'
        with open(str(yaml_path), 'w') as f:
            f.write(yaml_content)
        return str(yaml_path) 