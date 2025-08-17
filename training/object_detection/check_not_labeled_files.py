from glob import glob
from pathlib import Path
from definitions import ROOT_DIR

if __name__ == "__main__":
    label_root = str(Path(ROOT_DIR).parent / 'data' / 'labels')
    image_root = str(Path(ROOT_DIR).parent / 'data' / 'images')
    label_pathes = [Path(elem).name.replace('.txt', '') for elem in glob(label_root + '/*.txt')]
    image_pathes = [Path(elem).name.replace('.png', '') for elem in glob(image_root + '/*.png')]
    print(f"\nFound {len(image_pathes)} images and {len(label_pathes)} labels. Comparing...")
    print(f"\nThere are {len(set(image_pathes) - set(label_pathes))} unique images and {len(set(label_pathes) - set(image_pathes))} unique labels")
    print(f"\nExamples: {image_pathes[0]}, {label_pathes[0]}")
