import argparse
import csv
import cv2
import json
import os

from tqdm import tqdm


def main(args):
    dataset_dir = args.dataset_dir

    if args.output_dir is None:
        output_dir = os.path.expanduser(
            os.path.join(dataset_dir, 'COCO-Format'))
    else:
        output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Get category mapping from openimages symbol to openimages names.
    with open(os.path.expanduser(os.path.join(dataset_dir, 'class-descriptions-boxable.csv')), 'r', encoding='utf-8') as f:
        csv_f = csv.reader(f)
        openimages_class_mapping_dict = dict()
        for row in csv_f:
            openimages_class_mapping_dict.update({row[0]: row[1]})

    # Get annotation csv path and image directories
    annotations_csv_path = os.path.expanduser(
        os.path.join(dataset_dir, 'train-annotations-bbox.csv'))
    image_dir = os.path.expanduser(os.path.join(dataset_dir, 'images'))
    id_list = [image[:-4] for image in os.listdir(image_dir)]

    # Begin processing annotations
    with open(annotations_csv_path, 'r', encoding='utf-8') as f:
        csv_f = csv.reader(f)

        processed_ids = []
        images_list = []
        annotations_list = []

        with tqdm(total=len(id_list)) as pbar:
            for i, row in enumerate(csv_f):
                image_id = row[0]
                if image_id in id_list:
                    image = cv2.imread(
                        os.path.join(
                            image_dir,
                            image_id) + '.jpg')
                    width = image.shape[1]
                    height = image.shape[0]

                    if image_id not in processed_ids:
                        pbar.update(1)
                        images_list.append({'id': image_id,
                                            'width': width,
                                            'height': height,
                                            'file_name': image_id + '.jpg',
                                            'license': 1})
                        processed_ids.append(image_id)
                else:
                    continue

    licenses = [{'id': 1,
                 'name': 'none',
                 'url': 'none'}]

    categories = [
        {"supercategory": "person", "id": 1, "name": "person"},
        {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
        {"supercategory": "vehicle", "id": 3, "name": "car"},
        {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
        {"supercategory": "vehicle", "id": 5, "name": "airplane"},
        {"supercategory": "vehicle", "id": 6, "name": "bus"},
        {"supercategory": "vehicle", "id": 7, "name": "train"},
        {"supercategory": "vehicle", "id": 8, "name": "truck"},
        {"supercategory": "vehicle", "id": 9, "name": "boat"},
        {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
        {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
        {"supercategory": "outdoor", "id": 12, "name": "stop sign"},
        {"supercategory": "outdoor", "id": 13, "name": "parking meter"},
        {"supercategory": "outdoor", "id": 14, "name": "bench"},
        {"supercategory": "animal", "id": 15, "name": "bird"},
        {"supercategory": "animal", "id": 16, "name": "cat"},
        {"supercategory": "animal", "id": 17, "name": "dog"},
        {"supercategory": "animal", "id": 18, "name": "horse"},
        {"supercategory": "animal", "id": 19, "name": "sheep"},
        {"supercategory": "animal", "id": 20, "name": "cow"},
        {"supercategory": "animal", "id": 21, "name": "elephant"},
        {"supercategory": "animal", "id": 22, "name": "bear"},
        {"supercategory": "animal", "id": 23, "name": "zebra"},
        {"supercategory": "animal", "id": 24, "name": "giraffe"},
        {"supercategory": "accessory", "id": 25, "name": "backpack"},
        {"supercategory": "accessory", "id": 26, "name": "umbrella"},
        {"supercategory": "accessory", "id": 27, "name": "handbag"},
        {"supercategory": "accessory", "id": 28, "name": "tie"},
        {"supercategory": "accessory", "id": 29, "name": "suitcase"},
        {"supercategory": "sports", "id": 30, "name": "frisbee"},
        {"supercategory": "sports", "id": 31, "name": "skis"},
        {"supercategory": "sports", "id": 32, "name": "snowboard"},
        {"supercategory": "sports", "id": 33, "name": "sports ball"},
        {"supercategory": "sports", "id": 34, "name": "kite"},
        {"supercategory": "sports", "id": 35, "name": "baseball bat"},
        {"supercategory": "sports", "id": 36, "name": "baseball glove"},
        {"supercategory": "sports", "id": 37, "name": "skateboard"},
        {"supercategory": "sports", "id": 38, "name": "surfboard"},
        {"supercategory": "sports", "id": 39, "name": "tennis racket"},
        {"supercategory": "kitchen", "id": 40, "name": "bottle"},
        {"supercategory": "kitchen", "id": 41, "name": "wine glass"},
        {"supercategory": "kitchen", "id": 42, "name": "cup"},
        {"supercategory": "kitchen", "id": 43, "name": "fork"},
        {"supercategory": "kitchen", "id": 44, "name": "knife"},
        {"supercategory": "kitchen", "id": 45, "name": "spoon"},
        {"supercategory": "kitchen", "id": 46, "name": "bowl"},
        {"supercategory": "food", "id": 47, "name": "banana"},
        {"supercategory": "food", "id": 48, "name": "apple"},
        {"supercategory": "food", "id": 49, "name": "sandwich"},
        {"supercategory": "food", "id": 50, "name": "orange"},
        {"supercategory": "food", "id": 51, "name": "broccoli"},
        {"supercategory": "food", "id": 52, "name": "carrot"},
        {"supercategory": "food", "id": 53, "name": "hot dog"},
        {"supercategory": "food", "id": 54, "name": "pizza"},
        {"supercategory": "food", "id": 55, "name": "donut"},
        {"supercategory": "food", "id": 56, "name": "cake"},
        {"supercategory": "furniture", "id": 57, "name": "chair"},
        {"supercategory": "furniture", "id": 58, "name": "couch"},
        {"supercategory": "furniture", "id": 59, "name": "potted plant"},
        {"supercategory": "furniture", "id": 60, "name": "bed"},
        {"supercategory": "furniture", "id": 61, "name": "dining table"},
        {"supercategory": "furniture", "id": 62, "name": "toilet"},
        {"supercategory": "electronic", "id": 63, "name": "tv"},
        {"supercategory": "electronic", "id": 64, "name": "laptop"},
        {"supercategory": "electronic", "id": 65, "name": "mouse"},
        {"supercategory": "electronic", "id": 66, "name": "remote"},
        {"supercategory": "electronic", "id": 67, "name": "keyboard"},
        {"supercategory": "electronic", "id": 68, "name": "cell phone"},
        {"supercategory": "appliance", "id": 69, "name": "microwave"},
        {"supercategory": "appliance", "id": 70, "name": "oven"},
        {"supercategory": "appliance", "id": 71, "name": "toaster"},
        {"supercategory": "appliance", "id": 72, "name": "sink"},
        {"supercategory": "appliance", "id": 73, "name": "refrigerator"},
        {"supercategory": "indoor", "id": 74, "name": "book"},
        {"supercategory": "indoor", "id": 75, "name": "clock"},
        {"supercategory": "indoor", "id": 76, "name": "vase"},
        {"supercategory": "indoor", "id": 77, "name": "scissors"},
        {"supercategory": "indoor", "id": 78, "name": "teddy bear"},
        {"supercategory": "indoor", "id": 79, "name": "hair drier"},
        {"supercategory": "indoor", "id": 80, "name": "toothbrush"}]

    json_dict_val = {'info': {'year': 2020},
                     'licenses': licenses,
                     'categories': categories,
                     'images': images_list,
                     'annotations': annotations_list}

    val_file_name = os.path.join(output_dir, 'val_coco_format.json')
    with open(val_file_name, 'w') as outfile:
        json.dump(json_dict_val, outfile)


if __name__ == "__main__":
    # Create arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=str,
        help='bdd100k dataset directory')

    parser.add_argument(
        "--output-dir",
        required=False,
        type=str,
        help='converted dataset write directory')

    args = parser.parse_args()
    main(args)
