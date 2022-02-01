import argparse
import cv2
import json
import numpy as np
import os

from pascal_voc_tools import XmlParser


def create_coco_lists(ids_list, image_dir, annotations_dir, category_mapper):
    """
    Creates lists in coco format to be written to JSON file.
    """
    parser = XmlParser()

    images_list = []
    annotations_list = []
    count = 0#15661

    for image_id in ids_list:

        image = cv2.imread(os.path.join(image_dir, image_id) + '.jpg')
        images_list.append({'id': image_id,
                            'width': image.shape[1],
                            'height': image.shape[0],
                            'file_name': image_id + '.jpg',
                            'license': 1})

        gt_frame = parser.load(
            os.path.join(
                annotations_dir,
                image_id) + '.xml')
        object_list = gt_frame['object']
        category_names = [object_inst['name'] for object_inst in object_list]
        # import ipdb; ipdb.set_trace()
        # Convert British nouns used in PascalVOC to American nouns used in
        # COCO
        category_names = ['dining table' if category_name ==
                          'diningtable' else category_name for category_name in category_names]
        category_names = ['motorcycle' if category_name ==
                          'motorbike' else category_name for category_name in category_names]
        category_names = ['potted plant' if category_name ==
                          'pottedplant' else category_name for category_name in category_names]
        category_names = ['airplane' if category_name ==
                          'aeroplane' else category_name for category_name in category_names]
        category_names = ['tv' if category_name ==
                          'tvmonitor' else category_name for category_name in category_names]
        category_names = ['couch' if category_name ==
                          'sofa' else category_name for category_name in category_names]

        frame_boxes = np.array(
            [
                [
                    object_inst['bndbox']['xmin'],
                    object_inst['bndbox']['ymin'],
                    object_inst['bndbox']['xmax'],
                    object_inst['bndbox']['ymax']] for object_inst in object_list]).astype(
            np.float)
        # import ipdb;
        # ipdb.set_trace()
        for bbox, category_name in zip(frame_boxes, category_names):
            bbox_coco = [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1]]

            annotations_list.append({'image_id': image_id,
                                     'id': count,
                                     'category_id': category_mapper[category_name],
                                     'bbox': bbox_coco,
                                     'area': bbox_coco[2] * bbox_coco[3],
                                     'iscrowd': 0})
            count += 1

    return images_list, annotations_list


def main(args):
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    dataset_dir = args.dataset_dir

    image_dir = os.path.expanduser(os.path.join(dataset_dir, 'JPEGImages'))
    annotations_dir = os.path.expanduser(
        os.path.join(dataset_dir, 'Annotations'))

    train_ids_file = os.path.expanduser(
        os.path.join(
            dataset_dir,
            'ImageSets',
            'Main',
            'trainval') + '.txt')
    val_ids_file = os.path.expanduser(
        os.path.join(
            dataset_dir,
            'ImageSets',
            'Main',
            'test') + '.txt')

    if args.output_dir is None:
        output_dir = os.path.expanduser(
            os.path.join(dataset_dir, 'COCO-Format'))
    else:
        output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    licenses = [{'id': 1,
                 'name': 'none',
                 'url': 'none'}]

    categories = [{'id': 1, 'name': 'person', 'supercategory': 'person'},
                  {'id': 2, 'name': 'bird', 'supercategory': 'animal'},
                  {'id': 3, 'name': 'cat', 'supercategory': 'animal'},
                  {'id': 4, 'name': 'cow', 'supercategory': 'animal'},
                  {'id': 5, 'name': 'dog', 'supercategory': 'animal'},
                  {'id': 6, 'name': 'horse', 'supercategory': 'animal'},
                  {'id': 7, 'name': 'sheep', 'supercategory': 'animal'},
                  {'id': 8, 'name': 'airplane', 'supercategory': 'vehicle'},
                  {'id': 9, 'name': 'bicycle', 'supercategory': 'vehicle'},
                  {'id': 10, 'name': 'boat', 'supercategory': 'vehicle'},
                  {'id': 11, 'name': 'bus', 'supercategory': 'vehicle'},
                  {'id': 12, 'name': 'car', 'supercategory': 'vehicle'},
                  {'id': 13, 'name': 'motorcycle', 'supercategory': 'vehicle'},
                  {'id': 14, 'name': 'train', 'supercategory': 'vehicle'},
                  {'id': 15, 'name': 'bottle', 'supercategory': 'indoor'},
                  {'id': 16, 'name': 'chair', 'supercategory': 'indoor'},
                  {'id': 17, 'name': 'dining table', 'supercategory': 'indoor'},
                  {'id': 18, 'name': 'potted plant', 'supercategory': 'indoor'},
                  {'id': 19, 'name': 'couch', 'supercategory': 'indoor'},
                  {'id': 20, 'name': 'tv', 'supercategory': 'indoor'},
                  ]

    category_mapper = {}
    category_keys = [category['name'] for category in categories]

    for category_name, category in zip(category_keys, categories):
        category_mapper[category_name] = category['id']

    # Process Training Labels
    with open(train_ids_file, 'r') as f:
        train_ids_list = [line for line in f.read().splitlines()]

    training_image_list, training_annotation_list = create_coco_lists(
        train_ids_list, image_dir, annotations_dir, category_mapper)

    json_dict_training = {'info': {'year': 2020},
                          'licenses': licenses,
                          'categories': categories,
                          'images': training_image_list,
                          'annotations': training_annotation_list}

    training_file_name = os.path.join(output_dir, 'train_coco_format.json')

    with open(training_file_name, 'w') as outfile:
        json.dump(json_dict_training, outfile)

    print("Finished processing PascalVOC training data!")

    # # Process Validation Labels
    # with open(val_ids_file, 'r') as f:
    #     val_ids_list = [line for line in f.read().splitlines()]
    #
    # validation_image_list, validation_annotation_list = create_coco_lists(
    #     val_ids_list, image_dir, annotations_dir, category_mapper)
    #
    # json_dict_validation = {'info': {'year': 2020},
    #                         'licenses': licenses,
    #                         'categories': categories,
    #                         'images': validation_image_list,
    #                         'annotations': validation_annotation_list}
    #
    # validation_file_name = os.path.join(output_dir, 'val_coco_format.json')
    # with open(validation_file_name, 'w') as outfile:
    #     json.dump(json_dict_validation, outfile)
    #
    # print("Converted PascalVOC to COCO format!")


if __name__ == "__main__":
    # Create arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        # required=True,
        type=str,
        default='/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC2012/',
        help='bdd100k dataset directory')

    parser.add_argument(
        "--output-dir",
        # required=False,
        type=str,
        default='/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC2012_converted/',
        help='converted dataset write directory')

    args = parser.parse_args()
    main(args)
