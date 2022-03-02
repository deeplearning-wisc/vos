import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Project imports
import core.datasets.metadata as metadata


def setup_all_datasets(dataset_dir, image_root_corruption_prefix=None):
    """
    Registers all datasets as instances from COCO

    Args:
        dataset_dir(str): path to dataset directory

    """
    setup_voc_dataset(dataset_dir)
    setup_coco_dataset(
        dataset_dir,
        image_root_corruption_prefix=image_root_corruption_prefix)
    setup_coco_ood_dataset(dataset_dir)
    setup_openim_odd_dataset(dataset_dir)
    setup_bdd_dataset(dataset_dir)
    setup_coco_ood_bdd_dataset(dataset_dir)

def setup_coco_dataset(dataset_dir, image_root_corruption_prefix=None):
    """
    sets up coco dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.
    """
    train_image_dir = os.path.join(dataset_dir, 'train2017')

    if image_root_corruption_prefix is not None:
        test_image_dir = os.path.join(
            dataset_dir, 'val2017' + image_root_corruption_prefix)
    else:
        test_image_dir = os.path.join(dataset_dir, 'val2017')

    train_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_train2017.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017.json')

    register_coco_instances(
        "coco_2017_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "coco_2017_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_openim_dataset(dataset_dir):
    """
    sets up openimages dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.

    Only validation is supported.
    """
    # import ipdb; ipdb.set_trace()
    test_image_dir = os.path.join(dataset_dir, 'images')

    test_json_annotations = os.path.join(
        dataset_dir, 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openimages_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_val").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_openim_odd_dataset(dataset_dir):
    """
    sets up openimages out-of-distribution dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.

    Only validation is supported.
    """
    test_image_dir = os.path.join(dataset_dir + 'ood_classes_rm_overlap', 'images')

    test_json_annotations = os.path.join(
        dataset_dir + 'ood_classes_rm_overlap', 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openimages_ood_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_ood_val").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID



def setup_voc_id_dataset(dataset_dir):
    train_image_dir = os.path.join(dataset_dir, 'JPEGImages')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')

    train_json_annotations = os.path.join(
        dataset_dir, 'voc0712_train_all.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_coco_format.json')

    register_coco_instances(
        "voc_custom_train_id",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "voc_custom_train_id").thing_classes = metadata.VOC_ID_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_train_id").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain

    register_coco_instances(
        "voc_custom_val_id",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_custom_val_id").thing_classes = metadata.VOC_ID_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_val_id").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain



def setup_bdd_dataset(dataset_dir):
    train_image_dir = os.path.join(dataset_dir, 'images/100k/train')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'images/100k/val')

    train_json_annotations = os.path.join(
        dataset_dir, 'train_bdd_converted.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_bdd_converted.json')

    register_coco_instances(
        "bdd_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "bdd_custom_train").thing_classes = metadata.BDD_THING_CLASSES
    MetadataCatalog.get(
        "bdd_custom_train").thing_dataset_id_to_contiguous_id = metadata.BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "bdd_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "bdd_custom_val").thing_classes = metadata.BDD_THING_CLASSES
    MetadataCatalog.get(
        "bdd_custom_val").thing_dataset_id_to_contiguous_id = metadata.BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_voc_dataset(dataset_dir):
    train_image_dir = os.path.join(dataset_dir, 'JPEGImages')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')

    train_json_annotations = os.path.join(
        dataset_dir, 'voc0712_train_all.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_coco_format.json')

    register_coco_instances(
        "voc_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "voc_custom_train").thing_classes = metadata.VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_train").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "voc_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_custom_val").thing_classes = metadata.VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_val").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_voc_ood_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')

    test_json_annotations = os.path.join(
        dataset_dir, 'val_coco_format.json')

    register_coco_instances(
        "voc_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_ood_val").thing_classes = metadata.VOC_OOD_THING_CLASSES
    MetadataCatalog.get(
        "voc_ood_val").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain


def setup_coco_ood_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'val2017')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017_ood_rm_overlap.json')

    register_coco_instances(
        "coco_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "coco_ood_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_ood_val").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_coco_ood_bdd_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'val2017')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017_ood_wrt_bdd_rm_overlap.json')

    register_coco_instances(
        "coco_ood_val_bdd",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "coco_ood_val_bdd").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_ood_val_bdd").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_coco_ood_train_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'train2017')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_train2017_ood.json')

    register_coco_instances(
        "coco_ood_train",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "coco_ood_train").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_ood_train").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_openimages_ood_oe_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'images')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_ood_oe",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "openimages_ood_oe").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_ood_oe").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID