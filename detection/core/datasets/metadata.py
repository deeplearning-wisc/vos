from collections import ChainMap

# Detectron imports
from detectron2.data import MetadataCatalog

# Useful Dicts for OpenImages Conversion
OPEN_IMAGES_TO_COCO = {'Person': 'person',
                       'Bicycle': 'bicycle',
                       'Car': 'car',
                       'Motorcycle': 'motorcycle',
                       'Airplane': 'airplane',
                       'Bus': 'bus',
                       'Train': 'train',
                       'Truck': 'truck',
                       'Boat': 'boat',
                       'Traffic light': 'traffic light',
                       'Fire hydrant': 'fire hydrant',
                       'Stop sign': 'stop sign',
                       'Parking meter': 'parking meter',
                       'Bench': 'bench',
                       'Bird': 'bird',
                       'Cat': 'cat',
                       'Dog': 'dog',
                       'Horse': 'horse',
                       'Sheep': 'sheep',
                       'Elephant': 'cow',
                       'Cattle': 'elephant',
                       'Bear': 'bear',
                       'Zebra': 'zebra',
                       'Giraffe': 'giraffe',
                       'Backpack': 'backpack',
                       'Umbrella': 'umbrella',
                       'Handbag': 'handbag',
                       'Tie': 'tie',
                       'Suitcase': 'suitcase',
                       'Flying disc': 'frisbee',
                       'Ski': 'skis',
                       'Snowboard': 'snowboard',
                       'Ball': 'sports ball',
                       'Kite': 'kite',
                       'Baseball bat': 'baseball bat',
                       'Baseball glove': 'baseball glove',
                       'Skateboard': 'skateboard',
                       'Surfboard': 'surfboard',
                       'Tennis racket': 'tennis racket',
                       'Bottle': 'bottle',
                       'Wine glass': 'wine glass',
                       'Coffee cup': 'cup',
                       'Fork': 'fork',
                       'Knife': 'knife',
                       'Spoon': 'spoon',
                       'Bowl': 'bowl',
                       'Banana': 'banana',
                       'Apple': 'apple',
                       'Sandwich': 'sandwich',
                       'Orange': 'orange',
                       'Broccoli': 'broccoli',
                       'Carrot': 'carrot',
                       'Hot dog': 'hot dog',
                       'Pizza': 'pizza',
                       'Doughnut': 'donut',
                       'Cake': 'cake',
                       'Chair': 'chair',
                       'Couch': 'couch',
                       'Houseplant': 'potted plant',
                       'Bed': 'bed',
                       'Table': 'dining table',
                       'Toilet': 'toilet',
                       'Television': 'tv',
                       'Laptop': 'laptop',
                       'Computer mouse': 'mouse',
                       'Remote control': 'remote',
                       'Computer keyboard': 'keyboard',
                       'Mobile phone': 'cell phone',
                       'Microwave oven': 'microwave',
                       'Oven': 'oven',
                       'Toaster': 'toaster',
                       'Sink': 'sink',
                       'Refrigerator': 'refrigerator',
                       'Book': 'book',
                       'Clock': 'clock',
                       'Vase': 'vase',
                       'Scissors': 'scissors',
                       'Teddy bear': 'teddy bear',
                       'Hair dryer': 'hair drier',
                       'Toothbrush': 'toothbrush'}

# Construct COCO metadata
COCO_THING_CLASSES = MetadataCatalog.get('coco_2017_train').thing_classes
COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID = MetadataCatalog.get(
    'coco_2017_train').thing_dataset_id_to_contiguous_id

# Construct OpenImages metadata
OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(len(COCO_THING_CLASSES))]))

# MAP COCO to OpenImages contiguous id to be used for inference on OpenImages for models
# trained on COCO.
COCO_TO_OPENIMAGES_CONTIGUOUS_ID = dict(ChainMap(
    *[{COCO_THING_CLASSES.index(openimages_thing_class): COCO_THING_CLASSES.index(openimages_thing_class)} for openimages_thing_class in
      COCO_THING_CLASSES]))
# import ipdb; ipdb.set_trace()

# Construct VOC metadata
VOC_THING_CLASSES = ['person',
                     'bird',
                     'cat',
                     'cow',
                     'dog',
                     'horse',
                     'sheep',
                     'airplane',
                     'bicycle',
                     'boat',
                     'bus',
                     'car',
                     'motorcycle',
                     'train',
                     'bottle',
                     'chair',
                     'dining table',
                     'potted plant',
                     'couch',
                     'tv',
                     ]
VOC_ID_THING_CLASSES = [
'person', 'dog', 'horse', 'sheep', 'motorcycle', 'train', 'dining table', 'potted plant', 'couch', 'tv'
]
VOC_OOD_THING_CLASSES = [
'bird', 'cat', 'cow' , 'airplane', 'bicycle', 'boat', 'bus', 'car', 'bottle', 'chair'
]

# COCO_OOD_THING_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                           'bus', 'train', 'truck', 'boat', 'traffic light',
#                           'fire hydrant', 'stop sign', 'parking meter', 'bench',
#                           'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
#                           'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
#                           'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#                           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#                           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
#                           'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#                           'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#                           'hair drier', 'toothbrush']

# VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
#     ChainMap(*[{i + 1: i} for i in range(len(VOC_THING_CLASSES))]))
# import ipdb; ipdb.set_trace()
VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain = dict(
    ChainMap(*[{i + 1: i} for i in range(10)]))

VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(20)]))

# MAP COCO to VOC contiguous id to be used for inference on VOC for models
# trained on COCO.
COCO_TO_VOC_CONTIGUOUS_ID = dict(ChainMap(
    *[{COCO_THING_CLASSES.index(voc_thing_class): VOC_THING_CLASSES.index(voc_thing_class)} for voc_thing_class in
      VOC_THING_CLASSES]))
# import ipdb; ipdb.set_trace()


BDD_THING_CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']#, "OOD"]



BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(10)]))
