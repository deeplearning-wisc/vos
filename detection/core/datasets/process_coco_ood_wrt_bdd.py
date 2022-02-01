import json
from pycocotools.coco import COCO

data = json.load(open('/nobackup-slow/dataset/my_xfdu/coco2017/annotations/instances_val2017.json'))
new_dict = dict()

new_dict['info'] = data['info']
new_dict['licenses'] = data['licenses']
new_dict['categories'] = data['categories']

images = []
annotations = []
keep_image_ids = []


coco = COCO('/nobackup-slow/dataset/my_xfdu/coco2017/annotations/instances_val2017.json')
# import ipdb; ipdb.set_trace()
# CLASSES = (
#                 'truck', 'traffic light', 'fire hydrant',
#                'stop sign', 'parking meter', 'bench',
#                   'elephant', 'bear', 'zebra', 'giraffe',
#                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#                 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                'hot dog', 'pizza', 'donut', 'cake',
#                'bed',  'toilet',  'laptop',
#                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
#                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
CLASSES = ('airplane',
                        'boat',
                          'fire hydrant', 'parking meter', 'bench',
                          'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                          'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                          'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                          'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                          'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                          'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                          'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                          'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                          'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                          'hair drier', 'toothbrush')
cat_ids = coco.get_cat_ids(cat_names=CLASSES)
cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
img_ids = coco.get_img_ids()
# import ipdb; ipdb.set_trace()


for i in img_ids:
    mark = 0
    info = coco.load_imgs([i])[0]
    # info['filename'] = info['file_name']

    # added part.
    ann_ids = coco.get_ann_ids(img_ids=[info['id']])
    ann_info = coco.load_anns(ann_ids)
    for object1 in ann_info:
        if object1['category_id'] not in cat_ids:
            mark = 1
            continue
    if mark == 0:
        keep_image_ids.append(i)
# import ipdb; ipdb.set_trace()
# for index in keep_image_ids:
#     annotations.append()
for annotations1 in data['annotations']:
    if annotations1['image_id'] in keep_image_ids:
        annotations.append(annotations1)
        # keep_image_ids.append(annotations1['image_id'])

for image_info in data['images']:
    if image_info['id'] in keep_image_ids:
        images.append(image_info)

new_dict['images'] = images
new_dict['annotations'] = annotations

with open('/nobackup-slow/dataset/my_xfdu/coco2017/annotations/instances_val2017_ood_wrt_bdd.json', 'w') as file:
    json.dump(new_dict, file)