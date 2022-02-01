import json
import os

voc2007_train = json.load(open('/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC2007_converted_ID/train_coco_format.json'))
voc2007_test = json.load(open('/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC2007_converted_ID/val_coco_format.json'))

voc2012_train = json.load(open('/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC2012_converted_ID/train_coco_format.json'))


index = 0
for instance in voc2012_train['annotations']:
    instance['id'] = len(voc2007_train['annotations']) + index
    index += 1

if not os.path.exists('/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC_0712_converted_ID'):
    os.makedirs('/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC_0712_converted_ID')
filename = '/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC_0712_converted_ID/voc0712_train_all.json'
with open(filename, 'w') as file:
    new = dict()
    new['info'] = voc2012_train['info']
    new['licenses']=voc2012_train['licenses']
    new['categories']=voc2012_train['categories']
    new['images'] = voc2012_train['images'] + voc2007_train['images']
    new['annotations'] = voc2012_train['annotations'] + voc2007_train['annotations']
    json.dump(new, file)

# filename = '/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC_0712_converted/voc0712_train_id.json'
# id_images = []
# for instance in voc2007_train['annotations']:
#     if instance['category_id']