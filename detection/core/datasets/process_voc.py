import json

voc2007_train = json.load(open('/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC2007_converted/train_coco_format.json'))
voc2007_test = json.load(open('/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC2007_converted/val_coco_format.json'))

voc2012_train = json.load(open('/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC2012_converted/train_coco_format.json'))

assert voc2007_train['annotations'][-1]['id'] == 15661
index = 0
for instance in voc2012_train['annotations']:
    instance['id'] = 15662 + index
    index += 1

filename = '/nobackup-slow/dataset/my_xfdu/VOCdevkit/VOC_0712_converted/voc0712_train_all.json'
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