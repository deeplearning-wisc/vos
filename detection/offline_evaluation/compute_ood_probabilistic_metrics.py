import itertools
import os
import torch
import ujson as json
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')
import numpy as np

from prettytable import PrettyTable

# Detectron imports
from detectron2.engine import launch

# Project imports
from core.evaluation_tools import scoring_rules
from core.evaluation_tools.evaluation_utils import eval_predictions_preprocess
from core.setup import setup_config, setup_arg_parser
from inference.inference_utils import get_inference_output_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coco_categories = [
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
MAPPER_coco_open = {}
for instance in coco_categories:
    MAPPER_coco_open[instance['id'] - 1] = instance['name']

voc_all_cate = ['person',
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
MAPPER_voc_coco = {}
for instance in range(len(voc_all_cate)):
    MAPPER_voc_coco[instance] = voc_all_cate[instance]

voc_id_cate = [
'person', 'dog', 'horse', 'sheep', 'motorcycle', 'train', 'dining table', 'potted plant', 'couch', 'tv'
]
MAPPER_split = {}
for instance in range(len(voc_id_cate)):
    MAPPER_split[instance] = voc_id_cate[instance]



def main(
        args,
        cfg=None,
        min_allowed_score=None):

    # Setup config
    if cfg is None:
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    cfg.defrost()
    cfg.ACTUAL_TEST_DATASET = args.test_dataset

    # Setup torch device and num_threads
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Build path to gt instances and inference output
    inference_output_dir = get_inference_output_dir(
        cfg['OUTPUT_DIR'],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level)

    if min_allowed_score is None:
        # Check if F-1 Score has been previously computed ON THE ORIGINAL
        # DATASET, and not on VOC.
        try:
            train_set_inference_output_dir = get_inference_output_dir(
                cfg['OUTPUT_DIR'],
                cfg.DATASETS.TEST[0],
                args.inference_config,
                0)
            # import ipdb; ipdb.set_trace()
            with open(os.path.join(train_set_inference_output_dir, "mAP_res.txt"), "r") as f:
                min_allowed_score = f.read().strip('][\n').split(', ')[-1]
                min_allowed_score = round(float(min_allowed_score), 4)
        except FileNotFoundError:
            # If not, process all detections. Not recommended as the results might be influenced by very low scoring
            # detections that would normally be removed in robotics/vision
            # applications.
            min_allowed_score = 0.0

    # Get matched results by either generating them or loading from file.
    with torch.no_grad():
        try:
            preprocessed_predicted_instances = torch.load(
                os.path.join(
                    inference_output_dir,
                    "preprocessed_predicted_instances_odd_{}.pth".format(min_allowed_score)),
                map_location=device)
        # Process predictions
        except FileNotFoundError:

            prediction_file_name = os.path.join(
                inference_output_dir,
                'coco_instances_results.json')
            predicted_instances = json.load(open(prediction_file_name, 'r'))
            preprocessed_predicted_instances = eval_predictions_preprocess(
                predicted_instances, min_allowed_score=min_allowed_score, is_odd=True)
            torch.save(
                preprocessed_predicted_instances,
                os.path.join(
                    inference_output_dir,
                    "preprocessed_predicted_instances_odd_{}.pth".format(min_allowed_score)))

        predicted_boxes = preprocessed_predicted_instances['predicted_boxes']
        predicted_cov_mats = preprocessed_predicted_instances['predicted_covar_mats']
        predicted_cls_probs = preprocessed_predicted_instances['predicted_cls_probs']
        predicted_inter_feat = preprocessed_predicted_instances['predicted_inter_feat']
        predicted_logistic_score = None
        if 'predicted_logistic_score' in list(preprocessed_predicted_instances.keys()):
            predicted_logistic_score = preprocessed_predicted_instances['predicted_logistic_score']


        predicted_boxes = list(itertools.chain.from_iterable(
            [predicted_boxes[key] for key in predicted_boxes.keys()]))
        predicted_cov_mats = list(itertools.chain.from_iterable(
            [predicted_cov_mats[key] for key in predicted_cov_mats.keys()]))
        predicted_cls_probs = list(itertools.chain.from_iterable(
            [predicted_cls_probs[key] for key in predicted_cls_probs.keys()]))
        predicted_inter_feat = list(itertools.chain.from_iterable(
            [predicted_inter_feat[key] for key in predicted_inter_feat.keys()]))
        if predicted_logistic_score is not None:
            predicted_logistic_score = list(itertools.chain.from_iterable(
            [predicted_logistic_score[key] for key in predicted_logistic_score.keys()]))

        num_false_positives = len(predicted_boxes)
        assert num_false_positives == len(predicted_inter_feat)
        valid_idxs = torch.as_tensor(
            [i for i in range(num_false_positives)]).to(device)

        predicted_boxes = torch.stack(predicted_boxes, 1).transpose(0, 1)
        predicted_cov_mats = torch.stack(predicted_cov_mats, 1).transpose(0, 1)
        predicted_cls_probs = torch.stack(
            predicted_cls_probs,
            1).transpose(
            0,
            1)
        # import ipdb; ipdb.set_trace()

        false_positives_dict = {
            'predicted_box_means': predicted_boxes,
            'predicted_box_covariances': predicted_cov_mats,
            'predicted_cls_probs': predicted_cls_probs}
        # import ipdb; ipdb.set_trace()
        false_positives_reg_analysis = scoring_rules.compute_reg_scores_fn(
            false_positives_dict, valid_idxs, entropy=False)

        if cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticRetinaNet':
            predicted_class_probs, predicted_class_idx = predicted_cls_probs.max(
                1)
            false_positives_dict['predicted_score_of_gt_category'] = 1.0 - \
                predicted_class_probs
            false_positives_cls_analysis = scoring_rules.sigmoid_compute_cls_scores(
                false_positives_dict, valid_idxs)

        else:
            false_positives_dict['predicted_score_of_gt_category'] = predicted_cls_probs[:, -1]
            _, predicted_class_idx = predicted_cls_probs[:, :-1].max(
                1)
            false_positives_cls_analysis = scoring_rules.softmax_compute_cls_scores(
                false_positives_dict, valid_idxs)

        ### for error analsysis.
        # MAPPER = MAPPER_coco_open
        #
        #
        # predicted_class_name = []
        # for index in predicted_class_idx:
        #     predicted_class_name.append(MAPPER[int(index.cpu().data.numpy())])
        # labels, counts = np.unique(predicted_class_name, return_counts=True)
        # ticks = range(len(MAPPER))
        # plot_energy_dict = {}
        # plot_number_dict = {}
        # for key in list(MAPPER.values()):
        #     plot_energy_dict[key] = 0
        #     plot_number_dict[key] = 0
        #
        #
        # # for number only.
        # for index in range(len(labels)):
        #     assert labels[index] in list(plot_number_dict.keys())
        #     plot_number_dict[labels[index]] += counts[index]
        #
        #
        # # for energy only.
        # # breakpoint()
        # for index in range(len(predicted_inter_feat)):
        #     energy_score = 1 * torch.logsumexp(predicted_inter_feat[index] / 1, dim=0).cpu().data.numpy()
        #     # breakpoint()
        #     plot_energy_dict[MAPPER[int(predicted_class_idx[index].cpu().data.numpy())]] += energy_score
        # for key in list(plot_energy_dict.keys()):
        #     if plot_number_dict[key] != 0:
        #         plot_energy_dict[key] /= plot_number_dict[key]
        # ##
        #
        #
        # plt.figure(figsize=(32, 9))
        #
        # # plt.bar(ticks, list(plot_number_dict.values()), align='center')
        # # plt.xticks(ticks, list(plot_number_dict.keys()), fontsize=10, rotation=45)
        #
        # plt.bar(ticks, list(plot_energy_dict.values()), align='center')
        # plt.xticks(ticks, list(plot_energy_dict.keys()), fontsize=10, rotation=45)
        #
        #
        # # from collections import Counter
        # # letter_counts = Counter(predicted_class_name)
        # # plot_bar_from_counter(letter_counts)
        # plt.savefig('coco_open_open_ood_error_analysis.jpg', dpi=250)
        # import ipdb;
        # ipdb.set_trace()
        #######

        # Summarize and print all
        table = PrettyTable()
        table.field_names = (['Output Type',
                              'Number of Instances',
                              'Cls Ignorance Score',
                              'Cls Brier/Probability Score',
                              'Reg Ignorance Score'])
                              # 'Reg Energy Score'])
        table.add_row(
            [
                "False Positives:",
                num_false_positives,
                '{:.4f}'.format(
                    false_positives_cls_analysis['ignorance_score_mean'],),
                '{:.4f}'.format(
                    false_positives_cls_analysis['brier_score_mean']),
                '{:.4f}'.format(
                    false_positives_reg_analysis['total_entropy_mean'])])
                # '{:.4f}'.format(
                #     false_positives_reg_analysis['fp_energy_score_mean'])])
        print(table)

        text_file_name = os.path.join(
            inference_output_dir,
            'probabilistic_scoring_res_odd_{}.txt'.format(min_allowed_score))

        with open(text_file_name, "w") as text_file:
            print(table, file=text_file)

        dictionary_file_name = os.path.join(
            inference_output_dir,
            'probabilistic_scoring_res_odd_{}.pkl'.format(min_allowed_score))
        false_positives_reg_analysis.update(false_positives_cls_analysis)
        # breakpoint()
        # features = []
        # for feature in predicted_inter_feat:
        #     features.append(np.asarray(feature.cpu().data.numpy())[-2:])
        # features = np.asarray(features)
        # predicted_inter_feat = torch.stack(predicted_inter_feat)[:,-1:]
        false_positives_reg_analysis.update({'inter_feat': predicted_inter_feat})
        false_positives_reg_analysis.update({'predicted_cls_id': predicted_class_idx})
        false_positives_reg_analysis.update({'logistic_score': predicted_logistic_score})
        with open(dictionary_file_name, "wb") as pickle_file:
            pickle.dump(false_positives_reg_analysis, pickle_file)


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
