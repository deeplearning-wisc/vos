import numpy as np
import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Coco evaluator tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Project imports
from core.setup import setup_config, setup_arg_parser
from inference.inference_utils import get_inference_output_dir


def main(args, cfg=None):
    # Setup config
    if cfg is None:
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    # Build path to inference output
    inference_output_dir = get_inference_output_dir(
        cfg['OUTPUT_DIR'],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level)
    # inference_output_dir = '/nobackup-slow/dataset/my_xfdu'
    prediction_file_name = os.path.join(
        inference_output_dir,
        'coco_instances_results.json')

    meta_catalog = MetadataCatalog.get(args.test_dataset)

    # Evaluate detection results
    gt_coco_api = COCO(meta_catalog.json_file)
    res_coco_api = gt_coco_api.loadRes(prediction_file_name)
    results_api = COCOeval(gt_coco_api, res_coco_api, iouType='bbox')

    results_api.params.catIds = list(
        meta_catalog.thing_dataset_id_to_contiguous_id.keys())

    # Calculate and print aggregate results
    results_api.evaluate()
    results_api.accumulate()
    results_api.summarize()

    # Compute optimal micro F1 score threshold. We compute the f1 score for
    # every class and score threshold. We then compute the score threshold that
    # maximizes the F-1 score of every class. The final score threshold is the average
    # over all classes.
    precisions = results_api.eval['precision'].mean(0)[:, :, 0, 2]
    recalls = np.expand_dims(results_api.params.recThrs, 1)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_f1_score = f1_scores.argmax(0)
    scores = results_api.eval['scores'].mean(0)[:, :, 0, 2]
    optimal_score_threshold = [scores[optimal_f1_score_i, i]
                               for i, optimal_f1_score_i in enumerate(optimal_f1_score)]
    optimal_score_threshold = np.array(optimal_score_threshold)
    optimal_score_threshold = optimal_score_threshold[optimal_score_threshold != 0]
    optimal_score_threshold = optimal_score_threshold.mean()

    print("Classification Score at Optimal F-1 Score: {}".format(optimal_score_threshold))

    text_file_name = os.path.join(
        inference_output_dir,
        'mAP_res.txt')
    # optimal_score_threshold = 0.0
    with open(text_file_name, "w") as text_file:
        print(results_api.stats.tolist() +
              [optimal_score_threshold, ], file=text_file)


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
