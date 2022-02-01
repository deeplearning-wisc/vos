import cv2
import os

from abc import ABC, abstractmethod

# Detectron Imports
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from core.visualization_tools.probabilistic_visualizer import ProbabilisticVisualizer
from detectron2.data import MetadataCatalog

# Project Imports
from inference import inference_utils
import numpy as np
import torch


class ProbabilisticPredictor(ABC):
    """
    Abstract class for probabilistic predictor.
    """

    def __init__(self, cfg):
        # Create common attributes.
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model_list = []

        # Parse config
        self.inference_mode = self.cfg.PROBABILISTIC_INFERENCE.INFERENCE_MODE
        self.mc_dropout_enabled = self.cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.ENABLE
        self.num_mc_dropout_runs = self.cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS

        # Set model to train for MC-Dropout runs
        if self.mc_dropout_enabled:
            self.model.train()
        else:
            self.model.eval()

        # Create ensemble if applicable.
        if self.inference_mode == 'ensembles':
            ensemble_random_seeds = self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.RANDOM_SEED_NUMS

            for i, random_seed in enumerate(ensemble_random_seeds):
                model = build_model(self.cfg)
                model.eval()

                checkpoint_dir = os.path.join(
                    os.path.split(
                        self.cfg.OUTPUT_DIR)[0],
                    'random_seed_' +
                    str(random_seed))
                # Load last checkpoint.
                DetectionCheckpointer(
                    model,
                    save_dir=checkpoint_dir).resume_or_load(
                    cfg.MODEL.WEIGHTS,
                    resume=True)
                self.model_list.append(model)
        else:
            # Or Load single model last checkpoint.
            DetectionCheckpointer(
                self.model,
                save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS,
                resume=True)

    def __call__(self, input_im):
        # Generate detector output.
        if self.inference_mode == 'standard_nms':
            results = self.post_processing_standard_nms(input_im)
        elif self.inference_mode == 'mc_dropout_ensembles':
            results = self.post_processing_mc_dropout_ensembles(
                input_im)
        elif self.inference_mode == 'output_statistics':
            results = self.post_processing_output_statistics(
                input_im)
        elif self.inference_mode == 'ensembles':
            results = self.post_processing_ensembles(input_im, self.model_list)
        elif self.inference_mode == 'bayes_od':
            results = self.post_processing_bayes_od(input_im)
        else:
            raise ValueError(
                'Invalid inference mode {}.'.format(
                    self.inference_mode))

        # Perform post processing on detector output.
        height = input_im[0].get("height", results.image_size[0])
        width = input_im[0].get("width", results.image_size[1])
        results = inference_utils.probabilistic_detector_postprocess(results,
                                                                     height,
                                                                     width)
        return results

    def visualize_inference(self, inputs, results, savedir, name, cfg, energy_threshold=None):
        """
        A function used to visualize final network predictions.
        It shows the original image and up to 20
        predicted object bounding boxes on the original image.

        Valuable for debugging inference methods.

        Args:
            inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        max_boxes = 20

        required_width = inputs[0]['width']
        required_height = inputs[0]['height']

        img = inputs[0]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if self.model.input_format == "RGB":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, (required_width, required_height))

        predicted_boxes = results.pred_boxes.tensor.cpu().numpy()
        predicted_covar_mats = results.pred_boxes_covariance.cpu().numpy()

        v_pred = ProbabilisticVisualizer(img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        # print(len(predicted_boxes))
        labels = results.det_labels[0:max_boxes]
        scores = results.scores[0:max_boxes]
        # breakpoint()

        inter_feat = results.inter_feat[0:max_boxes]
        if energy_threshold:
            labels[(np.argwhere(
                torch.logsumexp(inter_feat[:, :-1], dim=1).cpu().data.numpy() < energy_threshold)).reshape(-1)] = 10
        # # if name == '133631':
        #     # breakpoint()
        # # breakpoint()
        if len(scores) == 0 or max(scores) <= 0.5:
            return

        v_pred = v_pred.overlay_covariance_instances(
            labels=labels,
            scores=scores,
            boxes=predicted_boxes[0:max_boxes],
            score_threshold = 0.5)
            # covariance_matrices=predicted_covar_mats[0:max_boxes])

        prop_img = v_pred.get_image()
        vis_name = f"{max_boxes} Highest Scoring Results"
        # cv2.imshow(vis_name, prop_img)
        # cv2.savefig
        cv2.imwrite(savedir + '/' + name + '.jpg', prop_img)
        cv2.waitKey()

    @abstractmethod
    def post_processing_standard_nms(self, input_im):
        pass

    @abstractmethod
    def post_processing_output_statistics(self, input_im):
        pass

    @abstractmethod
    def post_processing_mc_dropout_ensembles(self, input_im):
        pass

    @abstractmethod
    def post_processing_ensembles(self, input_im, model_list):
        pass

    @abstractmethod
    def post_processing_bayes_od(self, input_im):
        pass