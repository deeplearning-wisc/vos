import numpy as np
import matplotlib as mpl
from scipy.stats import norm, chi2

from detectron2.utils.visualizer import Visualizer, ColorMode, _SMALL_OBJECT_AREA_THRESH
from detectron2.utils.colormap import random_color


class ProbabilisticVisualizer(Visualizer):
    """
    Extends detectron2 Visualizer to draw corner covariance matrices.
    """

    def __init__(
            self,
            img_rgb,
            metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale=scale, instance_mode=instance_mode)

    def _create_text_labels(self, classes11, scores, class_names, is_crowd=None):
        """
        Args:
            classes (list[int] or None):
            scores (list[float] or None):
            class_names (list[str] or None):
            is_crowd (list[bool] or None):
        Returns:
            list[str] or None
        """
        labels = None
        if classes11 is not None and class_names is not None and len(class_names) > 0:
            # import ipdb; ipdb.set_trace()
            class_names += ['OOD']
            try:
                labels = [class_names[i] for i in classes11]
            except:
                breakpoint()
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
        if is_crowd is not None:
            labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
        return labels


    def overlay_covariance_instances(
        self,
        *,
        boxes=None,
        covariance_matrices=None,
        labels=None,
        scores=None,
        assigned_colors=None,
        alpha=0.5,
        score_threshold = 0.3
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,

            covariance_matrices (ndarray): numpy array containing the corner covariance matrices
            labels (list[str]): the text to be displayed for each instance.
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.
            alpha: alpha value

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        # import ipdb;
        # ipdb.set_trace()
        labels = self._create_text_labels(labels, scores, self.metadata.get("thing_classes", None))

        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
        
            assigned_colors = [
                random_color(
                    rgb=True,
                    maximum=1) for _ in range(num_instances)]


        if num_instances == 0:
            return self.output

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k]
                      for k in sorted_idxs] if labels is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            scores = [scores[k] for k in sorted_idxs] if scores is not None else None

        
        for i in range(num_instances):
            color = np.array([0.,1.,0.], dtype=np.float32) if 'OOD' in labels[i] else np.array([1.,0.,0], dtype=np.float32)
            # # breakpoint()
            if 'OOD' in labels[i]:
                labels[i] = 'OOD'
            # color = assigned_colors[i]
            # if self.output.get_image() == '66523':
            #     print('hhh')
            #     breakpoint()
            #     # continue
            # breakpoint()
            # if 'pedestrian 87%' in labels[i]:
            #     continue
            if scores[i] < score_threshold:
                continue
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color, alpha=1)
            if covariance_matrices is not None:
                self.draw_ellipse(
                    boxes[i],
                    covariance_matrices[i],
                    edge_color=color,
                    alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    # if drawing boxes, put text on the box corner.
                    text_pos = (x0, y0)
                    horiz_align = "left"
                else:
                    # drawing the box confidence for keypoints isn't very
                    # useful.
                    continue
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (instance_area < _SMALL_OBJECT_AREA_THRESH *
                        self.output.scale or y1 - y0 < 40 * self.output.scale):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / \
                    np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(
                    color, brightness_factor=0.7)
                # breakpoint()
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size * 3
                )
                # font_size = self._default_font_size * 1.5
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale * 5,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output
    def draw_ellipse(
            self,
            box_coord,
            cov,
            alpha=0.5,
            edge_color="g",
            line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            cov (nd array): 4x4 corner covariance matrix.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        linewidth = max(self._default_font_size / 4, 1)

        width, height, rotation = self.cov_ellipse(cov[0:2, 0:2])

        width[width < 0] = 0
        height[height < 0] = 0
        if not (np.isnan(width) or np.isnan(height) or np.isnan(rotation)):
            width = width.astype(np.int32)
            height = height.astype(np.int32)
            rotation = rotation.astype(np.int32) + 180

            self.output.ax.add_patch(
                mpl.patches.Ellipse(
                    (x0, y0),
                    width,
                    height,
                    angle=rotation,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=linewidth * self.output.scale,
                    alpha=alpha,
                    linestyle=line_style,
                ))

        width, height, rotation = self.cov_ellipse((cov[2:4, 2:4]))

        width[width < 0] = 0
        height[height < 0] = 0
        if not (np.isnan(width) or np.isnan(height) or np.isnan(rotation)):
            width = width.astype(np.int32)
            height = height.astype(np.int32)
            rotation = rotation.astype(np.int32) + 180

            self.output.ax.add_patch(
                mpl.patches.Ellipse(
                    (x1, y1),
                    width,
                    height,
                    angle=rotation,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=linewidth * self.output.scale,
                    alpha=alpha,
                    linestyle=line_style,
                ))

        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5
    ):
        """
        Modified from super class to give access to alpha for box plotting.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [
                random_color(
                    rgb=True,
                    maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k]
                      for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx]
                     for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color, alpha=alpha)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2),
                                      color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    # if drawing boxes, put text on the box corner.
                    text_pos = (x0, y0)
                    horiz_align = "left"
                elif masks is not None:
                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    # drawing the box confidence for keypoints isn't very
                    # useful.
                    continue
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (instance_area < _SMALL_OBJECT_AREA_THRESH *
                        self.output.scale or y1 - y0 < 40 * self.output.scale):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / \
                    np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(
                    color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output

    @staticmethod
    def cov_ellipse(cov, q=None, nsig=2):
        """
        Parameters
        ----------
        cov : (2, 2) array
            Covariance matrix.
        q : float, optional
            Confidence level, should be in (0, 1).
        nsig : int, optional
            Confidence level in unit of standard deviations.
            E.g. 1 stands for 68.3% and 2 stands for 95.4%.

        Returns
        -------
        width, height, rotation :
             The lengths of two axises and the rotation angle in degree
        for the ellipse.
        """

        if q is not None:
            q = np.asarray(q)
        elif nsig is not None:
            q = 2 * norm.cdf(nsig) - 1
        else:
            raise ValueError('One of `q` and `nsig` should be specified.')
        r2 = chi2.ppf(q, 2)

        val, vec = np.linalg.eigh(cov)
        width, height = 2 * np.sqrt(val[:, None] * r2)
        rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

        return width, height, rotation
