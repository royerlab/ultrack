import shutil
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import scipy.ndimage as ndi
import torch as th
from numpy.typing import ArrayLike
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries

from ultrack.utils.cuda import torch_default_device


class MicroSAM:
    """
    Using a SAM model, generates masks for the entire image.
    Generates a grid of point prompts over the image, then filters
    low quality and duplicate masks. The default settings are chosen
    for SAM with a ViT-H backbone.

    Parameters
    ----------
    device : Union[str, th.device, None] = None,
        The device to run the model on. If None, defaults to the default.
    points_per_side : int or None, optional
        The number of points to be sampled along one side of the image.
        The total number of points is points_per_side**2. If None, 'point_grids'
        must provide explicit point sampling.
    points_per_batch : int, optional
        Sets the number of points run simultaneously by the model.
        Higher numbers may be faster but use more GPU memory.
    pred_iou_thresh : float, optional
        A filtering threshold in [0,1], using the model's predicted mask quality.
    stability_score_thresh : float, optional
        A filtering threshold in [0,1], using the stability of the mask under
        changes to the cutoff used to binarize the model's mask predictions.
    stability_score_offset : float, optional
        The amount to shift the cutoff when calculating the stability score.
    box_nms_thresh : float, optional
        The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
    crop_n_layers : int, optional
        If >0, mask prediction will be run again on crops of the image. Sets the
        number of layers to run, where each layer has 2**i_layer number of image crops.
    crop_nms_thresh : float, optional
        The box IoU cutoff used by non-maximal suppression to filter duplicate
        masks between different crops.
    crop_overlap_ratio : float, optional
        Sets the degree to which crops overlap. In the first crop layer, crops will
        overlap by this fraction of the image length. Later layers with more crops
        scale down this overlap.
    crop_n_points_downscale_factor : int, optional
        The number of points-per-side sampled in layer n is scaled down by
        crop_n_points_downscale_factor**n.
    point_grids : list[np.ndarray] or None, optional
        A list over explicit grids of points used for sampling, normalized to [0,1].
        The nth grid in the list is used in the nth crop layer. Exclusive with points_per_side.
    min_mask_region_area : int, optional
        If >0, postprocessing will be applied to remove disconnected regions and holes
        in masks with an area smaller than min_mask_region_area. Requires opencv.
    max_mask_region_prop : float, optional
        Regions larger than max_mask_region_prop * image area will be removed.
    mutex_watershed : bool, optional
        If True, uses a mutex watershed algorithm to generate masks.
    tile_shape : Tuple[int, int], optional
        The shape of the tiles used to precompute image embeddings.
    halo_shape : Tuple[int, int], optional
        The shape of the halo (overlap) around each tile used to precompute image embeddings.
    """

    def __init__(
        self,
        model_type: str = "vit_h_lm",
        device: Union[str, th.device, None] = None,
        points_per_side: Optional[int] = 64,
        points_per_batch: int = 128,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.5,
        stability_score_offset: float = 0.75,
        box_nms_thresh: float = 0.5,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        max_mask_region_prop: float = 0.1,
        mutex_watershed: bool = False,
        tile_shape: Tuple[int, int] = (512, 512),
        halo_shape: Tuple[int, int] = (128, 128),
    ) -> None:
        from micro_sam import instance_segmentation, util

        if device is None:
            device = torch_default_device()

        if isinstance(device, str):
            device = th.device(device)

        self._predictor = util.get_sam_model(device=device, model_type=model_type)

        self._mutex_watershed = mutex_watershed
        if mutex_watershed:
            self._amg = instance_segmentation.TiledEmbeddingMaskGenerator(
                self._predictor,
                n_threads=1,
                stability_score_offset=stability_score_offset,
            )
        else:
            self._amg = instance_segmentation.TiledAutomaticMaskGenerator(
                self._predictor,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                point_grids=point_grids,
                stability_score_offset=stability_score_offset,
            )

        self._pred_iou_thresh = pred_iou_thresh
        self._stability_score_thresh = stability_score_thresh
        self._box_nms_thresh = box_nms_thresh
        self._crop_n_layers = crop_n_layers
        self._crop_nms_thresh = crop_nms_thresh
        self._crop_overlap_ratio = crop_overlap_ratio
        self._crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self._min_mask_region_area = min_mask_region_area
        self._max_mask_region_prop = max_mask_region_prop

        self._embedding_cache_dir = Path("/tmp/micro-sam")

        self._tile_shape = tile_shape
        self._halo_shape = halo_shape

    def __call__(self, image: ArrayLike) -> np.ndarray:
        """
        Estimate contour of objects of an image.

        Background regions not assigned to any label are set to -1.

        Parameters
        ----------
        image : ArrayLike
            The input image to be processed.

        Returns
        -------
        np.ndarray
            The processed image with contours derived from the identified masks.
        """
        from micro_sam.util import precompute_image_embeddings
        from segment_anything.utils.amg import area_from_rle, rle_to_mask

        image = np.asarray(image)

        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got {image.ndim}D.")

        embedding_path = self._embedding_cache_dir / f"{uuid4()}.zarr"

        if embedding_path.exists():
            shutil.rmtree(embedding_path)

        embeddings = precompute_image_embeddings(
            self._predictor,
            image,
            save_path=str(embedding_path),
            tile_shape=self._tile_shape,
            halo=self._halo_shape,
        )

        self._amg.initialize(image, embeddings, verbose=True)

        if self._mutex_watershed:
            preds = self._amg.generate(
                pred_iou_thresh=self._pred_iou_thresh,
                stability_score_thresh=self._stability_score_thresh,
                box_nms_thresh=self._box_nms_thresh,
                min_mask_region_area=self._min_mask_region_area,
            )
            contour = find_boundaries(preds, mode="inner")
            detection = preds > 0

        else:
            preds = self._amg.generate(
                pred_iou_thresh=self._pred_iou_thresh,
                stability_score_thresh=self._stability_score_thresh,
                box_nms_thresh=self._box_nms_thresh,
                crop_nms_thresh=self._crop_nms_thresh,
                min_mask_region_area=self._min_mask_region_area,
                output_mode="uncompressed_rle",
            )

            contour = th.zeros(image.shape, dtype=th.float32)
            detection = th.zeros(image.shape, dtype=bool)
            max_area = np.prod(image.shape, dtype=float) * self._max_mask_region_prop
            for pred in preds:
                if area_from_rle(pred["segmentation"]) > max_area:
                    continue
                mask = rle_to_mask(pred["segmentation"])
                contour += find_boundaries(mask, mode="inner")
                detection[mask.astype(bool)] = True

            contour /= contour.max()
            contour = contour.numpy()

        contour[~detection] = -1

        shutil.rmtree(embedding_path)

        return contour


def set_peak_maxima_prompt(
    sam: MicroSAM,
    sigma: float,
    min_distance: int,
    threshold_rel: float,
    **kwargs,
) -> Callable:
    """
    Configure a function to locate peak maxima in a given image using the MicroSAM framework.
    The function first applies a Gaussian blur to the image, then finds local maxima,
    and finally processes the image using the MicroSAM framework with the identified maxima.

    Parameters
    ----------
    sam : MicroSAM
        A MicroSAM instance used for image processing.
    sigma : float
        Standard deviation for Gaussian filter applied to the image.
    min_distance : int
        Minimum number of pixels separating peaks. Peaks closer than this are considered a single peak.
    threshold_rel : float
        Minimum intensity difference between the peak and the pixels surrounding the peak for it to be
        considered a true peak. It is a relative threshold with respect to the maximum intensity in the image.
    **kwargs :
        Additional keyword arguments to pass to the `peak_local_max` function.

    Returns
    -------
    Callable
        A function that, when given an image, will return an image processed by
        MicroSAM with contours derived from the identified peak maxima.

    Examples
    --------
    >>> sam_instance = ...  # Initialize or provide a MicroSAM instance.
    >>> peak_maxima_function = set_peak_maxima_prompt(
    >>>     sam_instance, sigma=1, min_distance=10, threshold_rel=0.1,
    >>> )
    >>> processed_image = peak_maxima_function(input_image)
    """

    def _peak_maxima_micro_sam(image: ArrayLike) -> np.ndarray:
        """
        Inner function that processes the given image, finds peak maxima,
        and applies the MicroSAM processing with the identified maxima.

        Parameters
        ----------
        image : ArrayLike
            The input image to be processed.

        Returns
        -------
        np.ndarray
            The processed image with contours derived from the identified peak maxima.
        """
        image = np.asarray(image)
        # find maxima coordinates
        blurred_image = ndi.gaussian_filter(image, sigma=sigma)
        coords = peak_local_max(
            blurred_image,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            **kwargs,
        )
        coords = coords / image.shape

        prev_grid = sam._amg.point_grids
        # set point grid to maxima coordinates
        sam._amg.point_grids = [coords]
        # run sam
        contour = sam(image)
        # reset point grid
        sam._amg.point_grids = prev_grid

        return contour

    return _peak_maxima_micro_sam
