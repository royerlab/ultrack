from typing import Optional, Tuple, Union

import dask.array as da
import numpy as np
import torch as th
from numpy.typing import ArrayLike

from ultrack.utils.cuda import import_module, to_cpu, torch_default_device


class PlantSeg:
    """
    A class for performing boundary detection using the Plant-Seg model.
    Plant-Seg documentation for more details, https://github.com/hci-unihd/plant-seg

    Parameters
    ----------
    model_name : str
        Name of the pre-trained segmentation model.
    model_update : bool, optional
        Update the model if True. Default is False.
    device : {str, torch.device}, optional
        Device for model execution. If None, the default device is used.
        Default is None.
    patch : tuple[int], optional
        Patch size for model inference. Default is (80, 160, 160).
    stride_ratio : float, optional
        Stride ratio for patch sampling. Default is 0.75.
    batch_size : int, optional
        Batch size for inference. Default is None.
    preprocess_sigma : float, optional
        Sigma value for Gaussian preprocessing filter. Default is None.
    postprocess_sigma : float, optional
        Sigma value for Gaussian postprocessing filter. Default is None.
    scale_factor : tuple[float], optional
        Scaling factors for input images. Default is None.

    See Also
    --------
    :class:`ArrayPredictor`: Class for making predictions using the PlantSeg model.

    Examples
    --------
    >>> seg_model = PlantSeg(model_name='generic_light_sheet_3D_unet', batch_size=4)
    >>> segmentation_result = seg_model(image_data)
    """

    def __init__(
        self,
        model_name: str,
        model_update: bool = False,
        device: Optional[Union[str, th.device]] = None,
        patch: Tuple[int] = (80, 160, 160),
        stride_ratio: float = 0.75,
        batch_size: Optional[int] = None,
        preprocess_sigma: Optional[float] = None,
        postprocess_sigma: Optional[float] = None,
        scale_factor: Optional[Tuple[float]] = None,
    ) -> None:
        """
        Initialized Plant-Seg model.
        """
        from plantseg.predictions.functional.array_predictor import ArrayPredictor
        from plantseg.predictions.functional.utils import (
            get_model_config,
            get_patch_halo,
        )

        if device is None:
            device = torch_default_device()

        model, model_config, model_path = get_model_config(model_name, model_update)
        patch_halo = get_patch_halo(model_name)
        state = th.load(model_path, map_location="cpu")
        model.load_state_dict(state)

        self.model_name = model_name
        self.patch = patch
        self.stride_ratio = stride_ratio
        self.preprocess_sigma = preprocess_sigma
        self.postprocess_sigma = postprocess_sigma
        self.scale_factor = (
            scale_factor if scale_factor is None else np.asarray(scale_factor)
        )
        self.predictor = ArrayPredictor(
            model=model,
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            device=device,
            patch=patch,
            patch_halo=patch_halo,
            single_batch_mode=False,
            headless=False,
            disable_tqdm=True,
        )

        # overwriting plantseg batch size
        if batch_size is not None:
            self.predictor.batch_size = batch_size

    def __call__(
        self,
        image: ArrayLike,
        transpose: Optional[Tuple[int]] = None,
    ) -> np.ndarray:
        """
        Perform boundary detection using the PlantSeg model.

        Parameters
        ----------
        image : ArrayLike
            Input image data as a numpy, cupy, or Dask array.
        transpose : tuple[int], optional
            Axes permutation for the input image.
            Permutation is applied after pre-processing and before inference and reverted after inference.
            Default is None.

        Returns
        -------
        np.ndarray
            Segmentation boundary probability map as a numpy array.
        """
        from plantseg.dataprocessing import fix_input_shape
        from plantseg.predictions.functional.utils import get_array_dataset

        if isinstance(image, da.Array):
            # avoiding building a large compute graph
            image = image.compute()

        image = fix_input_shape(image)
        orig_shape = image.shape

        ndi = import_module("scipy", "ndimage", image)

        if self.preprocess_sigma is not None:
            image = ndi.gaussian_filter(image, sigma=self.preprocess_sigma)

        if self.scale_factor is not None:
            image = ndi.zoom(image, zoom=self.scale_factor, order=1)

        if transpose is None:
            patch = self.patch
        else:
            # transposes axes
            image = np.transpose(image, axes=transpose)
            patch = [self.patch[i] for i in transpose]

        dataset = get_array_dataset(
            raw=to_cpu(image),  # making sure dataset used for inference is on the CPU
            model_name=self.model_name,
            patch=patch,
            stride_ratio=self.stride_ratio,
            global_normalization=True,
        )
        probs = self.predictor(dataset)[0]

        # reverting transpose
        if transpose:
            image = np.transpose(image, axes=np.argsort(transpose))

        if self.scale_factor is not None:
            probs = np.asarray(probs, like=image)
            inv_factor = tuple(o / s for o, s in zip(orig_shape, probs.shape))
            probs = ndi.zoom(probs, zoom=inv_factor, order=1)

        if self.postprocess_sigma is not None:
            probs = np.asarray(probs, like=image)
            probs = ndi.gaussian_filter(probs, sigma=self.postprocess_sigma)

        return to_cpu(probs)
