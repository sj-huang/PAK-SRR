# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from seg_sav import NiftiSaver, PNGSaver
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode, exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")

# from predictor import SamPredictor
import os

from segment_anything import build_sam, build_sam_vit_b,SamPredictor
import cv2
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
from numpy import *
import warnings
from scipy.ndimage import zoom
def bounding_box(mask):
    mask=mask.detach().cpu().numpy()
    coords = np.column_stack(np.where(mask>0.5))
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return np.array([y_min, x_min,y_max, x_max])
def find_hole(mask):
    mask = mask.detach().cpu().numpy()
    imageio.imwrite("slice.png", mask)
    mask = cv2.imread("slice.png")
    # blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edged = cv2.Canny(mask, 30, 150)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)
def detect_slice(engine_output):
    index_list=[]
    for s in range(1,engine_output.shape[-1]-1):
        # if engine_output[:,:,s].sum()<1 and engine_output[:,:,s-1].sum()>1 and engine_output[:,:,s+1].sum()>1:
        #     index_list.append(s)
        # elif engine_output[:,:,s].sum()<(0.25*(engine_output[:,:,s-1]+engine_output[:,:,s+1])).sum():
        #     index_list.append(s)
        # elif find_hole(engine_output[:,:,s])>1:
        #     index_list.append(s)
        if engine_output[:,:,s].sum()<0.95*engine_output[:,:,s-1].sum() and engine_output[:,:,s].sum()<0.95*engine_output[:,:,s+1].sum():
            index_list.append(s)
            # print(s)
    return index_list
def sam_process(engine_output, meta_data):
    index=detect_slice(engine_output[0,0])
    sitk_image=sitk.ReadImage(meta_data["filename_or_obj"])
    np_image=sitk.GetArrayFromImage(sitk_image)[0].transpose(2,1,0)
    T=(engine_output.shape[2]/np_image.shape[0],engine_output.shape[3]/np_image.shape[1],1)
    np_image=zoom(np_image,T)
    for i in index:
        image = np_image[:,:,i]
        label=engine_output[0,0,:,:,i-1]+engine_output[0,0,:,:,i+1]
        imageio.imwrite("slice.png", image)
        image = cv2.imread("slice.png")
        input_box = bounding_box(label)
        predictor = SamPredictor(
            build_sam_vit_b(checkpoint="../../models/sam_vit_b_01ec64.pth").to(device="cuda"))
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        masks = masks * 1
        engine_output[0,0,:,:,i] = torch.tensor(masks[0])
    return engine_output

def merge_samonai(engine_output_monai, engine_output_sam):
    output=engine_output_monai.clone()
    for i in range(1,engine_output_monai.shape[-1]-1):
        if (engine_output_monai[:,:,:,:,i]*engine_output_monai[:,:,:,:,i-1]*engine_output_monai[:,:,:,:,i+1]).sum()<(engine_output_sam[:,:,:,:,i]*engine_output_sam[:,:,:,:,i-1]*engine_output_sam[:,:,:,:,i+1]).sum():
            output[:,:,:,:,i]=engine_output_sam[:,:,:,:,i]
        else:
            output[:,:,:,:,i]=engine_output_monai[:,:,:,:,i]
    return output
class SegmentationSaver:
    """
    Event handler triggered on completing every iteration to save the segmentation predictions into files.
    """

    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "seg",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: Union[GridSampleMode, InterpolateMode, str] = "nearest",
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        scale: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            output_dir: output image directory.
            output_postfix: a string appended to all output file names.
            output_ext: output file extension name.
            resample: whether to resample before saving the data array.
            mode: This option is used when ``resample = True``. Defaults to ``"nearest"``.

                - NIfTI files {``"bilinear"``, ``"nearest"``}
                    Interpolation mode to calculate output values.
                    See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                - PNG files {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                    The interpolation mode.
                    See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

            padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.

                - NIfTI files {``"zeros"``, ``"border"``, ``"reflection"``}
                    Padding mode for outside grid values.
                    See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                - PNG files
                    This option is ignored.

            scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
                [0, 255] (uint8) or [0, 65535] (uint16). Default is None to disable scaling.
                It's used for PNG format only.
            dtype: convert the image data to save to this data type.
                If None, keep the original type of data. It's used for Nifti format only.
            batch_transform: a callable that is used to transform the
                ignite.engine.batch into expected format to extract the meta_data dictionary.
            output_transform: a callable that is used to transform the
                ignite.engine.output into the form expected image data.
                The first dimension of this transform's output will be treated as the
                batch dimension. Each item in the batch will be saved individually.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        self.saver: Union[NiftiSaver, PNGSaver]
        if output_ext in (".nii.gz", ".nii"):
            self.saver = NiftiSaver(
                output_dir=output_dir,
                output_postfix=output_postfix,
                output_ext=output_ext,
                resample=resample,
                mode=GridSampleMode(mode),
                padding_mode=padding_mode,
                dtype=dtype,
            )
        elif output_ext == ".png":
            self.saver = PNGSaver(
                output_dir=output_dir,
                output_postfix=output_postfix,
                output_ext=output_ext,
                resample=resample,
                mode=InterpolateMode(mode),
                scale=scale,
            )
        self.batch_transform = batch_transform
        self.output_transform = output_transform

        self.logger = logging.getLogger(name)
        self._name = name

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        """
        This method assumes self.batch_transform will extract metadata from the input batch.
        Output file datatype is determined from ``engine.state.output.dtype``.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        meta_data = self.batch_transform(engine.state.batch)
        engine_output_monai = self.output_transform(engine.state.output)
        # try:
        engine_output_sam=sam_process(engine_output_monai, meta_data)
        engine_output = merge_samonai(engine_output_monai, engine_output_sam)
        # except:pass
        self.saver.save_batch(engine_output, meta_data)
        # print(meta_data)
        self.logger.info("saved all the model outputs into files.")