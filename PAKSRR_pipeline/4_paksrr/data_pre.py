
import niftymic.base.data_reader as dr
import niftymic.utilities.data_preprocessing as dp
import niftymic.base.stack as st
import numpy as np
import pysitk.simple_itk_helper as sitkh
import niftymic.utilities.segmentation_propagation as segprop
import SimpleITK as sitk
import itkExtras
import copy
from niftymic.base.stack import Stack



def data_process(args):
    data_reader = dr.MultipleImagesReader(
        file_paths=args.filenames,
        file_paths_masks=args.filenames_masks,
        suffix_mask=args.suffix_mask,
        stacks_slice_thicknesses=args.slice_thicknesses,
    )
    data_reader.read_data()
    stacks = data_reader.get_data()
    segmentation_propagator = segprop.SegmentationPropagation(
        dilation_radius=args.dilation_radius,
        dilation_kernel="Ball",
    )
    data_preprocessing = dp.DataPreprocessing(
        stacks=stacks,
        segmentation_propagator=segmentation_propagator,
        use_cropping_to_mask=True,
        use_N4BiasFieldCorrector=args.bias_field_correction,
        target_stack_index=args.target_stack_index,
        boundary_i=args.boundary_stacks[0],
        boundary_j=args.boundary_stacks[1],
        boundary_k=args.boundary_stacks[2],
        unit="mm",
    )
    data_preprocessing.run()
    stacks = data_preprocessing.get_preprocessed_stacks()
    # reference = st.Stack.from_stack(stacks[args.target_stack_index])
    return stacks



def dis_data_process(args):
    data_reader = dr.MultipleImagesReader(
        file_paths=args.dis_filenames,
        file_paths_masks=args.filenames_masks,
        suffix_mask=args.suffix_mask,
        stacks_slice_thicknesses=args.slice_thicknesses,
    )
    data_reader.read_data()
    stacks = data_reader.get_data()
    segmentation_propagator = segprop.SegmentationPropagation(
        dilation_radius=args.dilation_radius,
        dilation_kernel="Ball",
    )
    data_preprocessing = dp.DataPreprocessing(
        stacks=stacks,
        segmentation_propagator=segmentation_propagator,
        use_cropping_to_mask=True,
        use_N4BiasFieldCorrector=False,
        target_stack_index=args.target_stack_index,
        boundary_i=args.boundary_stacks[0],
        boundary_j=args.boundary_stacks[1],
        boundary_k=args.boundary_stacks[2],
        unit="mm",
    )
    data_preprocessing.run()
    stacks = data_preprocessing.get_preprocessed_stacks()

    return stacks


def target_select(stacks):
    target_index=-1
    max_overlap=0
    for i in range(len(stacks)):
        stack=sitk.GetArrayFromImage(stacks[i].sitk_mask)
        stack[stack!=0]=1
        overlap=0
        for j in range(stack.shape[0]-1):
            overlap+=(stack[j]*stack[j+1]).sum()
        if overlap>max_overlap:
            max_overlap=overlap
            target_index+=1
    return target_index













