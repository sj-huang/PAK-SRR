
import niftymic.base.data_reader as dr
import niftymic.utilities.data_preprocessing as dp
import niftymic.base.stack as st
import pysitk.simple_itk_helper as sitkh
import niftymic.utilities.segmentation_propagation as segprop






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
    reference = st.Stack.from_stack(stacks[args.target_stack_index])
    return stacks,reference






























