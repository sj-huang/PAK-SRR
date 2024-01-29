import numpy as np
import pysitk.python_helper as ph
import niftymic.base.data_writer as dw
import niftymic.registration.niftyreg as niftyreg
import niftymic.registration.simple_itk_registration as regsitk
import niftymic.reconstruction.tikhonov_solver as tk
import niftymic.utilities.intensity_correction as ic
import niftymic.utilities.joint_image_mask_builder as imb
import niftymic.utilities.volumetric_reconstruction_pipeline as pipeline
import niftymic.reconstruction.scattered_data_approximation as sda
import argparse
from data_pre import data_process
import os
# list_index=[35,27]
# for i in range(20,21):
parser = argparse.ArgumentParser()
root_path="../data/reo_image/"
os.makedirs(root_path.replace("reo_image","output"),exist_ok=True)
path_list=os.listdir(root_path)

list_img=[os.path.join(root_path,path) for path in path_list]
list_label=[os.path.join(root_path.replace("reo_image","reo_label"),path) for path in path_list]
list_distance=[os.path.join(root_path.replace("reo_image","distance"),path) for path in path_list]
atlas_path="../data/atlas/atlas.nii.gz"
out_path="../data/output/output_niftymic.nii.gz"

parser.add_argument("--filenames", default=list_img, help="filename")
parser.add_argument("--filenames_masks", default=list_label)
parser.add_argument("--suffix_mask", default="_mask")
parser.add_argument("--slice_thicknesses", default=None)
parser.add_argument("--boundary_stacks", default=[10, 10, 0])
parser.add_argument("--bias_field_correction", default=True)
parser.add_argument("--target_stack_index", default=0)
parser.add_argument("--isotropic_resolution", default=0.8)
parser.add_argument("--extra_frame_target", default=10)
parser.add_argument("--metric", default="Correlation")
parser.add_argument("--shrink_factors", default=[3, 2, 1])
parser.add_argument("--smoothing_sigmas", default=[1.5, 1, 0])
parser.add_argument("--alpha_first", default=0.2)
parser.add_argument("--use_masks_srr", default=0)
parser.add_argument("--alpha", default=0.015)
parser.add_argument("--threshold_first", default=0.5)
parser.add_argument("--threshold", default=0.7)
parser.add_argument("--two_step_cycles", default=3)
parser.add_argument("--interleave", default=3)
parser.add_argument("--viewer", default="itksnap")
parser.add_argument("--verbose", default=0)
parser.add_argument("--multiresolution", default=0)
parser.add_argument("--iter_max", default=10)
parser.add_argument("--sigma", default=1.0)
parser.add_argument("--iter_max_first", default=5)
parser.add_argument("--outlier_rejection", default=1)
parser.add_argument("--output", default=out_path)
parser.add_argument("--reconstruction_type", default="TK1L2")
parser.add_argument("--dilation_radius", default=3)
rejection_measure = "NCC"
args = parser.parse_args()



# ------------------------Volume-to-Volume Registration--------------------
stacks,reference=data_process(args)

# stacks_contour=data_enhance(args)
vol_registration = niftyreg.RegAladin(
                registration_type="Rigid",
                use_fixed_mask=True,
                use_moving_mask=True,
                use_verbose=False,
                )

v2vreg=pipeline.VolumeToVolumeRegistration(
    stacks=stacks,
    reference=reference,
    registration_method=vol_registration,
    verbose=False,
)
v2vreg.run()
stacks = v2vreg.get_stacks()
stacks_original=stacks.copy()
# ---------------------------Intensity Correction--------------------------

intensity_corrector = ic.IntensityCorrection()
intensity_corrector.use_individual_slice_correction(False)
intensity_corrector.use_reference_mask(True)
intensity_corrector.use_stack_mask(True)
intensity_corrector.use_verbose(False)

for i, stack in enumerate(stacks):
    if i == args.target_stack_index:
        ph.print_info("Stack %d (%s): Reference image. Skipped." % (
            i + 1, stack.get_filename()))
        continue
    else:
        ph.print_info("Stack %d (%s): Intensity Correction ... " % (
            i + 1, stack.get_filename()), newline=False)
    intensity_corrector.set_stack(stack)
    intensity_corrector.set_reference(
        stacks[args.target_stack_index].get_resampled_stack(
            resampling_grid=stack.sitk,
            interpolator="NearestNeighbor",
        ))
    intensity_corrector.run_linear_intensity_correction()
    stacks[i] = intensity_corrector.get_intensity_corrected_stack()



# ---------------------------Create first volume---------------------------
HR_volume = reference.get_isotropically_resampled_stack(
    resolution=args.isotropic_resolution)
joint_image_mask_builder = imb.JointImageMaskBuilder(
    stacks=stacks,
    target=HR_volume,
    dilation_radius=1,
)
joint_image_mask_builder.run()
HR_volume = joint_image_mask_builder.get_stack()
# Crop to space defined by mask (plus extra margin)
HR_volume = HR_volume.get_cropped_stack_based_on_mask(
    boundary_i=args.extra_frame_target,
    boundary_j=args.extra_frame_target,
    boundary_k=args.extra_frame_target,
    unit="mm",
)
SDA = sda.ScatteredDataApproximation(
            stacks, HR_volume, sigma=args.sigma)
SDA.run()
HR_volume = SDA.get_reconstruction()

SDA = sda.ScatteredDataApproximation(
            stacks, HR_volume, sigma=args.sigma, sda_mask=True)
SDA.run()
HR_volume = SDA.get_reconstruction()
# -----------Two-step Slice-to-Volume Registration-Reconstruction----------
print("# -----------Two-step Slice-to-Volume Registration-Reconstruction----------")
registration = regsitk.SimpleItkRegistration(
            moving=HR_volume,
            use_fixed_mask=True,
            use_moving_mask=True,
            interpolator="Linear",
            metric=args.metric,
            metric_params=None,
            use_multiresolution_framework=args.multiresolution,
            shrink_factors=args.shrink_factors,
            smoothing_sigmas=args.smoothing_sigmas,
            initializer_type="SelfGEOMETRY",
            optimizer="ConjugateGradientLineSearch",
            optimizer_params={
                "learningRate": 1,
                "numberOfIterations": 100,
                "lineSearchUpperLimit": 2,
            },
            scales_estimator="Jacobian",
            use_verbose=False,
        )

recon_method = tk.TikhonovSolver(
                stacks=stacks,
                reconstruction=HR_volume,
                reg_type="TK1",
                minimizer="lsmr",
                alpha=args.alpha_first,
                iter_max=np.min([args.iter_max_first, args.iter_max]),
                verbose=True,
                use_masks=args.use_masks_srr,
            )
alpha_range = [args.alpha_first, args.alpha]
alphas = np.linspace(
            alpha_range[0], alpha_range[1], args.two_step_cycles)

# Define outlier rejection threshold after each S2V-reg step
thresholds = np.linspace(
    args.threshold_first, args.threshold, args.two_step_cycles)

two_step_s2v_reg_recon = \
    pipeline.TwoStepSliceToVolumeRegistrationReconstruction(
        stacks=stacks,
        # stacks_contours=stacks_contour,
        reference=HR_volume,
        registration_method=registration,
        reconstruction_method=recon_method,
        cycles=args.two_step_cycles,
        alphas=alphas[0:args.two_step_cycles - 1],
        outlier_rejection=args.outlier_rejection,
        threshold_measure=rejection_measure,
        thresholds=thresholds,
        interleave=args.interleave,
        viewer=args.viewer,
        verbose=args.verbose,
    )
two_step_s2v_reg_recon.run()
HR_volume_iterations = \
    two_step_s2v_reg_recon.get_iterative_reconstructions()
stacks = two_step_s2v_reg_recon.get_stacks()



# ---------------------Final Volumetric Reconstruction---------------------
recon_method = tk.TikhonovSolver(
    stacks=stacks,
    reconstruction=HR_volume,
    reg_type="TK1" if args.reconstruction_type == "TK1L2" else "TK0",
    use_masks=args.use_masks_srr,
)
recon_method.set_alpha(args.alpha)
recon_method.set_iter_max(args.iter_max)
recon_method.set_verbose(True)
recon_method.run()
HR_volume_final = recon_method.get_reconstruction()

SDA = sda.ScatteredDataApproximation(
    stacks, HR_volume_final, sigma=args.sigma
    , sda_mask=True)
SDA.run()
HR_volume_final = SDA.get_reconstruction()

# Write SRR results
filename = recon_method.get_setting_specific_filename()
HR_volume_final.set_filename(filename)
dw.DataWriter.write_image(
    HR_volume_final.sitk,
    args.output,
    description=filename)
dw.DataWriter.write_mask(
    HR_volume_final.sitk_mask,
    ph.append_to_filename(args.output, "_mask"),
)
