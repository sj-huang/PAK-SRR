import pysitk.python_helper as ph
import niftymic.registration.niftyreg as niftyreg
import niftymic.registration.simple_itk_registration as regsitk
# import niftymic.reconstruction.tikhonov_solver as tk
import lsmr as tk
import niftymic.utilities.intensity_correction as ic
import niftymic.utilities.joint_image_mask_builder as imb
import pipeline
import niftymic.reconstruction.scattered_data_approximation as sda
import argparse
from data_pre import *
import SimpleITK as sitk
import time
import os
from niftymic.base.stack import Stack

ep=0.8
parser = argparse.ArgumentParser()
root_path="../data/reo_image/"
os.makedirs(root_path.replace("reo_image","output"),exist_ok=True)
path_list=os.listdir(root_path)

list_img=[os.path.join(root_path,path) for path in path_list]
list_label=[os.path.join(root_path.replace("reo_image","reo_label"),path) for path in path_list]
list_distance=[os.path.join(root_path.replace("reo_image","distance"),path) for path in path_list]
atlas_path="../data/atlas/atlas.nii.gz"
out_path="../data/output/output.nii.gz"



parser.add_argument("--filenames", default=list_img, help="filename")
parser.add_argument("--filenames_masks", default=list_label)
parser.add_argument("--dis_filenames", default=list_distance)
parser.add_argument("--atlas_path", default=atlas_path)

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
parser.add_argument("--threshold", default=0.8)
parser.add_argument("--two_step_cycles", default=3)
parser.add_argument("--interleave", default=3)
parser.add_argument("--viewer", default="itksnap")
parser.add_argument("--verbose", default=0)
parser.add_argument("--multiresolution", default=0)
parser.add_argument("--iter_max", default=10)
parser.add_argument("--sigma", default=1.0)
parser.add_argument("--iter_max_first", default=5)
parser.add_argument("--outlier_rejection", default=1)
parser.add_argument("--out_path", default=out_path)
parser.add_argument("--reconstruction_type", default="TK1L2")
parser.add_argument("--dilation_radius", default=3)
rejection_measure = "NCC"
args = parser.parse_args()

# ------------------------Volume-to-Volume Registration--------------------
stacks=data_process(args)
args.target_stack_index=target_select(stacks)


reference = Stack.from_stack(stacks[args.target_stack_index])
stacks_dis=dis_data_process(args)
reference_dis = Stack.from_stack(stacks_dis[args.target_stack_index])
vol_registration = niftyreg.RegAladin(
                registration_type="Rigid",
                use_fixed_mask=True,
                use_moving_mask=True,
                use_verbose=False,
                )

v2vreg= pipeline.VolumeToVolumeRegistration(
    stacks=stacks,
    stacks_dis=stacks_dis,
    reference=reference,
    registration_method=vol_registration,
    verbose=False,
)



start = time.time()
v2vreg.run()

stacks_o = v2vreg.get_stacks()



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


# -----------Two-step Slice-to-Volume Registration-Reconstruction----------

# registration = regsitk.SimpleItkRegistration(
#             moving=HR_volume,
#             use_fixed_mask=True,
#             use_moving_mask=True,
#             interpolator="Linear",
#             metric=args.metric,
#             metric_params=None,
#             use_multiresolution_framework=args.multiresolution,
#             shrink_factors=args.shrink_factors,
#             smoothing_sigmas=args.smoothing_sigmas,
#             initializer_type="SelfGEOMETRY",
#             optimizer="ConjugateGradientLineSearch",
#             optimizer_params={
#                 "learningRate": 1,
#                 "numberOfIterations": 100,
#                 "lineSearchUpperLimit": 2,
#             },
#             scales_estimator="Jacobian",
#             use_verbose=False,
#             # use_oriented_psf=True
#
#         )
from slice2volume import S2V
registration = S2V(moving=HR_volume,fixed=None,dis=None)

recon_method = tk.TikhonovSolver(
                stacks=stacks,
                stacks_dis=stacks_dis,
                reconstruction=HR_volume,
                atlas=args.atlas_path,
                reg_type="TK1",
                minimizer="lsmr",
                # minimizer="least_squares",
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
        stacks_dis=stacks_dis,
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
        verbose=ep,
    )
two_step_s2v_reg_recon.run()
HR_volume_iterations = \
    two_step_s2v_reg_recon.get_iterative_reconstructions()
stacks = two_step_s2v_reg_recon.get_stacks()


# ---------------------Final Volumetric Reconstruction---------------------
recon_method = tk.TikhonovSolver(
    stacks=stacks,
    stacks_dis=stacks_dis,
    reconstruction=HR_volume,
    atlas=args.atlas_path,
    reg_type="TK1" if args.reconstruction_type == "TK1L2" else "TK0",
    use_masks=args.use_masks_srr,
)
recon_method.set_alpha(args.alpha)
recon_method.set_index(ep)
recon_method.set_iter_max(args.iter_max)
recon_method.set_verbose(True)
recon_method.run()
HR_volume_final = recon_method.get_reconstruction()
sitk.WriteImage(HR_volume_final.sitk,args.out_path)
sitk.WriteImage(HR_volume_final.sitk_mask,args.out_path.replace(".nii.gz","_mask.nii.gz"))
end=time.time()
run_time=end-start
print("Run time: ", run_time)
































