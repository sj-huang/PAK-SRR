##
# \file volumetric_reconstruction_pipeline.py
# \brief      Collection of modules useful for registration and
#             reconstruction tasks.
#
# E.g. Volume-to-Volume Registration, Slice-to-Volume registration,
# Multi-component Reconstruction.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#
import six
import numpy as np
import SimpleITK as sitk
from abc import ABCMeta, abstractmethod

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.validation.motion_evaluator as me
import niftymic.utilities.outlier_rejector as outre
import niftymic.registration.transform_initializer as tinit
import niftymic.reconstruction.scattered_data_approximation as sda
import niftymic.utilities.binary_mask_from_mask_srr_estimator as bm

from niftymic.definitions import VIEWER
import torch as t
from torch.autograd import Variable as V
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
##
# Class which holds basic interface for all modules
# \date       2017-08-08 02:20:40+0100
#
class Pipeline(object):
    __metaclass__ = ABCMeta

    def __init__(self, stacks,stacks_dis, verbose, viewer):
        self._stacks = stacks
        self._stacks_dis = stacks_dis
        self._verbose = verbose
        self._viewer = viewer

        self._computational_time = ph.get_zero_time()

    def set_stacks(self, stacks):
        self._stacks = stacks

    def get_stacks(self):
        return [st.Stack.from_stack(stack) for stack in self._stacks]

    def set_stacks_dis(self, stacks_dis):
        self._stacks_dis = stacks_dis

    def get_stacks_dis(self):
        return [st.Stack.from_stack(stack_dis) for stack_dis in self._stacks_dis]

    def set_verbose(self, verbose):
        self._verbose = verbose

    def get_verbose(self):
        return self._verbose

    def get_computational_time(self):
        return self._computational_time

    def run(self):

        time_start = ph.start_timing()

        self._run()

        self._computational_time = ph.stop_timing(time_start)

        if self._verbose:
            ph.print_info("Required computational time: %s" %
                          (self.get_computational_time()))

    @abstractmethod
    def _run(self):
        pass


##
# Class which holds basic interface for all registration associated modules
# \date       2017-08-08 02:21:17+0100
#
class RegistrationPipeline(Pipeline):
    __metaclass__ = ABCMeta

    ##
    # Store variables relevant to register stacks to a certain reference volume
    # \date       2017-08-08 02:21:56+0100
    #
    # \param      self                 The object
    # \param      verbose              Verbose output, bool
    # \param      stacks               List of Stack objects
    # \param      reference            Reference as Stack object
    # \param      registration_method  Registration method, e.g.
    #                                  SimpleItkRegistration
    #
    def __init__(self, verbose, stacks,stacks_dis, reference, registration_method, viewer):

        Pipeline.__init__(self, stacks=stacks,stacks_dis=stacks_dis, verbose=verbose, viewer=viewer)

        self._reference = reference
        self._registration_method = registration_method

    def set_reference(self, reference):
        self._reference = reference

    def get_reference(self):
        return st.Stack.from_stack(self._reference)


##
# Class to perform Volume-to-Volume registration
# \date       2017-08-08 02:28:56+0100
#
class VolumeToVolumeRegistration(RegistrationPipeline):

    ##
    # Store relevant information to perform Volume-to-Volume registration
    # \date       2017-08-08 02:29:13+0100
    #
    # \param      self                 The object
    # \param      stacks               The stacks
    # \param      reference            The reference
    # \param      registration_method  The registration method
    # \param      verbose              The verbose
    #
    def __init__(self,
                 stacks,
                 stacks_dis,
                 reference,
                 registration_method,
                 verbose=1,
                 viewer=VIEWER,
                 robust=False,
                 ):
        RegistrationPipeline.__init__(
            self,
            stacks=stacks,
            stacks_dis=stacks_dis,
            reference=reference,
            registration_method=registration_method,
            viewer=viewer,
            verbose=verbose,
        )
        self._robust = robust

    def _run(self):

        ph.print_title("Volume-to-Volume Registration")

        for i in range(0, len(self._stacks)):
            txt = "Volume-to-Volume Registration -- " \
                "Stack %d/%d" % (i + 1, len(self._stacks))
            if self._verbose:
                ph.print_subtitle(txt)
            else:
                ph.print_info(txt)

            if self._robust:
                transform_initializer = tinit.TransformInitializer(
                    fixed=self._reference,
                    moving=self._stacks[i],
                    similarity_measure="NCC",
                    refine_pca_initializations=True,
                )
                transform_initializer.run()
                transform_sitk = transform_initializer.get_transform_sitk()
                transform_sitk = sitk.AffineTransform(
                    transform_sitk.GetInverse())

            else:
                self._registration_method.set_moving(self._reference)
                self._registration_method.set_fixed(self._stacks[i])
                self._registration_method.run()
                transform_sitk = self._registration_method.get_registration_transform_sitk()

            # Update position of stack
            self._stacks[i].update_motion_correction(transform_sitk)

##
# Class to perform Slice-To-Volume registration
# \date       2017-08-08 02:30:03+0100
#
class SliceToVolumeRegistration(RegistrationPipeline):

    ##
    # { constructor_description }
    # \date       2017-08-08 02:30:18+0100
    #
    # \param      self                 The object
    # \param      stacks               The stacks
    # \param      reference            The reference
    # \param      registration_method  Registration method, e.g.
    #                                  SimpleItkRegistration
    # \param      verbose              The verbose
    # \param      print_prefix         Print at each iteration at the
    #                                  beginning, string
    #
    def __init__(self,
                 stacks,
                 stacks_dis,
                 reference,
                 registration_method,
                 verbose=1,
                 print_prefix="",
                 interleave=2,
                 viewer=VIEWER,
                 ):
        RegistrationPipeline.__init__(
            self,
            stacks=stacks,
            stacks_dis=stacks_dis,
            reference=reference,
            registration_method=registration_method,
            verbose=verbose,
            viewer=viewer,
        )
        self._print_prefix = print_prefix
        self._interleave = interleave

    def set_print_prefix(self, print_prefix):
        self._print_prefix = print_prefix

    def _run(self):
        ph.print_title("Slice-to-Volume Registration")

        self._registration_method.set_moving(self._reference)

        for i, stack in enumerate(self._stacks):
            slices = stack.get_slices()
            slices_dis = self._stacks_dis[i].get_slices()

            transforms_sitk = {}

            for j, slice_j in enumerate(slices):

                txt = "%sSlice-to-Volume Registration -- " \
                      "Stack %d/%d (%s) -- Slice %d/%d" % (
                          self._print_prefix,
                          i + 1, len(self._stacks), stack.get_filename(),
                          j + 1, len(slices))
                if self._verbose:
                    ph.print_subtitle(txt)
                else:
                    ph.print_info(txt)

                self._registration_method.set_fixed(slice_j)
                self._registration_method.set_dis(slices_dis[j])
                self._registration_method.run()

                # Store information on registration transform
                transform_sitk = \
                    self._registration_method.get_registration_transform_sitk()
                transforms_sitk[slice_j.get_slice_number()] = transform_sitk

            # Update position of slice
            for slice in slices:
                slice_number = slice.get_slice_number()
                slice.update_motion_correction(transforms_sitk[slice_number])




class ReconstructionRegistrationPipeline(RegistrationPipeline):
    __metaclass__ = ABCMeta

    ##
    # Store variables relevant for two-step registration-reconstruction
    # pipeline.
    # \date       2017-10-16 10:30:39+0100
    #
    # \param      self                   The object
    # \param      verbose                Verbose output, bool
    # \param      stacks                 List of Stack objects
    # \param      reference              Reference as Stack object
    # \param      registration_method    Registration method, e.g.
    #                                    SimpleItkRegistration
    # \param      reconstruction_method  Reconstruction method, e.g. TK1
    # \param      alpha_range            Specify regularization parameter
    #                                    range, i.e. list [alpha_min,
    #                                    alpha_max]
    #
    def __init__(self,
                 verbose,
                 stacks,
                 stacks_dis,
                 reference,
                 registration_method,
                 reconstruction_method,
                 alphas,
                 viewer,

                 ):

        RegistrationPipeline.__init__(
            self,
            verbose=verbose,
            stacks=stacks,
            stacks_dis=stacks_dis,
            reference=reference,
            registration_method=registration_method,
            viewer=viewer,
        )

        self._reconstruction_method = reconstruction_method
        self._alphas = alphas

        self._reconstructions = [st.Stack.from_stack(
            self._reference,
            filename="Iter0_" + self._reference.get_filename())]
        self._computational_time_reconstruction = ph.get_zero_time()
        self._computational_time_registration = ph.get_zero_time()

    def get_iterative_reconstructions(self):
        return self._reconstructions

    def get_computational_time_reconstruction(self):
        return self._computational_time_reconstruction

    def get_computational_time_registration(self):
        return self._computational_time_registration


##
# Class to perform the two-step Slice-to-Volume registration and volumetric
# reconstruction iteratively
# \date       2017-08-08 02:30:43+0100
#
class TwoStepSliceToVolumeRegistrationReconstruction(
        ReconstructionRegistrationPipeline):
    def __init__(self,
                 stacks,
                 stacks_dis,
                 reference,
                 registration_method,
                 reconstruction_method,
                 alphas,
                 verbose,
                 cycles=3,
                 outlier_rejection=False,
                 threshold_measure="NCC",
                 thresholds=[0.6, 0.7, 0.8],
                 use_hierarchical_registration=False,
                 interleave=3,
                 viewer=VIEWER,
                 sigma_sda_mask=1.,
                 ):

        index=verbose
        verbose=0
        if len(alphas) != cycles - 1:
            raise ValueError(
                "Elements in alpha list must correspond to cycles-1")

        if outlier_rejection and len(thresholds) != cycles:
            raise ValueError(
                "Elements in outlier rejection threshold list must "
                "correspond to the number of cycles")

        ReconstructionRegistrationPipeline.__init__(
            self,
            stacks=stacks,
            stacks_dis=stacks_dis,
            reference=reference,
            registration_method=registration_method,
            reconstruction_method=reconstruction_method,
            alphas=alphas,
            viewer=viewer,
            verbose=verbose,
        )

        self._sigma_sda_mask = sigma_sda_mask

        self._cycles = cycles
        self._outlier_rejection = outlier_rejection
        self._threshold_measure = threshold_measure
        self._thresholds = thresholds
        self._use_hierarchical_registration = use_hierarchical_registration
        self._interleave = interleave
        self._index = index


    def _run(self):

        ph.print_title("Two-step S2V-Registration and SRR Reconstruction")

        s2vreg = SliceToVolumeRegistration(
            stacks=self._stacks,
            stacks_dis=self._stacks_dis,
            reference=self._reference,
            registration_method=self._registration_method,
            verbose=False,
            interleave=self._interleave,
        )

        reference = self._reference
        # self._cycles=3
        for cycle in range(0, self._cycles):
            s2vreg.set_reference(reference)
            s2vreg.set_print_prefix("Cycle %d/%d: " %
                                    (cycle + 1, self._cycles))
            s2vreg.run()

            self._computational_time_registration += \
                s2vreg.get_computational_time()

            # Reject misregistered slices
            if self._outlier_rejection:
                ph.print_subtitle("Slice Outlier Rejection (%s < %g)" % (
                    self._threshold_measure, self._thresholds[cycle]))
                outlier_rejector = outre.OutlierRejector(
                    stacks=self._stacks,
                    reference=self._reference,
                    threshold=self._thresholds[cycle],
                    measure=self._threshold_measure,
                    verbose=True,
                )
                outlier_rejector.run()
                self._reconstruction_method.set_stacks(
                    outlier_rejector.get_stacks())

                if len(self._stacks) == 0:
                    raise RuntimeError(
                        "All slices of all stacks were rejected "
                        "as outliers. Volumetric reconstruction is aborted.")

            # SRR step
            if cycle < self._cycles - 1:
                # ---------------- Perform Image Reconstruction ---------------
                ph.print_subtitle("Volumetric Image Reconstruction")
                if isinstance(
                    self._reconstruction_method,
                    sda.ScatteredDataApproximation
                ):
                    self._reconstruction_method.set_sigma(self._alphas[cycle])
                else:
                    self._reconstruction_method.set_alpha(self._alphas[cycle])
                # if cycle==0:
                #     self._reconstruction_method.set_index(0.8)
                # elif cycle==1:
                #     self._reconstruction_method.set_index(0.7)
                self._reconstruction_method.set_index(self._index)
                self._reconstruction_method.run()

                self._computational_time_reconstruction += \
                    self._reconstruction_method.get_computational_time()

                reference = self._reconstruction_method.get_reconstruction()

                # # ------------------ Perform Image Mask SDA -------------------
                ph.print_subtitle("Volumetric Image Mask Reconstruction")

                # -------------------- Store Reconstruction -------------------
                filename = "Iter%d_%s" % (
                    cycle + 1,
                    self._reconstruction_method.get_setting_specific_filename()
                )
                self._reconstructions.insert(0, st.Stack.from_stack(
                    reference, filename=filename))

                if self._verbose:
                    sitkh.show_stacks(self._reconstructions,
                                      segmentation=self._reference,
                                      viewer=self._viewer)



##
# Class to perform multi-component reconstruction
#
# Each stack is individually reconstructed at a given reconstruction space
# \date       2017-08-08 02:34:40+0100
#
class MultiComponentReconstruction(Pipeline):

    ##
    # Store information relevant for multi-component reconstruction
    # \date       2017-08-08 02:37:40+0100
    #
    # \param      self                   The object
    # \param      stacks                 The stacks
    # \param      reconstruction_method  The reconstruction method
    # \param      suffix                 Suffix added to filenames of each
    #                                    individual stack, string
    # \param      verbose                The verbose
    #
    def __init__(self,
                 stacks,
                 reconstruction_method,
                 suffix="_recon",
                 verbose=0,
                 viewer=VIEWER,
                 ):

        Pipeline.__init__(self, stacks=stacks, verbose=verbose, viewer=viewer)

        self._reconstruction_method = reconstruction_method
        self._reconstructions = None
        self._suffix = suffix

    def set_reconstruction_method(self, reconstruction_method):
        self._reconstruction_method = reconstruction_method

    def get_reconstruction_method(self):
        return self._reconstruction_method

    def set_suffix(self, suffix):
        self._suffix = suffix

    def get_suffix(self):
        return self._suffix

    def get_reconstructions(self):
        return [st.Stack.from_stack(stack) for stack in self._reconstructions]

    def _run(self):

        ph.print_title("Multi-Component Reconstruction")

        self._reconstructions = [None] * len(self._stacks)

        for i in range(0, len(self._stacks)):
            ph.print_subtitle("Multi-Component Reconstruction -- "
                              "Stack %d/%d" % (i + 1, len(self._stacks)))
            stack = self._stacks[i]
            self._reconstruction_method.set_stacks([stack])
            self._reconstruction_method.run()
            self._reconstructions[i] = st.Stack.from_stack(
                self._reconstruction_method.get_reconstruction())
            self._reconstructions[i].set_filename(
                stack.get_filename() + self._suffix)
