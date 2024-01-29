from abc import ABCMeta, abstractmethod
import itk
import niftymic.reconstruction.linear_operators as lin_op
import numpy as np
import SimpleITK as sitk
import ants
import nsol.linear_operators as linop
# import nsol.tikhonov_linear_solver as tk
import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
import datetime
from abc import ABCMeta, abstractmethod
from skimage.exposure import match_histograms
import os
import sys
import scipy

# from nsol.linear_solver import LinearSolver
from nsol.definitions import EPS
from nsol.loss_functions import LossFunctions as lf
import least_square
# from niftymic.reconstruction.solver import Solver
# Allowed data loss functions
DATA_LOSS = ['linear', 'soft_l1', 'huber', 'cauchy', 'arctan']

import os
import sys
from abc import ABCMeta, abstractmethod

import pysitk.python_helper as ph

from nsol.solver import Solver as solv
from nsol.loss_functions import LossFunctions as lf

class LinearSolver(solv):
    __metaclass__ = ABCMeta

    def __init__(self,
                 A,
                 A_adj,
                 b,
                 x0,
                 alpha,
                 x_scale,
                 data_loss,
                 data_loss_scale,
                 minimizer,
                 iter_max,
                 verbose):

        solv.__init__(self, x0=x0, x_scale=x_scale, verbose=verbose)

        self._A = A
        self._A_adj = A_adj
        self._b = b / self._x_scale
        self._alpha = float(alpha)
        self._data_loss = data_loss
        self._data_loss_scale = float(data_loss_scale)
        self._minimizer = minimizer
        self._iter_max = iter_max

    def get_A(self):
        return self._A

    def get_A_adj(self):
        return self._A_adj

    def get_b(self):
        return np.array(self._b) * self._x_scale

    def set_alpha(self, alpha):
        self._alpha = alpha

    def get_alpha(self):
        return self._alpha

    def set_index(self, index):
        self._index = index
    def get_index(self):
        return self._index

    def set_data_loss(self, data_loss):
        if data_loss not in lf.get_loss.keys():
            raise ValueError("data_loss must be in " +
                             str(lf.get_loss.keys()))
        self._data_loss = data_loss

    def get_data_loss(self):
        return self._data_loss

    def set_data_loss_scale(self, data_loss_scale):
        self._data_loss_scale = data_loss_scale

    def get_data_loss_scale(self):
        return self._data_loss_scale

    def set_minimizer(self, minimizer):
        self._minimizer = minimizer

    def get_minimizer(self):
        return self._minimizer

    def set_iter_max(self, iter_max):
        self._iter_max = iter_max

    def get_iter_max(self):
        return self._iter_max

    def get_total_cost(self):
        data_cost = self.get_cost_data_term()
        regularization_cost = self.get_cost_regularization_term()
        return data_cost + self._alpha * regularization_cost

    def get_cost_data_term(self):
        return self._get_cost_data_term(self._x)

    def get_ell2_cost_data_term(self):
        return self._get_ell2_cost_data_term(self._x)

    def get_cost_regularization_term(self):
        return self._get_cost_regularization_term(self._x)

    def print_statistics(self, fmt="%.3e"):

        cost_data = self.get_cost_data_term()
        cost_data_ell2 = self.get_ell2_cost_data_term()
        cost_regularizer = self.get_cost_regularization_term()

        ph.print_subtitle("Summary Optimization")
        ph.print_info("Computational time: %s" %
                      (self.get_computational_time()))
        ph.print_info("Cost data term (f, loss=%s, scale=%g): " %
                      (self._data_loss, self._data_loss_scale) +
                      fmt % (cost_data) +
                      " (ell2-cost: " + fmt % (cost_data_ell2) + ")")
        ph.print_info(
            "Cost regularization term (g): " +
            fmt % (cost_regularizer))
        ph.print_info(
            "Total cost (f + alpha g; alpha = %g" % (self._alpha) + "): " +
            fmt % (cost_data + self._alpha * cost_regularizer))

    def _get_cost_data_term(self, x):

        residual = self._A(x) - self._b
        cost = 0.5 * np.sum(
            lf.get_loss[self._data_loss](f2=residual ** 2,
                                         f_scale=self._data_loss_scale))

        return cost

    def _get_ell2_cost_data_term(self, x):

        residual = self._A(x) - self._b
        cost = 0.5 * np.sum(residual ** 2)

        return cost

    def _get_gradient_cost_data_term(self, x):

        residual = self._A(x) - self._b

        grad = self._A_adj(
            lf.get_gradient_loss[self._data_loss](f2=residual ** 2,
                                                  f_scale=self._data_loss_scale
                                                  ) * residual)

        return grad

    @abstractmethod
    def _get_cost_regularization_term(self, x):
        pass




class Solver(object):
    __metaclass__ = ABCMeta

    def __init__(self,
                 stacks,
                 stacks_dis,
                 reconstruction,
                 atlas,
                 alpha_cut,
                 alpha,
                 iter_max,
                 minimizer,
                 x_scale,
                 data_loss,
                 data_loss_scale,
                 huber_gamma,
                 deconvolution_mode,
                 predefined_covariance,
                 verbose,
                 image_type=itk.Image.D3,
                 use_masks=True,
                 ):

        # Initialize variables
        self._stacks = stacks
        self._stacks_dis = stacks_dis
        self._reconstruction = reconstruction
        self.atlas = atlas

        # Cut-off distance for Gaussian blurring filter
        self._alpha_cut = alpha_cut

        self._deconvolution_mode = deconvolution_mode
        self._predefined_covariance = predefined_covariance
        self._linear_operators = lin_op.LinearOperators(
            deconvolution_mode=self._deconvolution_mode,
            predefined_covariance=self._predefined_covariance,
            alpha_cut=self._alpha_cut,
            image_type=image_type
        )

        # Settings for solver
        self._alpha = alpha
        self._iter_max = iter_max

        self._use_masks = use_masks

        self._minimizer = minimizer
        self._data_loss = data_loss
        self._data_loss_scale = data_loss_scale

        if x_scale == "max":
            self._x_scale = sitk.GetArrayFromImage(
                reconstruction.sitk).max()

            # Avoid zero in case zero-image is given
            if self._x_scale == 0:
                self._x_scale = 1
        else:
            self._x_scale = x_scale

        self._huber_gamma = huber_gamma

        self._verbose = verbose

        # Allocate variables containing information about statistics of
        # reconstruction
        self._computational_time = None
        self._residual_ell2 = None
        self._residual_prior = None

        # Create PyBuffer object for conversion between NumPy arrays and ITK
        # images
        self._itk2np = itk.PyBuffer[image_type]

        # -----------------------------Set helpers-----------------------------
        self._N_stacks = len(self._stacks)

        # Compute total amount of pixels for all slices
        self._N_total_slice_voxels = 0
        for i in range(0, self._N_stacks):
            N_stack_voxels = np.array(self._stacks[i].sitk.GetSize()).prod()
            self._N_total_slice_voxels += N_stack_voxels

        # Extract information ready to use for itk image conversion operations
        self._reconstruction_shape = sitk.GetArrayFromImage(
            self._reconstruction.sitk).shape

        # Compute total amount of voxels of x:
        self._N_voxels_recon = np.array(
            self._reconstruction.sitk.GetSize()).prod()

    def set_stacks(self, stacks):
        self._stacks = stacks

        # Update helpers
        self._N_stacks = len(self._stacks)

        # Compute total amount of pixels for all slices
        self._N_total_slice_voxels = 0
        for i in range(0, self._N_stacks):
            N_stack_voxels = np.array(self._stacks[i].sitk.GetSize()).prod()
            self._N_total_slice_voxels += N_stack_voxels

    def set_use_masks(self, use_masks):
        self._use_masks = use_masks

    def set_reconstruction(self, reconstruction):
        self._reconstruction = reconstruction

        # Extract information ready to use for itk image conversion operations
        self._reconstruction_shape = sitk.GetArrayFromImage(
            self._reconstruction.sitk).shape

        # Compute total amount of voxels of x:
        self._N_voxels_recon = np.array(
            self._reconstruction.sitk.GetSize()).prod()

    def set_alpha(self, alpha):
        self._alpha = alpha

    def get_alpha(self):
        return self._alpha

    def set_index(self, index):
        self._index = index
    def get_index(self):
        return self._index

    def set_iter_max(self, iter_max):
        self._iter_max = iter_max

    def get_iter_max(self):
        return self._iter_max

    def set_minimizer(self, minimizer):
        self._minimizer = minimizer

    def get_minimizer(self):
        return self._minimizer

    def set_data_loss(self, data_loss):
        if data_loss not in DATA_LOSS:
            raise ValueError("Loss function must be in " + str(DATA_LOSS))
        self._data_loss = data_loss

    def set_huber_gamma(self, huber_gamma):
        self._huber_gamma = huber_gamma

    def get_huber_gamma(self):
        return self._huber_gamma

    def set_verbose(self, verbose):
        self._verbose = verbose

    def get_verbose(self):
        return self._verbose

    def run(self):

        # Run solver specific reconstruction
        self._run()

    # Get current estimate of reconstruction
    #  \return current estimate of reconstruction, instance of Stack
    def get_reconstruction(self):
        return self._reconstruction

    # Get cut-off distance
    #  \return scalar value
    def get_alpha_cut(self):
        return self._alpha_cut

    # Get computational time for reconstruction
    #  \return computational time in seconds
    def get_computational_time(self):
        return self._computational_time

    def get_A(self):
        return lambda x: self._MA(x)

    def get_A_adj(self):
        return lambda x: self._A_adj_M(x)

    def get_b(self):
        return self._get_M_y()

    def get_x0(self):
        return sitk.GetArrayFromImage(self._reconstruction.sitk).flatten()
    def get_reconstruct_x(self):
        return self._reconstruction
    def get_atlas(self):
        return self.atlas
    def get_x_scale(self):
        return self._x_scale

    @abstractmethod
    def get_setting_specific_filename(self, prefix=""):
        pass

    @abstractmethod
    def _run(self):
        pass

    @abstractmethod
    def get_solver(self):
        pass

    def get_predefined_covariance(self):
        return self._predefined_covariance

    def _get_M_y(self):
        My = np.zeros(self._N_total_slice_voxels)
        i_min = 0
        for i, stack in enumerate(self._stacks):
            slices = stack.get_slices()
            stacks_dis=self._stacks_dis[i]
            slices_dis = stacks_dis.get_slices()
            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()
            for j, slice_j in enumerate(slices):
                i_max = i_min + N_slice_voxels
                if self._use_masks:
                    slice_itk = self._linear_operators.M_itk(
                        slice_j.itk, slice_j.itk_mask)
                else:
                    slice_itk = slice_j.itk
                slice_nda_vec = self._itk2np.GetArrayFromImage(slice_itk).flatten()
                slice_nda_vec_dis = self._itk2np.GetArrayFromImage(slices_dis[j].itk).flatten()

                My[i_min:i_max] = slice_nda_vec*slice_nda_vec_dis
                i_min = i_max
        return My



    def _Mk_Ak(self, reconstruction_itk, slice_k):

        # Get slice spacing relevant for Gaussian blurring estimate
        in_plane_res = slice_k.get_inplane_resolution()
        slice_thickness = slice_k.get_slice_thickness()
        slice_spacing = np.array([in_plane_res, in_plane_res, slice_thickness])

        # Compute A_k x
        Ak_reconstruction_itk = self._linear_operators.A_itk(
            reconstruction_itk, slice_k.itk, slice_spacing)

        if not self._use_masks:
            return Ak_reconstruction_itk

        # Compute M_k A_k x
        Ak_reconstruction_itk = self._linear_operators.M_itk(
            Ak_reconstruction_itk, slice_k.itk_mask)

        return Ak_reconstruction_itk

    def _Ak_adj_Mk(self, slice_itk, slice_k):

        # Compute M_k y_k
        if self._use_masks:
            Mk_slice_itk = self._linear_operators.M_itk(
                slice_itk, slice_k.itk_mask)
        else:
            Mk_slice_itk = slice_itk

        # Get slice spacing relevant for Gaussian blurring estimate
        in_plane_res = slice_k.get_inplane_resolution()
        slice_thickness = slice_k.get_slice_thickness()
        slice_spacing = np.array([in_plane_res, in_plane_res, slice_thickness])

        # Compute A_k^* M_k y_k
        Mk_slice_itk = self._linear_operators.A_adj_itk(
            Mk_slice_itk, self._reconstruction.itk, slice_spacing)

        return Mk_slice_itk


    def _MA(self, reconstruction_nda_vec):

        # Convert reconstruction data array back to itk.Image object
        x_itk = self._get_itk_image_from_array_vec(
            reconstruction_nda_vec, self._reconstruction.itk)

        # Allocate memory
        MA_x = np.zeros(self._N_total_slice_voxels)

        # Define index for first voxel of first slice within array
        i_min = 0

        for i, stack in enumerate(self._stacks):
            slices = stack.get_slices()
            stack_dis=self._stacks_dis[i]
            slices_dis=stack_dis.get_slices()
            # Get number of voxels of each slice in current stack
            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()

            for j, slice_j in enumerate(slices):

                # Define index for last voxel to specify current slice
                # (exclusive)
                i_max = i_min + N_slice_voxels

                # Compute M_k A_k y_k
                slice_itk = self._Mk_Ak(x_itk, slice_j)

                slice_nda = self._itk2np.GetArrayFromImage(slice_itk)
                # slice_nda = self.refine_s2v(slice_itk,slices_dis[j].itk,slice_j)
                slice_nda_dis = self._itk2np.GetArrayFromImage(slices_dis[j].itk)

                # Fill corresponding elements
                MA_x[i_min:i_max] = slice_nda.flatten()*slice_nda_dis.flatten()

                # Define index for first voxel to specify subsequent slice
                # (inclusive)
                i_min = i_max

        return MA_x

    def _A_adj_M(self, stacked_slices_nda_vec):

        # Allocate memory
        A_adj_M_y = np.zeros(self._N_voxels_recon)

        # Define index for first voxel of first slice within array
        i_min = 0

        for i, stack in enumerate(self._stacks):
            slices = stack.get_slices()

            # Get number of voxels of each slice in current stack
            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()

            for j, slice_j in enumerate(slices):

                # Define index for last voxel to specify current slice
                # (exlusive)
                i_max = i_min + N_slice_voxels

                # Extract 1D corresponding to current slice and convert it to
                # itk.Object
                slice_itk = self._get_itk_image_from_array_vec(
                    stacked_slices_nda_vec[i_min:i_max], slice_j.itk)

                # Apply A_k' M_k on current slice
                Ak_adj_Mk_slice_itk = self._Ak_adj_Mk(slice_itk, slice_j)
                Ak_adj_Mk_slice_nda_vec = self._itk2np.GetArrayFromImage(
                    Ak_adj_Mk_slice_itk).flatten()

                # Add contribution
                A_adj_M_y += Ak_adj_Mk_slice_nda_vec

                # Define index for first voxel to specify subsequent slice
                # (inclusive)
                i_min = i_max

        return A_adj_M_y

    def _get_itk_image_from_array_vec(self, nda_vec, image_itk_ref):

        shape_nda = np.array(
            image_itk_ref.GetLargestPossibleRegion().GetSize())[::-1]

        image_itk = self._itk2np.GetImageFromArray(nda_vec.reshape(shape_nda))
        image_itk.SetOrigin(image_itk_ref.GetOrigin())
        image_itk.SetSpacing(image_itk_ref.GetSpacing())
        image_itk.SetDirection(image_itk_ref.GetDirection())

        return image_itk


class TikhonovSolver(Solver):

    def __init__(self,
                 stacks,
                 stacks_dis,
                 reconstruction,
                 atlas,
                 alpha_cut=3,
                 alpha=0.03,
                 iter_max=10,
                 reg_type="TK1",
                 minimizer="lsmr",
                 deconvolution_mode="full_3D",
                 x_scale="max",
                 data_loss="linear",
                 data_loss_scale=1,
                 huber_gamma=1.345,
                 predefined_covariance=None,
                 use_masks=True,
                 verbose=1,
                 ):

        # Run constructor of superclass
        Solver.__init__(self,
                        stacks=stacks,
                        stacks_dis=stacks_dis,
                        reconstruction=reconstruction,
                        atlas=atlas,
                        alpha_cut=alpha_cut,
                        alpha=alpha,
                        iter_max=iter_max,
                        minimizer=minimizer,
                        deconvolution_mode=deconvolution_mode,
                        x_scale=x_scale,
                        data_loss=data_loss,
                        data_loss_scale=data_loss_scale,
                        huber_gamma=huber_gamma,
                        predefined_covariance=predefined_covariance,
                        verbose=verbose,
                        use_masks=use_masks,
                        )

        # Settings for optimizer
        self._reg_type = reg_type

    def set_regularization_type(self, reg_type):
        self._reg_type = reg_type

    def get_regularization_type(self):
        return self._reg_type

    def get_setting_specific_filename(self, prefix="SRR_"):

        # Build filename
        filename = prefix
        filename += "stacks" + str(len(self._stacks))
        if self._alpha > 0:
            filename += "_" + self._reg_type
        filename += "_" + self._minimizer
        if self._data_loss not in ["linear"]:
            filename += "_" + self._data_loss
            if self._data_loss in ["huber"]:
                filename += str(self._huber_gamma)
            filename += "_fscale%g" % self._data_loss_scale
        filename += "_alpha" + str(self._alpha)
        filename += "_itermax" + str(self._iter_max)

        # Replace dots by 'p'
        filename = filename.replace(".", "p")

        return filename

    def get_solver(self):
        if self._reg_type not in ["TK0", "TK1"]:
            raise ValueError(
                "Error: regularization type can only be either 'TK0' or 'TK1'")

        # Get operators
        A = self.get_A()
        A_adj = self.get_A_adj()
        b = self.get_b()
        x0 = self.get_x0()
        reconstruct_x=self.get_reconstruct_x()
        atlas=self.get_atlas()
        x_scale = self.get_x_scale()
        index=self.get_index()

        if self._reg_type == "TK0":
            B = lambda x: x.flatten()
            B_adj = lambda x: x.flatten()

        elif self._reg_type == "TK1":
            spacing = np.array(self._reconstruction.sitk.GetSpacing())
            linear_operators = linop.LinearOperators3D(spacing=spacing)
            grad, grad_adj = linear_operators.get_gradient_operators()

            X_shape = self._reconstruction_shape
            Z_shape = grad(x0.reshape(*X_shape)).shape

            B = lambda x: grad(x.reshape(*X_shape)).flatten()
            B_adj = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

        # Set up solver
        solver = TikhonovLinearSolver(
            index,
            reconstruct_x,
            atlas,
            A=A,
            A_adj=A_adj,
            B=B,
            B_adj=B_adj,
            b=b,
            x0=x0,
            x_scale=x_scale,
            alpha=self._alpha,
            data_loss=self._data_loss,
            data_loss_scale=self._data_loss_scale,
            verbose=self._verbose,
            minimizer=self._minimizer,
            iter_max=self._iter_max,
            bounds=(0, np.inf),
        )
        return solver

    def _run(self):

        solver = self.get_solver()

        self._print_info_text()

        # Run reconstruction
        solver.run()

        # Get computational time
        self._computational_time = solver.get_computational_time()

        # After reconstruction: Update member attribute
        self._reconstruction.itk = self._get_itk_image_from_array_vec(
            solver.get_x(), self._reconstruction.itk)
        self._reconstruction.sitk = sitkh.get_sitk_from_itk_image(
            self._reconstruction.itk)

    def _print_info_text(self):

        ph.print_subtitle("Tikhonov Solver:")
        ph.print_info("Chosen regularization type: ", newline=False)
        if self._reg_type in ["TK0"]:
            print("Zeroth-order Tikhonov")

        else:
            print("First-order Tikhonov")

        if self._deconvolution_mode in ["only_in_plane"]:
            ph.print_info("(Only in-plane deconvolution is performed)")

        elif self._deconvolution_mode in ["predefined_covariance"]:
            ph.print_info("(Predefined covariance used: cov = %s)"
                          % (np.diag(self._predefined_covariance)))

        if self._data_loss in ["huber"]:
            ph.print_info("Loss function: %s (gamma = %g)" %
                          (self._data_loss, self._huber_gamma))
        else:
            ph.print_info("Loss function: %s" % (self._data_loss))

        if self._data_loss != "linear":
            ph.print_info("Loss function scale: %g" % (self._data_loss_scale))

        ph.print_info("Regularization parameter: " + str(self._alpha))
        ph.print_info("Minimizer: " + self._minimizer)
        ph.print_info(
            "Maximum number of iterations: " + str(self._iter_max))

class TikhonovLinearSolver(LinearSolver):

    def __init__(self,
                 index,
                 reconstruct_x,
                 atlas,
                 A, A_adj,
                 B, B_adj,
                 b,
                 x0,
                 alpha=0.01,
                 b_reg=0,
                 data_loss="linear",
                 data_loss_scale=1,
                 # minimizer="lsmr",
                 minimizer="SLSQP",
                 iter_max=10,
                 x_scale=1,
                 verbose=0,
                 bounds=(0, np.inf)):

        super(self.__class__, self).__init__(
            A=A, A_adj=A_adj, b=b, x0=x0, alpha=alpha, iter_max=iter_max,
            minimizer=minimizer, data_loss=data_loss,
            data_loss_scale=data_loss_scale, x_scale=x_scale, verbose=verbose)

        self._B = B
        self._B_adj = B_adj
        self._b_reg = b_reg / self._x_scale
        self._bounds = bounds
        self.reconstruct_x = reconstruct_x
        self.atlas = atlas
        self.index=index

    def get_B(self):
        return self._B

    def get_B_adj(self):
        return self._B_adj

    def get_b_reg(self):
        return self._b_reg * self._x_scale

    def _run(self):

        if self._minimizer == "lsmr" and self._data_loss != "linear":
            raise ValueError(
                "lsmr solver cannot be used with non-linear data loss")

        elif self._minimizer == "lsq_linear" and self._data_loss != "linear":
            raise ValueError(
                "lsq_linear solver cannot be used with non-linear data loss")

        # Monitor output
        if self._observer is not None:
            self._observer.add_x(self.get_x())

        # Get augmented linear system
        A, b = self._get_augmented_linear_system(self._alpha,self.reconstruct_x,self.atlas)
        # Define residual function and its Jacobian
        residual = lambda x: A*x - b
        jacobian_residual = lambda x: A

        # Clip to bounds
        if self._bounds is not None:
            self._x0 = np.clip(self._x0, self._bounds[0], self._bounds[1])

        # Use scipy.sparse.linalg.lsmr
        if self._minimizer == "lsmr" and self._data_loss == "linear":

            # Linear least-squares method
            self._x = scipy.sparse.linalg.lsmr(A, b,maxiter=self._iter_max,show=self._verbose,atol=0,btol=0)[0]
            if self._bounds is not None:
                # Clip to bounds
                self._x = np.clip(self._x, self._bounds[0], self._bounds[1])

        # Use scipy.optimize.lsq_linear
        elif self._minimizer == "lsq_linear" and self._data_loss == "linear":

            # linear least-squares problem with bounds on the variables
            self._x = scipy.optimize.lsq_linear(
                A, b,
                max_iter=self._iter_max,
                lsq_solver='lsmr',
                lsmr_tol='auto',
                bounds=self._bounds,
                verbose=2*self._verbose,
            ).x

        # Use scipy.optimize.least_squares
        elif self._minimizer == "least_squares":
            # BE AWARE:
            # Loss function is applied to both data and regularization term!
            # Remark: it seems that least_squares solver does not cope with
            # non-linear loss. Maybe because of the use of sparse linear
            # operator?

            # Non-linear least squares algorithm
            self._x = scipy.optimize.least_squares(fun=residual,jac=jacobian_residual,jac_sparsity=jacobian_residual,x0=self._x0,tr_solver='lsmr',bounds=self._bounds,loss=self._data_loss,f_scale=self._data_loss_scale,max_nfev=self._iter_max,verbose=2*self._verbose,).x
        # method="trf",
        # Use scipy.optimize.minimize
        else:
            bounds = [[self._bounds[0], self._bounds[1]]] * self._x0.size
            # cost_=self._get_cost_data_term(self._x)
            # Define cost function and its Jacobian
            if self._alpha > EPS:
                cost = lambda x: \
                    self._get_cost_data_term(x) + \
                    self._alpha * self._get_cost_regularization_term(x)
                grad_cost = lambda x: \
                    self._get_gradient_cost_data_term(x) + \
                    self._alpha * \
                    self._get_gradient_cost_regularization_term(x)

            else:
                cost = lambda x: self._get_cost_data_term(x)
                grad_cost = lambda x: self._get_gradient_cost_data_term(x)

            self._x = scipy.optimize.minimize(
                method=self._minimizer,
                fun=cost,
                jac=grad_cost,
                x0=self._x0,
                bounds=bounds,
                options={'maxiter': self._iter_max, 'disp': self._verbose}).x

        # Monitor output
        if self._observer is not None:
            self._observer.add_x(self.get_x())
    def get_correlation(self,move_img, reg_img_1,reg_img_2):
        I=move_img.numpy()
        J=reg_img_1.numpy()
        K=reg_img_2.numpy()
        u = I.reshape(-1, 1)
        v = J.reshape(-1, 1)
        u = u - u.mean(keepdims=True)
        v = v - v.mean(keepdims=True)
        NCC_1 = np.mean((np.multiply(u, v)) / (np.std(u) * (np.std(v))))

        w = K.reshape(-1, 1)
        w = w - w.mean(keepdims=True)
        NCC_2 = np.mean((np.multiply(u, w)) / (np.std(u) * (np.std(w))))

        if NCC_1>=NCC_2:
            reg_img=reg_img_1
        else:
            reg_img=reg_img_2

        return reg_img

    def atlas2ours_transform(self,atlas,reconstruct_x):
        fix_img = ants.image_read(atlas)
        rec=sitk.GetArrayFromImage(reconstruct_x.sitk)
        rec_m=sitk.GetArrayFromImage(reconstruct_x.sitk_mask)
        rec[rec_m==0]=0
        rec=sitk.GetImageFromArray(rec)
        rec.CopyInformation(reconstruct_x.sitk)
        sitk.WriteImage(rec,"reconstruction_x.nii.gz")
        move_img = ants.image_read("reconstruction_x.nii.gz")
        # 配准
        outs_1 = ants.registration(fix_img, move_img, type_of_transforme='SyN')
        outs_2 = ants.registration(move_img, fix_img, type_of_transforme='SyN')

        # 获取配准后的数据，并保存
        reg_img_1 = outs_1['warpedfixout']
        reg_img_2 = outs_2['warpedmovout']
        reg_img = self.get_correlation(move_img, reg_img_1,reg_img_2)


        ants.image_write(reg_img, "warp_out.nii.gz")




        out=sitk.ReadImage("warp_out.nii.gz")
        out=sitk.GetArrayFromImage(out)

        # ref = sitk.ReadImage("reconstruction_x.nii.gz")
        # ref = sitk.GetArrayFromImage(ref)
        #
        # out = match_histograms(ref, out)

        out=(self.index*out.flatten())/(out.max())
        # os.remove("warp_out.nii.gz")
        # os.remove("reconstruction_x.nii.gz")
        return out
    def _get_augmented_linear_system(self, alpha, reconstruct_x, atlas):

        # With regularization
        if alpha > EPS:
            atlas2ours = self.atlas2ours_transform(atlas, reconstruct_x)
            # Define forward and backward operators
            A_fw = lambda x: self._A_augmented(x, np.sqrt(alpha))
            A_bw = lambda x: self._A_augmented_adj(x, np.sqrt(alpha))

            # Define right-hand side b
            b = np.zeros(A_fw(self._x0).size)
            b[0:self._b.size] = self._b
            b[self._b.size:] = np.sqrt(alpha) * self._B(atlas2ours)

        # Without regularization
        else:

            # Define forward and backward operators
            A_fw = lambda x: self._A(x)
            A_bw = lambda x: self._A_adj(x)

            # Define right-hand side b
            b = self._b

        # Construct (sparse) linear operator A
        A = scipy.sparse.linalg.LinearOperator(
            shape=(b.size, self._x0.size),
            matvec=A_fw,
            rmatvec=A_bw)

        # return A, b, w
        return A, b

    def _A_augmented(self, x, sqrt_alpha):

        A_augmented_x = np.concatenate((
            self._A(x),
            sqrt_alpha * self._B(x)))


        return A_augmented_x

    def _A_augmented_adj(self, x, sqrt_alpha):

        x_upper = x[:self._b.size]
        x_lower = x[self._b.size:]

        A_augmented_adj_x = self._A_adj(x_upper) + \
            sqrt_alpha * self._B_adj(x_lower)

        return A_augmented_adj_x

    def _get_cost_regularization_term(self, x):
        return 0.5 * np.sum(self._B(x)**2)

    def _get_gradient_cost_regularization_term(self, x):
        return self._B_adj(self._B(x))















































