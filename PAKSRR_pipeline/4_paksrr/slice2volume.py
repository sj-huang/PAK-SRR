from simplereg.niftyreg_to_simpleitk_converter import \
    NiftyRegToSimpleItkConverter as nreg2sitk
import numpy as np
import SimpleITK as sitk
class S2V(object):
    def __init__(self, moving,fixed,dis):
        self._moving = moving
        self._fixed = fixed
        self._dis = dis


    def get_registration_transform_sitk(self):
        return self.transform_sitk
    def _get_warped_moving_sitk(self):
        return self.warped_moving_sitk

    def set_moving(self, moving):
        self._moving = moving.sitk
    def set_fixed(self, fixed):
        self._fixed = fixed.sitk
    def set_dis(self, dis):
        self._dis = dis.sitk

    def correlation(self,I, J, dis):
        if I.shape != J.shape:
            raise AssertionError("The inputs must be the same size.")
        u = I.reshape((I.shape[0] * I.shape[1], 1))
        v = J.reshape((J.shape[0] * J.shape[1], 1))
        d = dis.reshape((dis.shape[0] * dis.shape[1], 1))
        u = u - u.mean(keepdims=True)
        v = v - v.mean(keepdims=True)
        CC = (np.transpose(d).dot(u * v)) / (np.sqrt(np.transpose(u).dot(u)).dot(np.sqrt(np.transpose(v).dot(v))))
        # CC = (np.transpose(u).dot(v)) / (np.sqrt(np.transpose(u).dot(u)).dot(np.sqrt(np.transpose(v).dot(v))))
        # print("CC: ", CC)
        return CC
    def rotate(self,x, y, z):
        Rx = np.array([[1, 0, 0], [0, np.cos(x), np.sin(x)], [0, -np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z), np.cos(z), 0], [0, 0, 1]])
        R = np.dot(np.dot(Rx, Ry), Rz)
        return R
    def ngradient(self,fun, x, h=1e-3):
        g = np.zeros_like(x)
        # print(fun(x)[2])
        for k in range(len(x)):
            x1 = x.copy()
            x1[k] = x1[k] + h / 2
            x2 = x.copy()
            x2[k] = x2[k] - h / 2
            t = (fun(x1)[0] - fun(x2)[0]) / h
            g[k] = t
        # print("g: ", g)
        return g

    def rigid_corr(self,I, Im, x, return_transform=True):
        SCALING = 100
        R = self.rotate(x[0], x[1], x[2])
        Transform = np.zeros((4, 4))
        Transform[:3, :3] = R
        Transform[:3, 3] = x[3:] * SCALING
        Transform[-1, -1] = 1
        registration_transform_sitk = nreg2sitk.convert_regaladin_to_sitk_transform(
            Transform, dim=I.GetDimension())
        warped_moving_sitk = sitk.Resample(
            Im,
            I,
            registration_transform_sitk,
            eval("sitk.sitk%s" % ("Linear")),
            0.,
            I.GetPixelIDValue()
        )
        # sitk.WriteImage(warped_moving_sitk, "warp.nii.gz")
        self.warped_moving_sitk=warped_moving_sitk
        Im_t = sitk.GetArrayFromImage(warped_moving_sitk).squeeze()
        C = self.correlation(sitk.GetArrayFromImage(I).squeeze(), Im_t,sitk.GetArrayFromImage(self._dis).squeeze())
        if return_transform:
            return C, Im_t, Transform
        else:
            return C


    def run(self):
        x = np.array([0., 0., 0., 0., 0., 0.])
        mu = 0.0003
        fun = lambda x: self.rigid_corr(self._fixed, self._moving, x)
        for k in np.arange(100):
            # print("X: ", x)
            g = self.ngradient(fun, x)
            x += g * mu
        R = self.rotate(x[0], x[1], x[2])
        Transform = np.zeros((4, 4))
        Transform[:3, :3] = R
        Transform[:3, 3] = x[3:] * 100
        Transform[-1, -1] = 1
        self.transform_sitk = nreg2sitk.convert_regaladin_to_sitk_transform(
            Transform, dim=self._fixed.GetDimension())

