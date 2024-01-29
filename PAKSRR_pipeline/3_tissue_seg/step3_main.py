import os
import SimpleITK as sitk
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
def get_label(label):
    all_label = np.zeros((len(label), 2, label.shape[1], label.shape[2]))
    for i in range(len(label)):
        new_label = np.zeros((2, label.shape[1], label.shape[2]))

        new_label[0] = label[i]
        new_label[0][new_label[0] == 1] = 100
        new_label[0][new_label[0] == 4] = 100
        new_label[0][new_label[0] != 100] = 0
        new_label[0][new_label[0] == 100] = 1

        new_label[1] = label[i]
        new_label[1][new_label[1] == 2] = 100
        new_label[1][new_label[1] == 3] = 100
        new_label[1][new_label[1] == 5] = 100
        new_label[1][new_label[1] == 6] = 100
        # new_label[1][new_label[1] == 7] = 100
        new_label[1][new_label[1] != 100] = 0
        new_label[1][new_label[1] == 100] = 1

        all_label[i] = new_label
    return all_label
def get_high(im_tensor):
    fft_src_np = torch.fft.fftn(im_tensor, dim=(-4, -3, -2, -1))
    fshift = torch.fft.fftshift(fft_src_np).cuda()
    np_zero = torch.ones_like(fshift).cuda()
    b=im_tensor.shape[2]//2
    c=im_tensor.shape[3]//2
    s_b = b //20
    s_c = c //20
    np_zero[:, :, b - s_b:b + s_b, c - s_c:c + s_c] = 0
    fshift = fshift * np_zero
    ishift = torch.fft.ifftshift(fshift).cuda()
    iimg = abs(torch.fft.ifftn(ishift, dim=(-4,-3, -2, -1)).cuda())
    # iimg=torch.cat((im_tensor,iimg),1)
    return iimg
def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    C: int = len(seg)
    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res
root_path="../data/reo_image"
paths_list=os.listdir(root_path)
model=torch.load("./save_0.05/U.pth").cuda()
if not os.path.exists(root_path.replace("reo_image", "reo_label")):
    os.makedirs(root_path.replace("reo_image", "reo_label"))
if not os.path.exists(root_path.replace("reo_image", "distance")):
    os.makedirs(root_path.replace("reo_image", "distance"))
for path in paths_list:
    path=os.path.join(root_path,path)
    img_sitk=sitk.ReadImage(path)
    np_img=sitk.GetArrayFromImage(img_sitk)
    tensor_img=torch.tensor(np_img).unsqueeze(1).cuda()
    tensor_img=(tensor_img-tensor_img.min())/(tensor_img.max()-tensor_img.min())
    tensor_img=get_high(tensor_img)
    pre_img,_=model(tensor_img)
    res = pre_img.detach().cpu().numpy()
    mask = np.argmax(res, axis=1)
    mask = np.squeeze(mask)
    mask = mask + 1
    mask[mask == 7] = 0
    label = sitk.GetImageFromArray(mask.astype("int16"))
    label.CopyInformation(img_sitk)
    sitk.WriteImage(label,path.replace("reo_image","reo_label"))


    img = get_label(mask)
    dis_final = np.zeros_like(mask).astype("float64")
    for j in range(1):
        dis_label = abs(one_hot2dist(img[:, j]))
        img_label = mask.copy()
        dis_label_boundary = (dis_label.max() - dis_label) / dis_label.max()
        for i in range(mask.shape[0]):
            if dis_label_boundary[i].sum() == dis_label_boundary.shape[1] * dis_label_boundary.shape[2]:
                dis_label_boundary[i] = 0
        for i in range(mask.shape[1]):
            if dis_label_boundary[:, i].sum() == dis_label_boundary.shape[0] * dis_label_boundary.shape[2]:
                dis_label_boundary[:, i] = 0
        for i in range(mask.shape[2]):
            if dis_label_boundary[:, :, i].sum() == dis_label_boundary.shape[1] * dis_label_boundary.shape[0]:
                dis_label_boundary[:, :, i] = 0
        # dis_label_boundary[img_label == 0] = 0
        dis_final += dis_label_boundary
        # dis_final = dis_final/2
        # dis_final = np.exp(np.exp(dis_final) - 1)
        dis_final = np.exp(np.exp(np.exp(dis_final) - 1) - 1)
        dis_final = dis_final / dis_final.max() + 0.3

        label = sitk.GetImageFromArray(dis_final)
        label.CopyInformation(img_sitk)
        sitk.WriteImage(label, path.replace("reo_image", "distance"))