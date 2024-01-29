import os
import sys
import SimpleITK as sitk
import numpy as np
os.chdir(sys.path[0])

root_path= './data/image/'
data_path_list = os.listdir(root_path)
s_list=[2,1,0]
for path in data_path_list:
    direction_index = np.array([[0, 1, 2],
                                [3, 4, 5],
                                [6, 7, 8]])
    print(path)
    img=sitk.ReadImage(root_path+path)
    np_img=sitk.GetArrayFromImage(img).astype("float32")
    label = sitk.ReadImage(root_path.replace("image","mask")+path)
    np_label = sitk.GetArrayFromImage(label).astype("float32")


    np_img[np_label==0]=np_img.min()
    np_img=np_img-np_img.min()
    spacing = img.GetSpacing()
    image_shape = list(np_img.shape)
    s1 = image_shape.index(min(image_shape))
    s2 = image_shape.index(max(image_shape))
    sp1 = spacing.index(max(spacing))
    sp2 = spacing.index(min(spacing))
    if s2==1:
        T=s1, s2, s_list[s1 + s2 - 1]
    elif s2==2:
        T=s1, s_list[s1 + s2 - 1], s2
    elif s2==0 and s1==1:
        T=s1, s2, s_list[s1 + s2 - 1]
    elif s2==0 and s1==2:
        T=s1, s_list[s1 + s2 - 1], s2
    np_img = np_img.transpose(T)
    np_label = np_label.transpose(T)
    print(T)
    im = sitk.GetImageFromArray(np_img.astype("float32"))
    label = sitk.GetImageFromArray(np_label.astype("int16"))

    # im = sitk.GetImageFromArray(mask)
    if not os.path.exists(root_path.replace("image", "reo_image")):
        os.makedirs(root_path.replace("image", "reo_image"),exist_ok=True)
    if not os.path.exists(root_path.replace("image", "reo_mask")):
        os.makedirs(root_path.replace("image", "reo_mask"),exist_ok=True)
    im.SetSpacing((min(spacing),min(spacing),max(spacing)))
    label.SetSpacing((min(spacing),min(spacing),max(spacing)))

    #
    direction=img.GetDirection()
    Origin=img.GetOrigin()
    if T==(1,0,2):
        direction_index[:, [1, 2]]=direction_index[:, [2, 1]]
    elif T == (2,1,0):
        direction_index[:, [0, 2]] = direction_index[:, [2, 0]]
    elif T == (0,1,2):
        pass
    new_direction=(direction[direction_index[0,0]],direction[direction_index[0,1]],direction[direction_index[0,2]],
                   direction[direction_index[1,0]],direction[direction_index[1,1]],direction[direction_index[1,2]],
                   direction[direction_index[2,0]],direction[direction_index[2,1]],direction[direction_index[2,2]])
    im.SetDirection(new_direction)
    im.SetOrigin(Origin)
    label.SetDirection(new_direction)
    label.SetOrigin(Origin)
    #
    #
    sitk.WriteImage(im, root_path.replace("image", "reo_image") + path)
    sitk.WriteImage(label, root_path.replace("image", "reo_mask") + path)




