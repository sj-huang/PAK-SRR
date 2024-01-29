D:\Anaconda\envs\monaifbs\python.exe Y:\code\Fetal_pipeline\monaifbs\src\utils\custom_transform.py 

注释了第172-184行
            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            # d[key], _, new_affine = spacing_transform(
            #     data_array=d[key],
            #     affine=meta_data["affine"],
            #     mode=self.mode[idx],
            #     padding_mode=self.padding_mode[idx],
            #     align_corners=self.align_corners[idx],
            #     dtype=self.dtype[idx],
            # )

            # store the modified affine
            # meta_data["affine"] = new_affine