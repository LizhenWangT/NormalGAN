# NormalGAN
**NormalGAN: Learning Detailed 3D Human from a Single RGB-D Image (ECCV 2020)**<br>
Lizhen Wang, Xiaochen Zhao, [Tao Yu](https://ytrock.com/), [Yebin Liu](http://www.liuyebin.com/) and Songtao Wang<br>
We propose NormalGAN, a fast adversarial learning-based method to reconstruct the complete and detailed 3D human from a single RGB-D image.

[Project Page](http://www.liuyebin.com/NormalGan/normalgan.html) [paper](https://export.arxiv.org/abs/2007.15340)

Note: As we can not release our dataset, we do not release the training code. Now you can try NormalGAN on 3D dataset like THUman2.0(https://github.com/ytrock/THuman2.0-Dataset). If you are interested with our training code, please fell free to send an e-mail to Lizhen Wang(wlz18@mails.tsinghua.edu.cn).

# Changelog
2020.08.11 Release the test code and pretrained models

# Requirements
The code and released model were trained on
 * Ubuntu 16.04 & Python 3.5.2
 * Pytorch 1.12
 * trimesh 3.2.36
 * Python-opencv 3.4.3 (for Python-opencv >= 4.0, please change Line 266 of **NormalGAN/src/ops.py** to `contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)`)

Optional for src/redner.py:
 * Pyrender 0.1.32

Recommend for Kinect v2 python implement
 * [pylibfreenect2](https://github.com/r9y9/pylibfreenect2)


# Pretrained models
Download the pretrained models in [Pretrained models](https://drive.google.com/file/d/1EJfDeow-yUcJm85zaKnZ3HsWGXk0Auze/view?usp=sharing).

Put the pretrained models in **NormalGAN/model/pretrained/**

# Testing on the given images
Generate the csv file for the demo images in **NormalGAN/datasets/testdata**. 
```
cd NormalGAN/datasets
python data_utils/createcsv.py testdata/ test.csv
cd ..
```

Run the **NormalGAN/test_offline.sh** file (which occupies about 3.5-GB GPU memory).
```
sh test_offline.sh test.csv testdata
```

Results are shown in **NormalGAN/results/testdata/ply**. You can also use Poisson Reconstruction for better performance of the edge area.

# Testing on your own data
Please note that, NormalGAN simulate noise for Kinect v2 (or similar ToF depth cameras), the image resolution should be **(512,424)**. Please change the camera intrinsics and image resolution in **NormalGAN/src/test_offline.py**.
 * You should first **apply the body mask** for your RGB-D images before testing them with NormalGAN. 
 * Create **NormalGAN/datasets/your_data/color** & **NormalGAN/datasets/your_data/depth** folders, put your own RGB-D data into the folders (use the same filename for your RGB-D image pairs, eg. NormalGAN/datasets/your_data/color/1.png & NormalGAN/datasets/your_data/depth/1.png). 
 * Use **NormalGAN/data_utils/createcsv.py** to create csv file for your own data.
```
cd NormalGAN/datasets
python data_utils/createcsv.py your_data_folder_name/ your_csv_file_name.csv
cd ..
```
 * Run **NormalGAN/test_offline.sh** to test your data.
```
sh test_offline.sh your_csv_file_name.csv your_save_folder_name
```


# Citation
```
@inproceedings{wang2020normalgan,
title={NormalGAN: Learning Detailed 3D Human from a Single RGB-D Image},
author={Wang, Lizhen and Zhao, Xiaochen and Yu, Tao and Wang, Songtao and Liu, Yebin},
booktitle={ECCV},
year={2020}
}
```


