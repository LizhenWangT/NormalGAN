# NormalGAN
**NormalGAN: Learning Detailed 3D Human from a Single RGB-D Image (ECCV 2020)**<br>
Lizhen Wang, Xiaochen Zhao, [Tao Yu](https://ytrock.com/), [Yebin Liu](http://www.liuyebin.com/) and Songtao Wang<br>
We propose NormalGAN, a fast adversarial learning-based method to reconstruct the complete and detailed 3D human from a single RGB-D image.

[Project Page](http://www.liuyebin.com/NormalGan/normalgan.html)

[paper](https://export.arxiv.org/abs/2007.15340)


# Changelog
2020.08.11 Release the test code and pretrained models

# Requirements
The code and released model were trained on
 * Ubuntu 16.04 & Python 3.5.2
 * Pytorch 1.12
 * trimesh 3.2.36
 * Python-opencv 3.4.3 (for Python-opencv >= 4.0, please change Line 266 of src/ops.py to "contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)")

Optional for src/redner.py:
 * Pyrender 0.1.32

Recommend for Kinect v2 python implement
 * [pylibfreenect2](https://github.com/r9y9/pylibfreenect2)


# Pretrained models
Download the pretrained models in [Comming soon].

Put the pretrained models in the directory model/pretrained/

# Testing
Generate the csv file for the image path. 
>cd datasets

>python data_utils/createcsv.py testdata/ test.csv

>cd ..

Run the test_offline.sh (your csv file name) (your save dir name)
>sh test_offline.sh test.csv testdata

Results are shown in results/testdata/ply. You can use Poisson Reconstruction to the meshes for better performance of the edge area.

To test on your own data, you need to **apply the mask to your image** before input them into NormalGAN. 

Then create datasets/yourdata/color and datasets/yourdata/depth, use the same filename for your RGB-D image pairs. 

Use data_utils/createcsv.py to create your csv file and use test_offline.sh to test your data.

Note that, NormalGAN simulate noise of Kinect v2 (or similar ToF depth cameras), the image resolution should be **(512,424)** (and **424,424** for network input)

Please change the intrinsics or image resolution in src/test_offline.py.

