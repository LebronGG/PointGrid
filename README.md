# PointGrid: A Deep Network for 3D Shape Understanding

## Prerequisites:
1. Python3 (with necessary common libraries such as numpy, scipy, etc.)
2. TensorFlow(1.10 or other)
3. You can prepare your data in *.npy file or other,here I use .npy file:
    * .npy file include :   x y z label.
    * I'm doing semantic segmentation,so I made some changes to the data.
## Prepare Data: 
	python ../data/pre_data.py
## Train:
	python train.py
## Test:
	python test.py

If you find this code useful, please cite raw work at <br />
<pre>
@article{PointGrid,
	author = {Truc Le and Ye Duan},
	titile = {{PointGrid: A Deep Network for 3D Shape Understanding}},
	journal = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2018},
}
</pre>
