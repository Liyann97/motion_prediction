#!/usr/bin/python
# copyright(c) strongnine
# 利用 滑动窗口 将一个长的序列信号生成多个 GAF 图片
# Using a sliding window to generate GAF images from a long sequence of signals
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from pyts.image import GramianAngularField

# 根据自己的需求修改一下变量的参数
filename = "./data/data.csv"  # 要处理的文件路径
savepath = "./data"  # GAF 图片保存的路径
img_sz = 128  # 生成的 GAF 图片的大小 (the size of each GAF image)
# 如果 滑动窗口的大小 等于 滑动步长 则滑动窗口之间没有重叠
window_sz = 128  # 滑动窗口的大小，需要满足 window_sz > img_sz
step = 16  # 滑动窗口的步长 (step of slide window)
assert window_sz >= img_sz, "window_sz < img_sz（滑动窗口大小 小于 GAF 图片尺寸）。"
method = 'summation'  # GAF 图片的类型，可选 'summation'（默认）和 'difference'

# 以下是 GAF 生成的代码
print("GAF 生成方法：%s，图片大小：%d * %d" % (method, img_sz, img_sz))
img_path = "%s/images" % savepath  # 可视化图片保存的文件夹
data_path = "%s/data" % savepath  # 数据文件保存的文件夹
if not os.path.exists(img_path):
    os.makedirs(img_path)  # 如果文件夹不存在就创建一个
if not os.path.exists(data_path):
    os.makedirs(data_path)  # 如果文件夹不存在就创建一个

print("开始生成...")
print("可视化图片保存在文件夹 %s 中，数据文件保存在文件夹 %s 中。" % (img_path, data_path))
gaf = GramianAngularField(image_size=img_sz, method=method)
# data = np.loadtxt(filename, delimiter=",").reshape(1, -1)
dataset = np.loadtxt(r'F:/my_data/W1/10hz/W1_4a.dat')
data = dataset[:, 1]
data = np.array(data)
data = data.reshape(1, -1)
_, n_sample = data.shape

img_num = 0  # 生成图片的总数 (the total numbers of GAF images)
start_index, end_index = 0, window_sz  # 序列开头以及末尾索引

while end_index <= n_sample:
    img_num += 1
    sub_series = data[:, start_index:end_index]
    gaf_img = gaf.fit_transform(sub_series)  # 转化为 GAF 图片

    img_save_path = "{}/{}.png".format(img_path, img_num)
    data_save_path = "{}/{}.csv".format(data_path, img_num)
    image.imsave(img_save_path, gaf_img[0])  # 保存图片 (save image)
    np.savetxt(data_save_path, gaf_img[0], delimiter=',')  # 保存数据为 csv 文件

    # 滑动窗口向后移动
    start_index += step
    end_index += step

print("生成完成，共处理 {} 个图片。".format(img_num))
