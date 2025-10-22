import matplotlib.pyplot as plt
import numpy as np

voc_mIoU = np.array(
    [
        66.63,
        61.85,
        49.6,
        16.98,
        70.32,
        64.44,
        50.18,
        15.15,
        62.38,
        56.64,
        42.67,
        12.57,
        32.07,
        28.84,
        10.35,
        3.66,
        39.01,
        36.54,
        16.57,
        3.2,
        32.93,
        30.69,
        13.53,
        2.93,
        56.89,
        53.1,
        51.08,
        41.76,
    ]
).reshape(7, 4)

city_mIoU = np.array(
    [
        28.73,
        25.91,
        18.91,
        3.85,
        29.25,
        27.32,
        20.58,
        10.96,
        26.46,
        23.81,
        16.54,
        2.87,
        11.48,
        10.72,
        5.2,
        1.41,
        11.28,
        10.68,
        4.49,
        0.79,
        8.79,
        8.32,
        3.39,
        0.87,
        24.06,
        23.69,
        22.45,
        18.57,
    ]
).reshape(7, 4)

ade_mIoU = np.array(
    [
        17.26,
        15.96,
        11.63,
        2.47,
        15.15,
        13.22,
        8.57,
        1.45,
        12.4,
        10.61,
        6.57,
        1.22,
        7.21,
        6.44,
        2.16,
        0.67,
        4.88,
        4.36,
        1.53,
        0.67,
        3.91,
        3.53,
        1.37,
        0.65,
        10.93,
        10.25,
        9.59,
        7.06,
    ]
).reshape(7, 4)

voc_mAcc = np.array(
    [
        80.6,
        76.73,
        65.33,
        29.98,
        83.63,
        78.83,
        66.83,
        28.27,
        76.11,
        70.76,
        57.16,
        22.73,
        47.75,
        44.22,
        20.37,
        8.92,
        55.18,
        52.4,
        28.98,
        8.6,
        47.78,
        45.12,
        24.11,
        7.51,
        73.68,
        70.01,
        68.11,
        59.74,
    ]
).reshape(7, 4)
city_mAcc = np.array(
    [
        47.64,
        42.59,
        31.05,
        11.24,
        47.43,
        43.98,
        33.4,
        11.91,
        43.86,
        39.53,
        27.57,
        9.22,
        22.16,
        20.76,
        12.28,
        6.43,
        20.86,
        19.76,
        10.5,
        5.51,
        17.24,
        16.54,
        9.1,
        5.48,
        52.3,
        51.5,
        50.26,
        46.06,
    ]
).reshape(7, 4)
ade_mAcc = np.array(
    [
        36.32,
        33.56,
        24.75,
        6.59,
        32.74,
        28.74,
        19.22,
        3.95,
        27.81,
        23.65,
        14.99,
        3.33,
        15.7,
        14.16,
        5.42,
        2.45,
        11.44,
        10.2,
        3.9,
        2.41,
        9.29,
        8.5,
        3.24,
        2.05,
        27.12,
        25.25,
        23.22,
        17.14,
    ]
).reshape(7, 4)

cbr = [0.01, 0.02, 0.03, 0.04][::-1]

name = ["mIoU", "mAcc"]
legend = [
    "SSCC+ProxyCLIP",
    "SSCC+ResCLIP",
    "SSCC+NaCLIP",
    "SwinJSCC+ProxyCLIP",
    "SwinJSCC+ResCLIP",
    "SwinJSCC+NaCLIP",
    "Ours",
]


fig = plt.figure(figsize=(18, 9))  # 增加高度为图例留空间

# VOC
ax1 = plt.subplot(2, 3, 1)
ax1.plot(voc_mIoU[0], cbr, marker="o", markersize=10, linestyle="--", label=legend[0])
ax1.plot(voc_mIoU[1], cbr, marker="o", markersize=10, linestyle="--", label=legend[1])
ax1.plot(voc_mIoU[2], cbr, marker="o", markersize=10, linestyle="--", label=legend[2])
ax1.plot(voc_mIoU[3], cbr, marker="s", markersize=10, label=legend[3])
ax1.plot(voc_mIoU[4], cbr, marker="s", markersize=10, label=legend[4])
ax1.plot(voc_mIoU[5], cbr, marker="*", markersize=18, label=legend[5])
ax1.plot(voc_mIoU[6], cbr, marker="s", markersize=10, label=legend[6])
ax1.set_ylabel("CBR", fontsize=18)
ax1.set_title("VOC", fontsize=18)
ax1.set_xlabel(f"{name[0]}", fontsize=18)

# Cityscapes
ax2 = plt.subplot(2, 3, 2)
ax2.plot(city_mIoU[0], cbr, marker="o", markersize=10, linestyle="--", label=legend[0])
ax2.plot(city_mIoU[1], cbr, marker="o", markersize=10, linestyle="--", label=legend[1])
ax2.plot(city_mIoU[2], cbr, marker="o", markersize=10, linestyle="--", label=legend[2])
ax2.plot(city_mIoU[3], cbr, marker="s", markersize=10, label=legend[3])
ax2.plot(city_mIoU[4], cbr, marker="s", markersize=10, label=legend[4])
ax2.plot(city_mIoU[5], cbr, marker="*", markersize=18, label=legend[5])
ax2.plot(city_mIoU[6], cbr, marker="s", markersize=10, label=legend[6])
ax2.set_ylabel("CBR", fontsize=18)
ax2.set_title("Cityscapes", fontsize=18)
ax2.set_xlabel(f"{name[0]}", fontsize=18)

# ADE20K
ax3 = plt.subplot(2, 3, 3)
ax3.plot(ade_mIoU[0], cbr, marker="o", markersize=10, linestyle="--", label=legend[0])
ax3.plot(ade_mIoU[1], cbr, marker="o", markersize=10, linestyle="--", label=legend[1])
ax3.plot(ade_mIoU[2], cbr, marker="o", markersize=10, linestyle="--", label=legend[2])
ax3.plot(ade_mIoU[3], cbr, marker="s", markersize=10, label=legend[3])
ax3.plot(ade_mIoU[4], cbr, marker="s", markersize=10, label=legend[4])
ax3.plot(ade_mIoU[5], cbr, marker="*", markersize=18, label=legend[5])
ax3.plot(ade_mIoU[6], cbr, marker="s", markersize=10, label=legend[6])
ax3.set_ylabel("CBR", fontsize=18)
ax3.set_title("ADE20K", fontsize=18)
ax3.set_xlabel(f"{name[0]}", fontsize=18)


# VOC
ax1 = plt.subplot(2, 3, 4)
ax1.plot(voc_mAcc[0], cbr, marker="o", markersize=10, linestyle="--", label=legend[0])
ax1.plot(voc_mAcc[1], cbr, marker="o", markersize=10, linestyle="--", label=legend[1])
ax1.plot(voc_mAcc[2], cbr, marker="o", markersize=10, linestyle="--", label=legend[2])
ax1.plot(voc_mAcc[3], cbr, marker="s", markersize=10, label=legend[3])
ax1.plot(voc_mAcc[4], cbr, marker="s", markersize=10, label=legend[4])
ax1.plot(voc_mAcc[5], cbr, marker="*", markersize=18, label=legend[5])
ax1.plot(voc_mAcc[6], cbr, marker="s", markersize=10, label=legend[6])
ax1.set_ylabel("CBR", fontsize=18)
ax1.set_title("VOC", fontsize=18)
ax1.set_xlabel(f"{name[1]}", fontsize=18)

# Cityscapes
ax2 = plt.subplot(2, 3, 5)
ax2.plot(city_mAcc[0], cbr, marker="o", markersize=10, linestyle="--", label=legend[0])
ax2.plot(city_mAcc[1], cbr, marker="o", markersize=10, linestyle="--", label=legend[1])
ax2.plot(city_mAcc[2], cbr, marker="o", markersize=10, linestyle="--", label=legend[2])
ax2.plot(city_mAcc[3], cbr, marker="s", markersize=10, label=legend[3])
ax2.plot(city_mAcc[4], cbr, marker="s", markersize=10, label=legend[4])
ax2.plot(city_mAcc[5], cbr, marker="*", markersize=18, label=legend[5])
ax2.plot(city_mAcc[6], cbr, marker="s", markersize=10, label=legend[6])
ax2.set_ylabel("CBR", fontsize=18)
ax2.set_title("Cityscapes", fontsize=18)
ax2.set_xlabel(f"{name[1]}", fontsize=18)

# ADE20K
ax3 = plt.subplot(2, 3, 6)
ax3.plot(ade_mAcc[0], cbr, marker="o", markersize=10, linestyle="--", label=legend[0])
ax3.plot(ade_mAcc[1], cbr, marker="o", markersize=10, linestyle="--", label=legend[1])
ax3.plot(ade_mAcc[2], cbr, marker="o", markersize=10, linestyle="--", label=legend[2])
ax3.plot(ade_mAcc[3], cbr, marker="s", markersize=10, label=legend[3])
ax3.plot(ade_mAcc[4], cbr, marker="s", markersize=10, label=legend[4])
ax3.plot(ade_mAcc[5], cbr, marker="*", markersize=18, label=legend[5])
ax3.plot(ade_mAcc[6], cbr, marker="s", markersize=10, label=legend[6])
ax3.set_ylabel("CBR", fontsize=18)
ax3.set_title("ADE20K", fontsize=18)
ax3.set_xlabel(f"{name[1]}", fontsize=18)

# 获取最后一个子图的handles和labels（所有子图的图例相同）
handles, labels = ax3.get_legend_handles_labels()

# 在整个figure底部添加图例
fig.legend(
    handles,
    labels,
    loc="lower center",  # 底部居中
    bbox_to_anchor=(0.5, -0.02),  # 微调位置
    ncol=7,  # 7列平铺
    fontsize=14,  # 字体大小
    frameon=True,
)  # 显示边框

plt.tight_layout(rect=[0, 0.03, 1, 1])  # [left, bottom, right, top]

plt.savefig(f"z_cbr_curve.png", bbox_inches="tight", dpi=300)
plt.close()
