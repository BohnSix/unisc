import matplotlib.pyplot as plt
import numpy as np


legend = [
    "SSCC+ProxyCLIP",
    "SSCC+ResCLIP",
    "SSCC+NaCLIP",
    "SwinJSCC+ProxyCLIP",
    "SwinJSCC+ResCLIP",
    "SwinJSCC+NaCLIP",
    "Ours",
]

voc = [
    [
        61.85,
        64.44,
        56.64,
        28.84,
        36.54,
        30.69,
        53.1,
        16.98,
        15.15,
        12.57,
        3.66,
        3.2,
        2.93,
        41.76,
    ],
    [
        76.73,
        78.83,
        70.76,
        44.22,
        52.4,
        45.12,
        70.01,
        29.98,
        28.27,
        22.73,
        8.92,
        8.6,
        7.51,
        59.74,
    ],
]
city = [
    [
        25.91,
        27.32,
        23.81,
        10.72,
        10.68,
        8.32,
        23.69,
        3.85,
        10.96,
        2.87,
        1.41,
        0.79,
        0.87,
        18.57,
    ],
    [
        42.59,
        43.98,
        39.53,
        20.76,
        19.76,
        16.54,
        51.5,
        11.24,
        11.91,
        9.22,
        6.43,
        5.51,
        5.48,
        46.06,
    ],
]
ade = [
    [
        15.96,
        13.22,
        10.61,
        6.44,
        4.36,
        3.53,
        10.25,
        2.47,
        1.45,
        1.22,
        0.67,
        0.67,
        0.65,
        7.06,
    ],
    [
        33.56,
        28.74,
        23.65,
        14.16,
        10.2,
        8.5,
        25.25,
        6.59,
        3.95,
        3.33,
        2.45,
        2.41,
        2.05,
        17.14,
    ],
]


name = ["aAcc", "mIoU", "mAcc"]
cbr = [0.01, 0.03][::-1]
city = np.array(city).reshape(2, 2, 7).transpose(0, 2, 1)
ade = np.array(ade).reshape(2, 2, 7).transpose(0, 2, 1)
voc = np.array(voc).reshape(2, 2, 7).transpose(0, 2, 1)


plt.figure(figsize=(18, 8))
for i in range(2):  # miou, macc
    plt.subplot(2, 3, 3 * i + 1)
    plt.plot(cbr, voc[i, 0], marker="o", markersize=10, linestyle="--", label=legend[0])
    plt.plot(cbr, voc[i, 1], marker="o", markersize=10, linestyle="--", label=legend[1])
    plt.plot(cbr, voc[i, 2], marker="o", markersize=10, linestyle="--", label=legend[2])
    plt.plot(cbr, voc[i, 3], marker="s", markersize=10, label=legend[3])
    plt.plot(cbr, voc[i, 4], marker="s", markersize=10, label=legend[4])
    plt.plot(cbr, voc[i, 5], marker="s", markersize=10, label=legend[5])
    plt.plot(cbr, voc[i, 6], marker="*", markersize=18, label=legend[6])
    if i == 1:
        plt.xlabel("CBR", fontsize=18)
    if i == 0:
        plt.title("VOC", fontsize=18)
    plt.ylabel(f"{name[i%3]}", fontsize=18)
    plt.tight_layout(pad=0.3)
    plt.subplot(2, 3, 3 * i + 2)
    plt.plot(
        cbr, city[i, 0], marker="o", markersize=10, linestyle="--", label=legend[0]
    )
    plt.plot(
        cbr, city[i, 1], marker="o", markersize=10, linestyle="--", label=legend[1]
    )
    plt.plot(
        cbr, city[i, 2], marker="o", markersize=10, linestyle="--", label=legend[2]
    )
    plt.plot(cbr, city[i, 3], marker="s", markersize=10, label=legend[3])
    plt.plot(cbr, city[i, 4], marker="s", markersize=10, label=legend[4])
    plt.plot(cbr, city[i, 5], marker="s", markersize=10, label=legend[5])
    plt.plot(cbr, city[i, 6], marker="*", markersize=18, label=legend[6])
    if i == 1:
        plt.xlabel("CBR", fontsize=18)
    if i == 0:
        plt.title("Cityscapes", fontsize=18)
    plt.tight_layout(pad=0.3)
    plt.subplot(2, 3, 3 * i + 3)
    plt.plot(cbr, ade[i, 0], marker="o", markersize=10, linestyle="--", label=legend[0])
    plt.plot(cbr, ade[i, 1], marker="o", markersize=10, linestyle="--", label=legend[1])
    plt.plot(cbr, ade[i, 2], marker="o", markersize=10, linestyle="--", label=legend[2])
    plt.plot(cbr, ade[i, 3], marker="s", markersize=10, label=legend[3])
    plt.plot(cbr, ade[i, 4], marker="s", markersize=10, label=legend[4])
    plt.plot(cbr, ade[i, 5], marker="s", markersize=10, label=legend[5])
    plt.plot(cbr, ade[i, 6], marker="*", markersize=18, label=legend[6])
    if i == 1:
        plt.xlabel("CBR", fontsize=18)
    if i == 0:
        plt.title("ADE20K", fontsize=18)
    plt.tight_layout(pad=0.3)
    plt.legend(fontsize=10, loc="upper left")
# plt.savefig(f"cbr_curve.pdf")
plt.savefig(f"z_cbr_curve.png")
plt.close()


plt.figure(figsize=(18, 8))
for i in range(3):  # miou, macc
    plt.subplot(2, 3, 3 * i + 1)
    plt.plot(voc[i, 0], cbr, marker="o", markersize=10, linestyle="--", label=legend[0])
    plt.plot(voc[i, 1], cbr, marker="o", markersize=10, linestyle="--", label=legend[1])
    plt.plot(voc[i, 2], cbr, marker="o", markersize=10, linestyle="--", label=legend[2])
    plt.plot(voc[i, 3], cbr, marker="s", markersize=10, label=legend[3])
    plt.plot(voc[i, 4], cbr, marker="s", markersize=10, label=legend[4])
    plt.plot(voc[i, 5], cbr, marker="s", markersize=10, label=legend[5])
    plt.plot(voc[i, 6], cbr, marker="*", markersize=18, label=legend[6])
    if i == 1:
        plt.xlabel("CBR", fontsize=18)
    if i == 0:
        plt.title("VOC", fontsize=18)
    plt.ylabel(f"{name[i%3]}", fontsize=18)
    plt.tight_layout(pad=0.3)
    plt.subplot(2, 3, 3 * i + 2)
    plt.plot(
        city[i, 0], cbr, marker="o", markersize=10, linestyle="--", label=legend[0]
    )
    plt.plot(
        city[i, 1], cbr, marker="o", markersize=10, linestyle="--", label=legend[1]
    )
    plt.plot(
        city[i, 2], cbr, marker="o", markersize=10, linestyle="--", label=legend[2]
    )
    plt.plot(city[i, 3], cbr, marker="s", markersize=10, label=legend[3])
    plt.plot(city[i, 4], cbr, marker="s", markersize=10, label=legend[4])
    plt.plot(city[i, 5], cbr, marker="s", markersize=10, label=legend[5])
    plt.plot(city[i, 6], cbr, marker="*", markersize=18, label=legend[6])
    if i == 1:
        plt.xlabel("CBR", fontsize=18)
    if i == 0:
        plt.title("Cityscapes", fontsize=18)
    plt.tight_layout(pad=0.3)
    plt.subplot(2, 3, 3 * i + 3)
    plt.plot(ade[i, 0], cbr, marker="o", markersize=10, linestyle="--", label=legend[0])
    plt.plot(ade[i, 1], cbr, marker="o", markersize=10, linestyle="--", label=legend[1])
    plt.plot(ade[i, 2], cbr, marker="o", markersize=10, linestyle="--", label=legend[2])
    plt.plot(ade[i, 3], cbr, marker="s", markersize=10, label=legend[3])
    plt.plot(ade[i, 4], cbr, marker="s", markersize=10, label=legend[4])
    plt.plot(ade[i, 5], cbr, marker="s", markersize=10, label=legend[5])
    plt.plot(ade[i, 6], cbr, marker="*", markersize=18, label=legend[6])
    if i == 1:
        plt.xlabel("CBR", fontsize=18)
    if i == 0:
        plt.title("ADE20K", fontsize=18)
    plt.tight_layout(pad=0.3)
    plt.legend(fontsize=10, loc="upper left")
# plt.savefig(f"cbr_curve.pdf")
plt.savefig(f"z_cbr_curve1.png")
plt.close()
