import numpy as np
import matplotlib.pyplot as plt

# SNR 横轴
a1_snr = np.array([-4, -1, 1, 4, 5, 10, 13, 17])
a1_aacc = np.array([0, 0, 0, 0, 67.32, 67.32, 67.32, 67.32])
a1_miou = np.array([0, 0, 0, 0, 26.32, 26.32, 26.32, 26.32])
a1_macc = np.array([0, 0, 0, 0, 44.41, 44.41, 44.41, 44.41])

a2_snr = np.array([-4, -1, 1, 4, 5, 10, 13, 17])
a2_aacc = np.array([0, 0, 0, 0, 68.67, 68.67, 68.67, 68.67])
a2_miou = np.array([0, 0, 0, 0, 27.32, 27.32, 27.32, 27.32])
a2_macc = np.array([0, 0, 0, 0, 43.98, 43.98, 43.98, 43.98])

a3_snr = np.array([-4, -1, 1, 4, 5, 10, 13, 17])
a3_aacc = np.array([0, 0, 0, 0, 63.26, 63.26, 63.26, 63.26])
a3_miou = np.array([0, 0, 0, 0, 23.81, 23.81, 23.81, 23.81])
a3_macc = np.array([0, 0, 0, 0, 39.53, 39.53, 39.53, 39.53])


b1_snr = np.array([-4, -1, 1, 5, 10, 17])
b1_aacc = np.array([22.55, 25.2, 35.3, 37.91, 39.40, 46.13])
b1_miou = np.array([2.91, 3.72, 6.20, 7.57, 8.72, 10.72])
b1_macc = np.array([7.83, 9.14, 13.55, 15.26, 17.93, 20.76])


b2_snr = np.array([-4, -1, 1, 5, 10, 17])
b2_aacc = np.array([26.38, 30.17, 40.15, 43.68, 45.91, 47.17])
b2_miou = np.array([2.87, 3.69, 6.47, 8.36, 9.82, 10.68])
b2_macc = np.array([7.1, 8.66, 12.95, 15.87, 18.97, 19.76])


b3_snr = np.array([-4, -1, 1, 5, 10, 17])
b3_aacc = np.array([20.58, 23.6, 33.97, 36.74, 38.08, 39.30])
b3_miou = np.array([2.53, 3.2, 5.4, 6.75, 7.74, 8.32])
b3_macc = np.array([6.82, 8.1, 11.56, 13.38, 15.63, 16.54])

ours_snr = np.array([-4, -1, 1, 4, 5, 10, 13, 17])
ours_aacc = np.array([37.45, 37.46, 37.48, 37.39, 37.28, 36.28, 34.37, 28.42])[::-1]
ours_miou = np.array([23.69, 23.58, 23.4, 22.52, 22.15, 20.22, 17.93, 12.92])[::-1]
ours_macc = np.array([51.5, 51.39, 51.21, 50.24, 49.83, 47.64, 44.76, 37.41])[::-1]


labels = [
    "sscC+Proxy",
    "SSCC+ResCLIP",
    "SSCC+NaCLIP",
    "Swin+Proxy",
    "Swin+ResCLIP",
    "Swin+NaCLIP",
    "Ours",
]


plt.figure(figsize=(15, 3.5))

# plt.subplot(1, 3, 1)
# plt.title('aAcc')
# plt.plot(a1_snr, a1_aacc, marker='o', label=labels[0])
# plt.plot(a2_snr, a2_aacc, marker='o', label=labels[1])
# plt.plot(a3_snr, a3_aacc, marker='o', label=labels[2])
# plt.plot(b1_snr, b1_aacc, marker='o', label=labels[3])
# plt.plot(b2_snr, b2_aacc, marker='o', label=labels[4])
# plt.plot(b3_snr, b3_aacc, marker='o', label=labels[5])
# plt.plot(ours_snr, ours_aacc, marker='*', label=labels[6])

plt.subplot(1, 2, 1)
plt.title("mIoU(%)", fontsize=18)
plt.plot(a1_snr, a1_miou, linestyle="--", marker="o", markersize=10, label=labels[0])
plt.plot(a2_snr, a2_miou, linestyle="--", marker="o", markersize=10, label=labels[1])
plt.plot(a3_snr, a3_miou, linestyle="--", marker="o", markersize=10, label=labels[2])
plt.plot(b1_snr, b1_miou, marker="s", markersize=10, label=labels[3])
plt.plot(b2_snr, b2_miou, marker="s", markersize=10, label=labels[4])
plt.plot(b3_snr, b3_miou, marker="s", markersize=10, label=labels[5])
plt.plot(ours_snr, ours_miou, marker="*", markersize=15, label=labels[6])
plt.grid(alpha=0.3)
plt.ylabel("AWGN Channel", fontsize=18)
plt.legend(fontsize=10, loc="lower right")
# plt.xlabel('SNR (dB)', fontsize=18)
plt.subplot(1, 2, 2)
plt.title("mAcc(%)", fontsize=18)
plt.plot(a1_snr, a1_macc, linestyle="--", marker="o", markersize=10, label=labels[0])
plt.plot(a2_snr, a2_macc, linestyle="--", marker="o", markersize=10, label=labels[1])
plt.plot(a3_snr, a3_macc, linestyle="--", marker="o", markersize=10, label=labels[2])
plt.plot(b1_snr, b1_macc, marker="s", markersize=10, label=labels[3])
plt.plot(b2_snr, b2_macc, marker="s", markersize=10, label=labels[4])
plt.plot(b3_snr, b3_macc, marker="s", markersize=10, label=labels[5])
plt.plot(ours_snr, ours_macc, marker="*", markersize=15, label=labels[6])
# plt.xlabel('SNR (dB)', fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0.1)
plt.savefig("z_snr.pdf")


labels = ["Swin+Proxy", "Swin+ResCLIP", "Swin+NaCLIP", "Ours"]
snr = np.array([-4, -1, 1, 5, 10, 13, 17])
a1_aAcc = np.array([20.64, 23.13, 30.46, 31.74, 31.84, 32.28, 32.22])
a1_miou = np.array([2.6, 3.13, 4.52, 5.31, 5.85, 6.08, 6.1])
a1_macc = np.array([7.34, 8.04, 10.78, 11.79, 12.72, 13.65, 13.96])

a2_aAcc = np.array([24.01, 27.61, 34.92, 37.1, 37.84, 38.55, 38.88])
a2_miou = np.array([2.56, 2.99, 4.54, 5.7, 6.44, 6.84, 6.9])
a2_macc = np.array([6.56, 7.33, 10.09, 11.74, 12.88, 13.57, 14.04])

a3_aAcc = np.array([18.47, 21.08, 29.14, 30.32, 30.56, 30.7, 30.72])
a3_miou = np.array([2.24, 2.62, 4.00, 4.59, 5.07, 5.27, 5.38])
a3_macc = np.array([6.35, 6.94, 9.27, 9.9, 11.1, 11.45, 12.05])

ours_aAcc = np.array([13.43, 18.6, 19.85, 25.28, 28.55, 29.3, 29.77])
ours_miou = np.array([3.3, 4.77, 6.17, 9.69, 13.73, 15.21, 16.21])
ours_macc = np.array([13.78, 16.99, 22.63, 30.60, 36.66, 38.33, 39.39])


plt.figure(figsize=(15, 3.5))

plt.subplot(1, 2, 1)
# plt.title('mIoU(%)', fontsize=18)
plt.plot([], [])
plt.plot([], [])
plt.plot([], [])
plt.plot(snr, a1_miou, marker="s", markersize=10, label=labels[0])
plt.plot(snr, a2_miou, marker="s", markersize=10, label=labels[1])
plt.plot(snr, a3_miou, marker="s", markersize=10, label=labels[2])
plt.plot(snr, ours_miou, marker="*", markersize=15, label=labels[3])
plt.grid(alpha=0.3)
plt.legend(fontsize=10, loc="upper left")
plt.ylabel("Rayleigh Channel", fontsize=18)
plt.xlabel("SNR (dB)", fontsize=18)
plt.subplot(1, 2, 2)
# plt.title('mAcc(%)', fontsize=18)
plt.plot([], [])
plt.plot([], [])
plt.plot([], [])
plt.plot(snr, a1_macc, marker="s", markersize=10, label=labels[0])
plt.plot(snr, a2_macc, marker="s", markersize=10, label=labels[1])
plt.plot(snr, a3_macc, marker="s", markersize=10, label=labels[2])
plt.plot(snr, ours_macc, marker="*", markersize=15, label=labels[3])
plt.xlabel("SNR (dB)", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0.1)
plt.savefig("z_snr_rayleigh.pdf")
plt.close()
