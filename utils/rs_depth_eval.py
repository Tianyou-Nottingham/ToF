import numpy as np
import scipy
import scipy.optimize

def Guassian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))

def depth_process(rs_depth_path, ToF_depth_path, R, T):
    rs_depth = np.load(rs_depth_path)
    ToF_depth = np.load(ToF_depth_path)
    rs_depth = rs_depth.astype(np.float32)
    ToF_depth = ToF_depth.astype(np.float32)
    if R is not None:
        calib_depth = np.dot(R, ToF_depth) + T
    else:
        calib_depth = ToF_depth

    rs_depth.resize((480, 480))
    rs_zones = np.zeros((8, 8))
    # 将rs深度图划分为8*8个小块，每个小块内统计距离，拟合曲线，得到期望深度
    for i in range(8):
        for j in range(8):
            rs_depth_block = rs_depth[i*60:(i+1)*60, j*60:(j+1)*60]
            bins = np.arange(0, 700, 1)
            hist, bin_edges = np.histogram(rs_depth_block, bins=bins)
            most_freq_depth = bin_edges[np.argmax(hist)]
            param, covar = scipy.optimize.curve_fit(Guassian, bin_edges[:-1], hist, p0=[most_freq_depth, 10])
            mu_depth, sigma = param
            rs_zones[i, j] = mu_depth
    return rs_zones, calib_depth

def eval(rs, ToF):
    rs = rs.flatten()
    ToF = ToF.flatten()
    rs = rs[rs > 0]
    ToF = ToF[ToF > 0]
    rs = rs[rs < 700]
    ToF = ToF[ToF < 700]
    diff = rs - ToF
    diff = np.abs(diff)
    rms_diff = np.sqrt(np.mean(diff**2))
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    return mean_diff, rms_diff, std_diff