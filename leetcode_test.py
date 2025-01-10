import numpy as np
import matplotlib.pyplot as plt
def plot():
    # Define parameters
    radius = 1.0  # Radius of the circular membrane
    num_points = 200  # Number of points in the angular direction
    modes = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), 
             (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3),
             (4, 1), (4, 2), (4, 3)]  # Vibrational modes (n, m)

    theta = np.linspace(0, 2 * np.pi, num_points)  # Angular coordinate
    r = np.linspace(0, radius, num_points)  # Radial coordinate
    R, T = np.meshgrid(r, theta)  # Create meshgrid for polar coordinates

    # Function to calculate mode shape
    def mode_shape(n, m, R, T):
        return np.cos(n * T) * np.sin(m * np.pi * R / radius)

    # Plot the modes
    fig, axes = plt.subplots(5, 3, subplot_kw={'projection': 'polar'}, figsize=(15, 15))
    for i, (n, m) in enumerate(modes):
        Z = mode_shape(n, m, R, T)
        # axes[i // 3, i % 3]
        axes[i // 3, i % 3].contourf(T, R, Z, 50, cmap='viridis')
        axes[i // 3, i % 3].set_title(f"Mode (n={n}, m={m})")
        axes[i // 3, i % 3].set_ylim(0, radius)

    plt.tight_layout()
    plt.show()

def test():
    # 参数
    radius = 1.0  # 圆盘半径 (m)
    f = 10.0      # 均匀外力密度 (N/m^2)
    T = 100.0     # 张力 (N/m)

    # 径向网格
    r = np.linspace(0, radius, 100)

    # 静态位移
    u_static = -f / (4 * T) * (radius**2 - r**2)

    # 绘图
    plt.figure(figsize=(6, 4))
    plt.plot(r, u_static, label="Static Displacement", color='b')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("Radius (r)")
    plt.ylabel("Displacement (u)")
    plt.title("Static Displacement of Circular Membrane under Uniform Load")
    plt.legend()
    plt.grid()
    plt.show()

def plot_circular():
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    # ax.set_rmax(2)
    ax.set_rticks([])  # Less radial ticks
    # with more angular ticks
    ax.set_xticks(np.pi/180. * np.linspace(0,  360, 36, endpoint=False))
    ax.set_xlim(-np.pi/2, np.pi/2)
    for i in range(-9, 10):
        ax.plot([0, np.pi/180. * i * 10], [0, 1], color='r', linewidth=1)
        ax.set_xticks(np.pi/180. * np.linspace(-90,  100, 19, endpoint=False))

    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_circular()
    