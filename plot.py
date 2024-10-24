import matplotlib.pyplot as plt

if __name__ == "__main__":
    frequency = 10 ## 10 Hz
    times = [i for i in range(3*60)]
    single = 1384/1024/1024 ## 1384 bytes to MB
    resolution = 256*256
    memory_wo_img = [single*frequency * 60*time for time in times]
    memory_w_img = [(single + resolution*3/1024/1024) * frequency * 60*time for time in times]
    plt.plot(times, memory_wo_img, label='Without Image')
    plt.plot(times, memory_w_img, label='With Image')
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage')
    plt.show()
