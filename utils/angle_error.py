import numpy as np
import matplotlib.pyplot as plt


def angle():
    file_name = (
        "E://OneDrive - The University of Nottingham//Documents//Code//ToF//angle.txt"
    )
    angle = [
        "-60",
        "-50",
        "-40",
        "-30",
        "-20",
        "-10",
        "0",
        "10",
        "20",
        "30",
        "40",
        "50",
        "60",
    ]
    data = {
        "-60": [],
        "-50": [],
        "-40": [],
        "-30": [],
        "-20": [],
        "-10": [],
        "0": [],
        "10": [],
        "20": [],
        "30": [],
        "40": [],
        "50": [],
        "60": [],
    }
    with open(file_name, "r", encoding="UTF-8") as f:
        for line in f:
            if "Plane" not in line:
                angle_ = line.split(":")[-1][:-2].strip()
                for i in range(5):
                    line_ = f.readline()
                    error = float(line_.split("Error: ")[1].split("\n")[0])
                    data[angle_].append(error)
    print(data)

    up_bound = [max(data[angle]) for angle in angle]
    low_bound = [min(data[angle]) for angle in angle]
    fig = plt.figure(figsize=(14, 7))
    plt.plot([np.mean(data[angle]) for angle in angle], label="Mean Error")
    # plt.plot(up_bound, label="Upper Bound", )
    # plt.plot(low_bound, label="Lower Bound")
    plt.fill_between(range(13), up_bound, low_bound, alpha=0.3)

    ## smooth the data
    plt.xticks(range(13), angle)
    plt.xlabel("Angle")
    plt.ylabel("Error/mm")
    plt.title("Error vs Angle")

    plt.legend()
    plt.show()


def compare():
    ours = np.array(
        [
            3.3610892698280677,
            3.071071350627453,
            3.2403698843207596,
            3.1959560996562555,
            3.498694745394526,
            2.9374221772147426,
            3.2809237010104093,
            4.349381721258749,
            3.253700170080281,
            3.3818851595599937,
        ]
    )
    direct = np.array(
        [
            15.160413092794583,
            1.7915463337884954,
            1.8658003282254483,
            14.937812316102354,
            14.947450486939838,
            14.942181922391972,
            1.8507121416647396,
            1.8067317777348466,
            14.939632172580927,
            1.9792169301967981,
        ]
    )
    plt.scatter(x=[1 for _ in range(10)], y=ours, label="Ours")
    plt.scatter(x=[2 for _ in range(10)], y=direct, label="Direct")
    plt.legend()

    plt.boxplot(
        [ours, direct],
        labels=["Ours", "Direct"],
        patch_artist=True,
        boxprops=dict(facecolor="lightcyan"),
        meanline=True,
        # meanprops=dict(color="red"),
        # showmeans=True,
        # medianprops=dict(color="lightcyan"),
        # showfliers=False,
    )
    plt.ylabel("Error/mm")
    plt.title("Error Comparison")
    plt.show()


if __name__ == "__main__":
    compare()
