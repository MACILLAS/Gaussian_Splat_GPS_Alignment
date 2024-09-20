import numpy as np
import matplotlib.pyplot as plt

def analyse_err(err, filenames):
    file_num = []
    for file in filenames:
        file_num.append(int(file.split('_')[2]))
    file_num = np.array(file_num)

    fig, ax = plt.subplots()
    ax.scatter(np.abs(err), file_num, s=1)
    fig.savefig("error.png")
    plt.show()
