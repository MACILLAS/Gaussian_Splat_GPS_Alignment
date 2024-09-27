import numpy as np
import matplotlib.pyplot as plt

def analyse_err(err, file_num):

    fig, ax = plt.subplots()
    ax.scatter(np.abs(err), file_num, s=1)
    fig.savefig("error.png")
    plt.show()
