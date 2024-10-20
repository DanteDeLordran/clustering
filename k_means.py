import numpy as np
from tkinter import filedialog

from numpy import ndarray


def csv_to_matrix() -> ndarray:
    matrix = np.loadtxt(filedialog.askopenfilename(), delimiter=',')
    return matrix


def main():
    matrix = csv_to_matrix()


if __name__ == '__main__':
    main()