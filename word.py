import numpy as np
import matplotlib.pyplot as plt
import std_mfs

class Word:

    def __init__(self, title, x, lmf, umf, base_on='std'):
        self.title = title
        self.U = np.arange(*x) if base_on == 'std' else x[:]
        lmf_type, *lmf_params = lmf
        umf_type, *umf_params = umf
        self.lmf = getattr(std_mfs, lmf_type)(self.U, *lmf_params) if base_on == 'std' else lmf[:]
        self.umf = getattr(std_mfs, umf_type)(self.U, *umf_params) if base_on == 'std' else umf[:]

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_ylabel(r'$\mu$(x)')
        ax.set_xlabel('x')
        ax.set_ylim([0, 1.2])
        ax.fill_between(self.U, self.lmf, self.umf, color="lightblue", edgecolor="blue", alpha=.9)
        plt.grid(True)
        plt.title(self.title)
        plt.show()

    def __str__(self):
        return self.title

    def __len__(self):
        return len(self.U)

    def similarity_measure(self, other):
        min_lmf = np.minimum(self.lmf, other.lmf)
        max_lmf = np.maximum(self.lmf, other.lmf)
        min_umf = np.minimum(self.umf, other.umf)
        max_umf = np.maximum(self.umf, other.umf)

        return (np.sum(min_umf) + np.sum(min_lmf)) / (np.sum(max_umf) + np.sum(max_lmf))

    def aggregate(self, other_words):
        """ Агрегування декількох фаззі-слів в одне загальне слово. """
        aggregated_lmf = np.fmin.reduce([w.lmf for w in other_words])
        aggregated_umf = np.fmin.reduce([w.umf for w in other_words])

        self.lmf = aggregated_lmf
        self.umf = aggregated_umf
        self.title = "Агрегована оцінка"
