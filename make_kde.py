# cern's root integration
import ROOT

# basic python stuff
import numpy as np
import matplotlib.pyplot as plt
import random

# kernel density estimation
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def my_scores(estimator, X):
    scores = estimator.score_samples(X)
    # Remove -inf
    scores = scores[scores != float('-inf')]
    # Return the mean values
    return np.mean(scores)

def main():
    
    # read in data from the root file
    input_file = ROOT.TFile("n_alpha.root","read")
    ex_7a = input_file.Get("h1r_Mult7;1")
    canvas = ROOT.TCanvas()
    ex_7a.Draw()
    canvas.Print("ex_7a.png")

    # read the data from the ROOT histogram into an list
    extracted_data = []
    for index in range(ex_7a.GetNbinsX()):
        bin_content = ex_7a.GetBinContent(index)
        bin_center = ex_7a.GetBinCenter(index)
        bin_width = ex_7a.GetBinWidth(index)
        for _ in range(int(bin_content)):
            x_value = random.uniform(float(bin_center - bin_width), float(bin_center + bin_width))
            extracted_data.append(x_value)

    # reformat the list into a np.array with the correct dimensionality
    extracted_data = np.array(extracted_data)[:, np.newaxis]
    
    # find optimized paramaters for KDE with the gridsearch
    kde_pts = np.linspace(50, 200, 100)[:, np.newaxis]
    
    kernels = ['gaussian', 'linear', 'tophat']
    h_vals = np.linspace(.01, 2.00, 2)

    
    grid = GridSearchCV(KernelDensity(), {'bandwidth': h_vals, 'kernel': kernels}, scoring=my_scores)
    grid.fit(extracted_data)    
    best_kde = grid.best_estimator_
    log_density = best_kde.score_samples(kde_pts)
    
    print(f'{best_kde.kernel=}')
    print(f'{best_kde.bandwidth=}')
    
    fig, ax = plt.subplots()
    ax.plot(kde_pts, np.exp(log_density))
    ax.set_xlabel('E*')
    ax.set_ylabel('normalized yield')


    fig.savefig('ex_7a_kde.png')
    # end of main

if __name__ == "__main__":
    main()