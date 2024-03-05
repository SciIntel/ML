
#-------------------------------------------
#
# FILENAME:	PYRO_detect_ML_toolbox.py
# 		
# CREATED: 	11.27.2021 - dserke
#
# NOTE:		8 methods overview at: https://medium.com/@rinu.gour123/8-machine-learning-algorithms-in-python-you-must-learn-e9b79b361f49
#
#-------------------------------------------

#-------------------------------------------
# IMPORT LIBRARIES
#-------------------------------------------
import sklearn

# ... K-means, unsupervised ML, solves clustering problem, classifies data using a number of round clusters
from sklearn.datasets       import make_blobs
from sklearn.cluster        import KMeans
from scipy.spatial.distance import cdist

# ... Gaussian Mixture Models, unsupervised ML, solves clustering problem, classifies data using Gaussian distributions
from   sklearn              import mixture
from   matplotlib.patches   import Ellipse

# ... Linear Regression, for continuous data
from   sklearn.datasets     import load_iris
from   sklearn.datasets     import load_boston
from   sklearn.linear_model import LinearRegression
from   sklearn              import neighbors, datasets

import scipy
from   scipy.signal         import medfilt2d
from   scipy                import ndimage

import seaborn              as     sns; sns.set()

import pyart
import pandas               as     pd
import numpy                as     np
import csv

import matplotlib           as     mpl
from   matplotlib           import pyplot as plt
from   matplotlib.colors    import ListedColormap
from   matplotlib.image     import NonUniformImage

import warnings
warnings.filterwarnings('ignore')

#-------------------------------------------
# DEFINE CONSTANTS
#-------------------------------------------

#-------------------------------------------
# LOAD INPUT FILES
#-------------------------------------------
# ... sample iris data
#iris                      = load_iris()

# ... sample Boston housing data
#data                      = load_boston()

# ... radar data into radar object
NEXRAD_site               = 'KBBX'  	# Camp, CA and Dixie, CA
ncfile_path               = '/d1/serke/projects/PYRO_detect_ATEC/data/NEXRAD/nc/' + NEXRAD_site + '/'
ncfile_name               = 'cfrad.20181108_155017.495_to_20181108_155134.996_KBBX_v992_s03_el1.49_SUR.nc'
ncfile_path_name          = ncfile_path + ncfile_name
radar                     = pyart.io.read(ncfile_path_name)

#-------------------------------------------
# MANIPULATE INPUT DATA
#-------------------------------------------
npix               = 5
# save 2D SW data as 1D np.array
X_SW               = np.array(radar.fields['SW']['data'][:,:]).flatten()
#X_SW               = np.array(radar.fields['SW']['data'][640:660, 72:170 ]).flatten()
X_SW[X_SW < 0]     = -0.1
X_RHO              = np.array(radar.fields['RHO']['data'][:,:]).flatten()
#X_RHO              = np.array(radar.fields['RHO']['data'][640:660, 72:170 ]).flatten()
X_RHO[X_RHO < 0]   = 0.20
X_REF              = np.array(radar.fields['REF']['data'][:,:]).flatten()
#X_REF              = np.array(radar.fields['REF']['data'][640:660, 72:170 ]).flatten()
X_REF[X_REF < -35] = -35
X_ZDR              = np.array(radar.fields['ZDR']['data'][:,:]).flatten()
#X_ZDR              = np.array(radar.fields['ZDR']['data'][640:660, 72:170 ]).flatten()
X_VEL              = np.array(radar.fields['VEL']['data'][:,:]).flatten()
#X_VEL              = np.array(radar.fields['VEL']['data'][640:660, 72:170 ]).flatten()
RHOstd             = ndimage.generic_filter(radar.fields['RHO']['data'][ : ], np.std, size=(npix, npix), mode='nearest')
SWstd              = ndimage.generic_filter(radar.fields['SW']['data'][ : ],  np.std, size=(npix, npix), mode='nearest')
X_RHOstd           = np.array(RHOstd[:,:]).flatten()
#X_RHOstd           = np.array(RHOstd[640:660, 72:170 ]).flatten()
X_SWstd            = np.array(SWstd[:,:]).flatten()
#X_SWstd            = np.array(SWstd[640:660, 72:170 ]).flatten()
#X_RAD        = np.column_stack((X_RAD_0, X_RAD_1))

#-------------------------------------------
# K-MEANS METHOD
#-------------------------------------------
# ... example found at: https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
#X, y                      = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
#plt.scatter(X[:,0], X[:,1])
#X, Y = np.meshgrid(X_RAD_0, X_RAD_1)
#def f(x, y):
#    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
#Z    = f(X, Y)

plt.figure(figsize=(15, 15))
#plt.imshow(Z)
#plt.contour(X, Y, Z, colors='black');
plt.scatter(X_SWstd, X_REF, c=X_RHOstd, vmin=0.08, vmax=0.15, cmap='viridis')
plt.colorbar()
plt.xlabel('SWstd')
plt.ylabel('REF')
plt.xlim([0.0,   5.0])
plt.ylim([-30.0, 60.0])

plt.figure(figsize=(15, 15))
#plt.imshow(Z)
#plt.contour(X, Y, Z, colors='black');
plt.scatter(X_REF, X_ZDR)
plt.xlabel('REF')
plt.ylabel('ZDR')
plt.xlim([ -30.0,  60])
plt.ylim([  -6.0,   6])

plt.figure(figsize=(15, 15))
#plt.imshow(Z)
#plt.contour(X, Y, Z, colors='black');
plt.scatter(X_SW, X_SWstd, c=X_RHOstd, vmin=0, vmax=0.25, cmap='viridis')
plt.colorbar()
plt.xlabel('SW')
plt.ylabel('SWstd')
plt.xlim([0.0,  20])
plt.ylim([0.0,   5])

plt.figure(figsize=(15, 15))
#plt.imshow(Z)
#plt.contour(X, Y, Z, colors='black');
plt.scatter(X_RHO, X_SWstd, c=X_SW, vmin=0, vmax=14, cmap='viridis')
plt.colorbar()
plt.xlabel('RHO')
plt.ylabel('SWstd')
plt.xlim([0.0,  1.1])
plt.ylim([0.0,  5.0])

plt.figure(figsize=(15, 15))
#plt.imshow(Z)
#plt.contour(X, Y, Z, colors='black');
plt.scatter(X_RHO, X_VEL, c=X_SW, vmin=0, vmax=14, cmap='viridis')
plt.colorbar()
plt.xlabel('RHO')
plt.ylabel('VEL')
plt.xlim([  0.0,  1.1])
plt.ylim([-20.0, 20.0])

plt.figure(figsize=(15, 15))
#plt.imshow(Z)
#plt.contour(X, Y, Z, colors='black');
plt.scatter(X_REF, X_SWstd, c=X_SW, cmap='viridis')
plt.colorbar()
plt.xlabel('REF')
plt.ylabel('SWstd')
plt.xlim([-30.0,  60.0])
plt.ylim([  0.0,   5.0])

plt.figure(figsize=(15, 15))
#plt.imshow(Z)
#plt.contour(X, Y, Z, colors='black');
plt.scatter(X_SWstd, X_RHOstd, c=X_REF, vmin=-20, vmax=50, cmap='viridis')
plt.colorbar()
plt.xlabel('SWstd')
plt.ylabel('RHOstd')
plt.xlim([  0.0,    5.00])
plt.ylim([  0.0,    0.35])

plt.figure(figsize=(15, 15))
#plt.imshow(Z)
#plt.contour(X, Y, Z, colors='black');
plt.scatter(X_SW, X_REF, c=X_SWstd, vmin=0, vmax=6, cmap='viridis')
plt.colorbar()
plt.xlabel('SW')
plt.ylabel('REF')
plt.xlim([  0.0,  15.0])
plt.ylim([-30.0,  60.0])

xedges            = [  0,   1,   2,   3,   4,  5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
yedges            = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
H, xedges, yedges = np.histogram2d(X_SW, X_REF, bins=(xedges, yedges))
fig               = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, title='pcolormesh: actual edges', aspect='auto')
X, Y              = np.meshgrid(xedges, yedges)
ax.pcolormesh(X, Y, H.T)

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_SW, X_REF, bins=30, cmin=0, cmax=25, cmap='Blues')
plt.xlim([0.0,  15.0])
plt.ylim([-20,  55])
plt.xlabel('SW')
plt.ylabel('REF')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_ZDR, X_REF, bins=200, cmin=0, cmax=25, cmap='Blues')
plt.xlim([-6.0,  6.0])
plt.ylim([-20,  55])
plt.xlabel('ZDR')
plt.ylabel('REF')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_ZDR, X_SW, bins=50, cmin=0, cmax=25, cmap='Blues')
plt.xlim([-6.0,  6.0])
plt.ylim([0,  15])
plt.xlabel('ZDR')
plt.ylabel('SW')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_ZDR, X_RHO, bins=100, cmin=0, cmax=25, cmap='Blues')
plt.xlim([-6.0,  6.0])
plt.ylim([0.2,  1.05])
plt.xlabel('ZDR')
plt.ylabel('RHO')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_RHOstd, X_RHO, bins=200, cmin=0, cmax=25, cmap='Blues')
plt.xlim([0.0,  65.0])
plt.ylim([0.2,  1.05])
plt.xlabel('RHOstd')
plt.ylabel('RHO')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_VEL, X_RHO, bins=200, cmin=0, cmax=25, cmap='Blues')
plt.xlim([-30.0,  30.0])
plt.ylim([0.2,  1.05])
plt.xlabel('VEL')
plt.ylabel('RHO')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_REF, X_RHO, bins=200, cmin=0, cmax=25, cmap='Blues')
plt.xlim([0.0,  55.0])
plt.ylim([0.2,  1.05])
plt.xlabel('REF')
plt.ylabel('RHO')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_SW, X_RHO, bins=30, cmin=0, cmax=25, cmap='Blues')
plt.xlim([0.0,  15.0])
plt.ylim([0.2,  1.05])
plt.xlabel('SW')
plt.ylabel('RHO')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_SW, X_SWstd, bins=100, cmin=0, cmax=25, cmap='Blues')
plt.xlim([0.0,  15.0])
plt.ylim([0.0,  7])
plt.xlabel('SW')
plt.ylabel('SWstd')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_REF, X_SWstd, bins=100, cmin=0, cmax=25, cmap='Blues')
plt.xlim([-20.0,  55.0])
plt.ylim([0.0,  6])
plt.xlabel('REF')
plt.ylabel('SWstd')
cb = plt.colorbar()
cb.set_label('counts in bin')

# focus in on REF > 20 dBZ space
offset     = 0.30
X_REF_V    = X_REF[   [X_VEL  < -offset]and[X_VEL > offset]and[X_RHO > 0.6]]
X_SW_V     = X_SW[    [X_VEL  < -offset]and[X_VEL > offset]and[X_RHO > 0.6]]
X_SWstd_V  = X_SWstd[ [X_VEL  < -offset]and[X_VEL > offset]and[X_RHO > 0.6]]
X_RHO_V    = X_RHO[   [X_VEL  < -offset]and[X_VEL > offset]and[X_RHO > 0.6]]
X_RHOstd_V = X_RHOstd[[X_VEL  < -offset]and[X_VEL > offset]and[X_RHO > 0.6]]
#X_REF_V    = X_REF[   [X_VEL  < -offset]and[X_VEL > offset]and[X_SW > 10]and[X_SWstd < 15]]
#X_SW_V     = X_SW[    [X_VEL  < -offset]and[X_VEL > offset]and[X_SW > 10]and[X_SWstd < 15]]
#X_SWstd_V  = X_SWstd[ [X_VEL  < -offset]and[X_VEL > offset]and[X_SW > 10]and[X_SWstd < 15]]
#X_RHO_V    = X_RHO[   [X_VEL  < -offset]and[X_VEL > offset]and[X_SW > 10]and[X_SWstd < 15]]
#X_RHOstd_V = X_RHOstd[[X_VEL  < -offset]and[X_VEL > offset]and[X_SW > 10]and[X_SWstd < 15]]
#X_REF_HI = X_REF[X_REF >= 10]
#X_SW_HI  = X_SW[X_REF >= 10]
#X_RHO_HI = X_RHO[X_REF >= 10]

fig               = plt.figure(figsize=(15, 15))
#plt.scatter(X_SW_HI, X_REF_HI, zorder=1)
plt.hist2d(X_SW_V, X_REF_V, bins=30, cmin=0, cmax=1325, cmap='Blues')
# calculate/plot polynomial over radar field scatter plot
#z  = np.polyfit(X_SW_V, X_REF_V, 4)
#p  = np.poly1d(z)
#xp = np.linspace(0, 100, 300)
#plt.plot(xp, p(xp), 'k-', zorder=3)
plt.xlim([0.0,  4.0])
plt.ylim([-15.0,  55])
plt.xlabel('SW_V')
plt.ylabel('REF_V')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_SW_V, X_RHO_V, bins=20, cmin=0, cmax=120, cmap='Blues')
plt.xlim([0.0,  15.0])
plt.ylim([0.2,   0.6])
plt.xlabel('SW_V')
plt.ylabel('RHO_V')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_SWstd_V, X_RHO_V, bins=30, cmin=0, cmax=25, cmap='Blues')
plt.xlim([0.0,  70.0])
plt.ylim([0.2,  1.05])
plt.xlabel('SWstd_V')
plt.ylabel('RHO_V')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_SW_V, X_SWstd_V, bins=30, cmin=0, cmax=225, cmap='Blues')
plt.xlim([0.0,  15.0])
plt.ylim([0.0,  70.0])
plt.xlabel('SW_V')
plt.ylabel('SWstd_V')
cb = plt.colorbar()
cb.set_label('counts in bin')

fig               = plt.figure(figsize=(15, 15))
plt.hist2d(X_SW_V, X_RHOstd_V, bins=30, cmin=0, cmax=425, cmap='Blues')
plt.xlim([0.0,  15.0])
plt.ylim([0.0,  65.0])
plt.xlabel('SW_V')
plt.ylabel('RHOstd_V')
cb = plt.colorbar()
cb.set_label('counts in bin')

#X.shape
#y.shape

# ... example found at: https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
# Generate some data
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
# flip axes for better plotting
X         = X[:, ::-1] 
# Plot the data with K Means Labels
kmeans    = KMeans(4, random_state=0)
labels    = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)
    # plot the input data
    ax     = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii   = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

# for k-means is that these cluster models must be circular: k-means has no built-in way of accounting for oblong or elliptical clusters.
kmeans      = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)
rng         = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))
kmeans      = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)

#-------------------------------------------
# GMM METHOD
#   Similar method to K-means, but specialized for more complex, non-separated data points
#   contains a probabalistic model under the hood
#   can therefore find prob cluster assignments
#   uses an expectation/maximization approach like K-means
#     which start guess for shape and location then repeats until converge
#
#In statistics, an expectation?maximization (EM) algorithm is an iterative method to 
#find (local) maximum likelihood or maximum a posteriori (MAP) estimates of parameters in 
#statistical models, where the model depends on unobserved latent variables. The EM iteration 
#alternates between performing an expectation (E) step, which creates a function for the 
#expectation of the log-likelihood evaluated using the current estimate for the parameters, and 
#a maximization (M) step, which computes parameters maximizing the expected log-likelihood 
#found on the E step. These parameter-estimates are then used to determine the distribution of 
#the latent variables in the next E step.
#-------------------------------------------
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draw an ellipse with a given position and covariance
    """
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle         = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle         = 0
        width, height = 2 * np.sqrt(covariance)
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))     
def plot_gmm(gmm, X, label=True, ax=None):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[22, 22])
    #ax       = ax or plt.gca()
    labels   = gmm.fit(X).predict(X)
    # visualize this uncertainty by making the size of each point proportional to the certainty of its prediction
    size     = 50 * probs.max(1) ** 2  # square emphasizes differences
    if label:
        axs.scatter(X[:, 0], X[:, 1], c=labels, s=size, cmap='viridis', zorder=2, alpha=0.40)
    else:
        axs.scatter(X[:, 0], X[:, 1],           s=size,                 zorder=2)
    #axs.axis('equal')
    axs.set(xlim=(0.00, 0.42), ylim=(0.20, 1.010))
    axs.set_xlabel('stdRHO-HV',                    fontsize = 30)
    axs.set_ylabel('RHO-HV',                       fontsize = 30)
    axs.set_title('Camp Fire: 2018/11/08 15:46UTC, KBBX, GMM (opt-comps=5, covariance=full)', fontsize = 30)
    axs.text(0.36, 0.23, 'N-tot='+str(X.shape[0]), fontsize = 30, c='black')
    axs.text(0.10, 0.95, 'C1: Wx',                 fontsize = 30, c='white')
    axs.text(0.28, 0.95, 'C2: Wx/GC/noise',        fontsize = 30, c='white')
    axs.text(0.16, 0.78, 'C3: unknown/mix',        fontsize = 30, c='white')
    axs.text(0.12, 0.54, 'C4: PYRO-plume',         fontsize = 30, c='white')
    axs.text(0.12, 0.32, 'C5: PYRO-smoke',         fontsize = 30, c='white')
    #axs.xticks(fontsize = 30)
    #axs.yticks(fontsize = 30)
    #axs.set_aspect('box')
    w_factor = 0.2 / gmm.weights_.max()
    #for index, w in enumerate(gmm.weights_):
        #draw_ellipse(gmm.means_[index,:], gmm.covariances_[index,:,:], alpha=gmm.weights_[index] * w_factor)  
# define input fields from radar2 object
RHO          = np.ma.getdata(radar2.fields['cross_correlation_ratio']['data'].flatten())
stdRHO       = np.ma.getdata(radar2.fields['cross_correlation_ratio-std']['data'].flatten())
# basic QC on input data fields
RHOQC        = np.where(               RHO <   0.21, np.nan, RHO)
RHOQC        = np.where(               RHOQC > 1.05, np.nan, RHOQC)
stdRHOQC     = stdRHO[np.logical_not(np.isnan(RHOQC))]
RHOQC        =  RHOQC[np.logical_not(np.isnan(RHOQC))]
# define 2d array of inputs
radar2fields = np.array([stdRHOQC, RHOQC]).transpose()
#
n_comps      = 5 # C1=weather, C2=noise/GC, C3=unknown, C4=plume/pyro, C5=plume/smoke
gmm          = mixture.GaussianMixture(n_components = n_comps).fit(radar2fields)
labels       = gmm.predict(radar2fields)
# find probabilistic cluster assignments
probs        = gmm.predict_proba(radar2fields)
print(probs[:5].round(3))
# plotting       
gmm          = mixture.GaussianMixture(n_components = n_comps, random_state = 42)
plot_gmm(gmm, radar2fields)

# how many components?
#   choice of number of components measures how well GMM works as a density estimator, not how well it works as a clustering algorithm.
n_components = np.arange(1, 21)
models       = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(radar2fields)
                for n in n_components]
fig, axs     = plt.subplots(nrows=1, ncols=1, figsize=[22, 22])
# Set general font size
plt.rcParams['font.size'] = '20'
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
	label.set_fontsize(16)
# Bayesian information criterion (BIC)
axs.plot(n_components, [m.bic(radar2fields) for m in models], label='BIC')
# Akaike information criterion (AIC) 
axs.plot(n_components, [m.aic(radar2fields) for m in models], label='AIC')
plt.legend(loc = 'best', prop={"size":20})
plt.xlabel('n_components', fontsize = 30)
# ... example found at: https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
# Generate some data
#X, y_true    = make_blobs(n_samples=400, centers=4,
#                       cluster_std=0.60, random_state=0)
# flip axes for better plotting
#X            = X[:, ::-1] 
#gmm          = mixture.GaussianMixture(n_components=4).fit(X)
#labels       = gmm.predict(X)
#plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
#probs        = gmm.predict_proba(X)
#print(probs[:5].round(3))
#gmm = mixture.GaussianMixture(n_components=4, random_state=42)
#plot_gmm(gmm, X)
# cov_type controls the degrees of freedom in the shape of each cluster; it is essential to set this carefully for any given problem.
#gmm = mixture.GaussianMixture(n_components=4, covariance_type='full', random_state=42)
#plot_gmm(gmm, X_stretched)

# ... follow same process using CNV_rad_match dataframe fields 
# ......use only index 1 and 9
CNV_rad_match = CNV_rad_match.loc[[1,9], :]
# ......drop all remaining comuns that have NaNs
CNV_rad_match = CNV_rad_match.dropna()
# save as np.array
X_IFI_0   = np.array(CNV_rad_match.loc[[1,9], ' mean.DBZ.FZDZ.pix'])
# save as np.array
X_IFI_1   = np.array(CNV_rad_match.loc[[1,9], ' mean.ZDR.corr.SLW.pix'])
# stack columns to make 2D np.array
X_IFI     = np.column_stack((np.array(CNV_rad_match.loc[[1,9], ' mean.DBZ.FZDZ.pix']), np.array(CNV_rad_match.loc[[1,9], ' mean.ZDR.corr.SLW.pix'])))
# save truth.cat2 index column as np.array
y_IFI     = np.array(CNV_rad_match.loc[[1,9],].index)
y_IFI[y_IFI == 1] = 10

# ..... use elbow method to identify optimal number of clusters
wcss      = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_IFI)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# ...... run KMeans using predetermined optimal number of clusters
kmeans     = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y     = kmeans.fit_predict(X_IFI)
plt.scatter(X_IFI[:,0], X_IFI[:,1], c=y_IFI)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red')
plt.show()

#-------------------------------------------
# SUPPORT VECTOR MACHINES (SVM) METHOD
#-------------------------------------------

#





##############################################################################################

# ... for ICICLE data
ICICLE_target_names = np.asarray(['NONE', 'FZDZ-H', 'FZDZ-L', 'SLW-H', 'SLW-L', 'FZRA', 'MPHA-DZ', 'MPHA-S', 'MPHA-I', 'ICEONLY'])
formatter           = plt.FuncFormatter(lambda i, *args: ICICLE_target_names[int(i)])
cmap                = mpl.cm.viridis
bounds              = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
norm                = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
plt.figure(figsize=(15, 15))
plt.scatter(CNV_rad_match.loc[:, ' NEV.LWC.gm3'], CNV_rad_match.loc[:, ' dmax.85.per.L.um'], c=CNV_rad_match.index)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), format=formatter)
plt.xlim([0, 1])
plt.ylim([0, 1000])
plt.xlabel(' NEV.LWC.gm3')
plt.ylabel(' dmax.85.per.L.um')
plt.tight_layout()
plt.show()

#fig, ax = plt.subplots()
#colors  = {'NONE':'white', 'FZDZ-H':'blue', 'FZDZ-L':'lightblue', 'SLW-H':'darkgreen', 'SLW-L':'green', 'FZRA':'darkblue', 'MPHA':'orange', 'MPHA':'orange', 'MPHA':'orange', 'ICEONLY':'grey'}
#grouped = CNV_rad_match.groupby('truth.cat2')
#for key, group in grouped:
#    group.plot(ax=ax, kind='scatter', x=' ZDR.MOMS.FZDZ.pix', y=' mean.DBZ.FZDZ.pix', label=key, color=colors[key])
#plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(CNV_rad_match.loc[[5], ' ZDR.MOMS.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[5], ' mean.DBZ.FZDZ.pix'].astype(np.float), c='darkblue')
plt.scatter(CNV_rad_match.loc[[2], ' ZDR.MOMS.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[2], ' mean.DBZ.FZDZ.pix'].astype(np.float), c='lightblue')
plt.scatter(CNV_rad_match.loc[[1], ' ZDR.MOMS.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[1], ' mean.DBZ.FZDZ.pix'].astype(np.float), c='blue')
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), format=formatter)
plt.xlim([-2, 2])
plt.ylim([-30, 40])
plt.xlabel(' ZDR.MOMS.FZDZ.pix')
plt.ylabel(' mean.DBZ.FZDZ.pix')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(CNV_rad_match.loc[[4], ' ZDR.MOMS.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[4], ' DBZ.MOMS.SLW.pix'].astype(np.float), c='lightgreen')
plt.scatter(CNV_rad_match.loc[[3], ' ZDR.MOMS.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[3], ' DBZ.MOMS.SLW.pix'].astype(np.float), c='green')
plt.scatter(CNV_rad_match.loc[[5], ' ZDR.MOMS.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[5], ' DBZ.MOMS.SLW.pix'].astype(np.float), c='darkblue')
plt.scatter(CNV_rad_match.loc[[2], ' ZDR.MOMS.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[2], ' DBZ.MOMS.SLW.pix'].astype(np.float), c='lightblue')
plt.scatter(CNV_rad_match.loc[[1], ' ZDR.MOMS.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[1], ' DBZ.MOMS.SLW.pix'].astype(np.float), c='blue')
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), format=formatter)
plt.xlim([-2, 2])
plt.ylim([-30, 40])
plt.xlabel(' ZDR.MOMS.FZDZ.pix')
plt.ylabel(' mean.DBZ.FZDZ.pix')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(CNV_rad_match.loc[[4], ' int.full.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[4], ' int.full.SLW.pix'].astype(np.float), c='lightgreen')
plt.scatter(CNV_rad_match.loc[[3], ' int.full.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[3], ' int.full.SLW.pix'].astype(np.float), c='green')
plt.scatter(CNV_rad_match.loc[[5], ' int.full.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[5], ' int.full.SLW.pix'].astype(np.float), c='darkblue')
plt.scatter(CNV_rad_match.loc[[2], ' int.full.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[2], ' int.full.SLW.pix'].astype(np.float), c='lightblue')
plt.scatter(CNV_rad_match.loc[[1], ' int.full.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[1], ' int.full.SLW.pix'].astype(np.float), c='blue')
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), format=formatter)
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
plt.xlabel(' int.full.FZDZ.pix')
plt.ylabel(' int.full.SLW.pix')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(CNV_rad_match.loc[[9], ' int.full.FZDZ.pix'].astype(np.float), CNV_rad_match.loc[[9], ' int.full.SLW.pix'].astype(np.float), c='grey')
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), format=formatter)
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
plt.xlabel(' int.full.FZDZ.pix')
plt.ylabel(' int.full.SLW.pix')
plt.tight_layout()
plt.show()

# 
plt.figure(figsize=(10, 10))
plt.contour(CNV_rad_match_F17.loc[[17], 'truth.cat2'].values, CNV_rad_match_F17.loc[[17], ' int.full.FZDZ.pix'].values.astype(np.float), c='blue', marker='.')
plt.colorbar()
plt.xlim([0, 9])
plt.ylim([0, 1.1])
plt.xlabel(' truth.cat2')
plt.ylabel(' int.full.FZDZ.pix')
plt.tight_layout()
plt.show()

#RadIA_int_names     = np.asarray(['LOW: 0.0-0.3', 'MEDIUM: 0.3-0.5', 'HIGH: 0.5-1.0'])
#formatter           = plt.FuncFormatter(lambda i, *args: RadIA_int_names[int(i)])
#cmap                = mpl.cm.rainbow
#bounds              = [0.1, 0.4, 0.7]
#norm                = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

# mapview plots of RadIA-m-V2 .....
# ... int.full.FZDZ.pix
plt.figure(figsize=(10, 8))
plt.scatter(CNV_rad_match_F17.loc[[17], ' lon.d'].astype(np.float), CNV_rad_match_F17.loc[[17], ' lat.d'].astype(np.float), c='grey', marker='.', s=1)
plt.scatter(CNV_rad_match_F17.loc[[17], ' lon.d'].astype(np.float), CNV_rad_match_F17.loc[[17], ' lat.d'].astype(np.float), c=CNV_rad_match_F17.loc[[17], ' int.full.FZDZ.pix'].astype(np.float), marker='o', s=50)
plt.xlabel('Lon [deg]')
plt.ylabel('Lat [deg]')
plt.colorbar()
plt.tight_layout()
plt.show()

# mapview plots of RadIA-m-V2 .....
# ... int.full.SLW.pix
plt.figure(figsize=(10, 8))
plt.scatter(CNV_rad_match_F17.loc[[17], ' lon.d'].astype(np.float), CNV_rad_match_F17.loc[[17], ' lat.d'].astype(np.float), c='grey', marker='.', s=1)
plt.scatter(CNV_rad_match_F17.loc[[17], ' lon.d'].astype(np.float), CNV_rad_match_F17.loc[[17], ' lat.d'].astype(np.float), c=CNV_rad_match_F17.loc[[17], ' int.full.SLW.pix'].astype(np.float), marker='o', s=50)
plt.xlabel('Lon [deg]')
plt.ylabel('Lat [deg]')
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(CNV_rad_match_F17.loc[[17], 'truth.cat2'].values,           marker='.', linestyle='None', c='black')
#plt.plot(CNV_rad_match_F17.loc[[17], ' int.full.FZDZ.pix'].values*4, marker='.', c='blue')
plt.show()

#---------------------------------------
# SUPERVISED LEARNING: A. CLASSIFICATION AND B. REGRESSION
#   USING NEAREST NEIGHBOR PREDICTION
#---------------------------------------
# A. CLASSIFICATION
# ... Create color maps for 3-class classification problem, as with iris
cmap_light   = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold    = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# ... supervised learning: classification and regression
iris         = datasets.load_iris()
X            = iris.data[:, :2]  # we only take the first two features. We could
                               # avoid this ugly slicing by using a two-dim dataset
y            = iris.target
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy       = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
# ... calc for n=1
knn_n1       = neighbors.KNeighborsClassifier(n_neighbors=1)
knn_n1.fit(X, y)
Z_n1         = knn_n1.predict(np.c_[xx.ravel(), yy.ravel()])
Z_n1         = Z_n1.reshape(xx.shape)
# ... calc for n=3
knn_n3       = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_n3.fit(X, y)
Z_n3         = knn_n3.predict(np.c_[xx.ravel(), yy.ravel()])
Z_n3         = Z_n3.reshape(xx.shape)

# ... plotting
plt.figure()
plt.pcolormesh(xx, yy, Z_n1, cmap=cmap_light)
# ... Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.axis('tight')
plt.figure()
plt.pcolormesh(xx, yy, Z_n3, cmap=cmap_light)
# ... Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.axis('tight')
# ... What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
print(iris.target_names[knn.predict([[3, 5, 4, 2]])])

# B. REGRESSION
# ... of iris data
# ...... x from 0 to 30
x     = 30 * np.random.random((20, 1))
# ...... y = a*x + b with noise
y     = 0.5 * x + 1.0 + np.random.normal(size=x.shape)
# ...... create a linear regression model
model = LinearRegression()
# ...... FIT TRAINING DATA.  for SL, accepts data (x) and labels (y). for UL, accept data (x) only
model.fit(x, y)  				
# ...... predict y from the data
x_new = np.linspace(0, 30, 100)
y_new = model.predict(x_new[:, np.newaxis])	# used for SL.  given trained model, predict label of x_new data, returns y_new
# also model.predict_proba() method returns probability that new obs has categorical label. label with highest prob returned by model.predict()
# for UL, model.transform() and model.fit_transform() are used
# ...... plot the results
plt.figure(figsize=(4, 3))
ax    = plt.axes()
ax.scatter(x, y)
ax.plot(x_new, y_new)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('tight')
plt.show()

# ... of Boston housing data
for index, feature_name in enumerate(data.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(data.data[:, index], data.target)
    plt.ylabel('Price', size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()
    
####################################################################
############################  LEFT OFF HERE 
####################################################################

# ... of ICICLE data
rad_match_feature_names = np.asarray([' T.c', 
			              ' mean.DBZ.FZDZ.pix', ' mean.ZDR.corr.SLW.pix', ' RHO.MOMS.FZDZ.pix',
			              ' sdev.DBZ.FZDZ.pix'])
CNV_match_feature_names = np.asarray([' NEV.LWC.gm3', ' NEV.IWC.gm3', ' T.DBZ.FZDZ.pix', ' T.c', 
			       ' dmax.85.per.L.um', ' NAW.zennad.REFL.30s.mean.updown', ' NAW.zennad.VEL.30s.mean.updown'])
CNV_rad_match2          = CNV_rad_match.dropna()
for index, feature_name in enumerate(rad_match_feature_names):
    print(index)
    plt.figure(figsize=(10, 5))
    if feature_name == str(' T.c'):
#        plt.scatter(CNV_rad_match.loc[9, rad_match_feature_names[index]], CNV_rad_match.loc[9, rad_match_feature_names[index+1]], c='black')
        plt.scatter(CNV_rad_match2.loc[1, rad_match_feature_names[index]], CNV_rad_match2.loc[1, rad_match_feature_names[index+1]], c='red')
        plt.xlim(-40, 20)
    else:
#        plt.scatter(CNV_rad_match.loc[9, feature_name], CNV_rad_match.loc[9, rad_match_feature_names[index+1]], c='black')
        plt.scatter(CNV_rad_match.loc[1, feature_name], CNV_rad_match.loc[1, rad_match_feature_names[index+1]], c='red')
#    plt.scatter(CNV_rad_match.loc[:, CNV_rad_match_feature_names[index]], CNV_rad_match.loc[:, 'truth.cat2'])
    plt.ylabel(rad_match_feature_names[index+1], size=15)
    plt.xlabel(feature_name, size=15)
#    plt.xticks(np.arange(min(x), max(x), (max(x)-min(x))/2))
#    plt.yaxis.set_major_locator(MaxNLocator(5))
#    plt.locator_params(axis='y', nbins=6)
    plt.tight_layout()

####################################################################
############################  LEFT OFF HERE 
####################################################################






