#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:


# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd


# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid')

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

# to scale the data using z-score
from sklearn.preprocessing import StandardScaler

# to compute distances
from scipy.spatial.distance import cdist, pdist

# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# to visualize the elbow curve and silhouette scores
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# to perform hierarchical clustering, compute cophenetic correlation, and create dendrograms
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

# to suppress warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


url= "dataset/stock_data.csv"
stock_data = pd.read_csv(url)
stock_data.head(10)


# Checking Rows and Columns

# In[3]:


stock_data.shape


# Checking Data Types

# In[4]:


stock_data.info()


# Duplicating Original Data

# In[5]:


stock_df = stock_data.copy()


# Checking Missing Values and Duplicates

# In[6]:


stock_df.isnull().sum()


# In[7]:


stock_df.duplicated().sum()


# Summary of Statistical Data

# In[8]:


stock_df.describe(include='all').T


# Univariate Analysis

# In[12]:


# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(stock_df, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=stock_df, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=stock_df, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=stock_df, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        stock_df[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        stock_df[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# Current Price

# In[13]:


histogram_boxplot(stock_df, 'Current Price')


# Price Change 

# In[14]:


histogram_boxplot(stock_df, 'Price Change')


# Volatility

# In[15]:


histogram_boxplot(stock_df, 'Volatility')


# ROE

# In[16]:


histogram_boxplot(stock_df, 'ROE')


# Cash Ratio 

# In[17]:


histogram_boxplot(stock_df, 'Cash Ratio')


# Net Cash Flow

# In[18]:


histogram_boxplot(stock_df, 'Net Cash Flow')


# Net Income

# In[19]:


histogram_boxplot(stock_df, 'Net Income')


# Earnings Per Share

# In[20]:


histogram_boxplot(stock_df, 'Earnings Per Share')


# Estimated Shares Outstanding

# In[21]:


histogram_boxplot(stock_df, 'Estimated Shares Outstanding')


# P/E Ratio

# In[22]:


histogram_boxplot(stock_df, 'P/E Ratio')


# P/B Ratio

# In[23]:


histogram_boxplot(stock_df, 'P/B Ratio')


# In[24]:


# function to create labeled barplots


def labeled_barplot(stock_df, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(stock_df[feature])  # length of the column
    count = stock_df[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=stock_df,
        x=feature,
        palette="Paired",
        order=stock_df[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# GICS Sector

# In[26]:


labeled_barplot(stock_df, 'GICS Sector', perc=True)


# GICS Sub Industry

# In[27]:


labeled_barplot(stock_df, 'GICS Sub Industry', perc=True)


# Bivariate Analysis

# In[28]:


# correlation check
plt.figure(figsize=(15, 7))
sns.heatmap(
    stock_df.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
)
plt.show()


# Stocks of the economic sector that has seen the maximum price increase on average

# In[31]:


plt.figure(figsize=(15,8))
sns.barplot(data=stock_df, x='GICS Sector', y='Price Change', ci=False) 
plt.xticks(rotation=90)
plt.show()


# Average cash ratio varies across economic sectors

# In[35]:


plt.figure(figsize=(15,8))
sns.barplot(data=stock_df, x='GICS Sector', y='Cash Ratio', ci=False)  
plt.xticks(rotation=90)
plt.show()


#  How P/E ratio varies, on average, across economic sectors

# In[37]:


plt.figure(figsize=(15,8))
sns.barplot(data=stock_df, x='GICS Sector', y='P/E Ratio', ci=False) 
plt.xticks(rotation=90)
plt.show()


# How volatility varies, on average, across economic sectors

# In[38]:


plt.figure(figsize=(15,8))
sns.barplot(data=stock_df, x='GICS Sector', y='Volatility', ci=False) 
plt.xticks(rotation=90)
plt.show()


# Data Preprocessing

# Outlier check 

# In[40]:


plt.figure(figsize=(15, 12))

numeric_columns = stock_df.select_dtypes(include=np.number).columns.tolist()

for i, variable in enumerate(numeric_columns):
    plt.subplot(3, 4, i + 1)
    plt.boxplot(stock_df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# Scaling

# In[44]:


# Selecting only numeric columns
numeric_columns = stock_df.select_dtypes(include=['float64', 'int64'])

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply the scaler to the numeric columns
subset_scaled = scaler.fit_transform(numeric_columns)

# Creating a DataFrame of the scaled data
subset_scaled_df = pd.DataFrame(subset_scaled, columns=numeric_columns.columns)


# K-Means Clustering

# Checking Elbow Plot

# In[46]:


k_means_df = subset_scaled_df.copy()


# In[47]:


clusters = range(1, 15)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k, random_state=1)
    model.fit(subset_scaled_df)
    prediction = model.predict(k_means_df)
    distortion = (
        sum(np.min(cdist(k_means_df, model.cluster_centers_, "euclidean"), axis=1))
        / k_means_df.shape[0]
    )

    meanDistortions.append(distortion)

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)

plt.plot(clusters, meanDistortions, "bx-")
plt.xlabel("k")
plt.ylabel("Average Distortion")
plt.title("Selecting k with the Elbow Method", fontsize=20)
plt.show()


# In[48]:


model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(1, 15), timings=True)
visualizer.fit(k_means_df)  # fit the data to the visualizer
visualizer.show() 


# Silhoutte Scores

# In[49]:


sil_score = []
cluster_list = range(2, 15)
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters, random_state=1)
    preds = clusterer.fit_predict((subset_scaled_df))
    score = silhouette_score(k_means_df, preds)
    sil_score.append(score)
    print("For n_clusters = {}, the silhouette score is {})".format(n_clusters, score))

plt.plot(cluster_list, sil_score)
plt.show()


# In[50]:


model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2, 15), metric="silhouette", timings=True)
visualizer.fit(k_means_df)  # fit the data to the visualizer
visualizer.show()  # finalize and render figure


# In[52]:


# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(n_clusters=3, random_state=1))  
visualizer.fit(k_means_df)
visualizer.show()


# Final Model

# In[56]:


# final K-means model
kmeans = KMeans(n_clusters=3, random_state=1) 
kmeans.fit(k_means_df)


# In[57]:


# creating a copy of the original data
df1 = stock_df.copy()

# adding kmeans cluster labels to the original and scaled dataframes
k_means_df["KM_segments"] = kmeans.labels_
df1["KM_segments"] = kmeans.labels_


# Cluster Profiling

# In[61]:


km_cluster_profile = df1.groupby("KM_segments").mean()  


# In[62]:


km_cluster_profile["count_in_each_segment"] = (
    df1.groupby("KM_segments")["Security"].count().values)


# In[63]:


km_cluster_profile.style.highlight_max(color="lightgreen", axis=0)


# In[64]:


for cl in df1["KM_segments"].unique():
    print("In cluster {}, the following companies are present:".format(cl))
    print(df1[df1["KM_segments"] == cl]["Security"].unique())
    print()


# In[65]:


df1.groupby(["KM_segments", "GICS Sector"])['Security'].count()


# In[67]:


plt.figure(figsize=(20, 20))
plt.suptitle("Boxplot of numerical variables for each cluster")

# selecting numerical columns
num_col = stock_df.select_dtypes(include=np.number).columns.tolist()

for i, variable in enumerate(num_col):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(data=df1, x="KM_segments", y=variable)

plt.tight_layout(pad=2.0)


# Computing Cophenetic Correlation

# In[73]:


hc_df = subset_scaled_df.copy()


# In[80]:


from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

# list of distance metrics
distance_metrics = ['euclidean', 'cosine', 'hamming']  

# list of linkage methods
linkage_methods = ['ward', 'single', 'complete', 'average']

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        # Skip incompatible combinations (Ward method with non-Euclidean distances)
        if lm == 'ward' and dm != 'euclidean':
            continue
        
        Z = linkage(hc_df, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(hc_df))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm

# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print('*'*100)
print(
    "Highest cophenetic correlation is {}, which is obtained with {} distance and {} linkage.".format(
        high_cophenet_corr, high_dm_lm[0].capitalize(), high_dm_lm[1]
    )
)


# In[81]:


# list of linkage methods
linkage_methods = ['ward', 'single', 'complete', 'average']

high_cophenet_corr = 0
high_dm_lm = ["euclidean", 0]  # Euclidean distance metric is fixed

for lm in linkage_methods:
    Z = linkage(hc_df, metric="euclidean", method=lm)
    c, coph_dists = cophenet(Z, pdist(hc_df))
    print("Cophenetic correlation for {} linkage is {}.".format(lm, c))
    if high_cophenet_corr < c:
        high_cophenet_corr = c
        high_dm_lm[1] = lm
        
# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print('*'*100)
print(
    "Highest cophenetic correlation is {}, which is obtained with {} linkage.".format(
        high_cophenet_corr, high_dm_lm[1]
    )
)


# Dendograms

# In[82]:


# list of linkage methods
linkage_methods = ['ward', 'single', 'complete', 'average']

# lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]
compare = []

# to create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(hc_df, metric="euclidean", method=method)

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")

    coph_corr, coph_dist = cophenet(Z, pdist(hc_df))
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )

    compare.append([method, coph_corr])

plt.tight_layout()
plt.show()


# In[83]:


# create and print a dataframe to compare cophenetic correlations for different linkage methods
df_cc = pd.DataFrame(compare, columns=compare_cols)
df_cc = df_cc.sort_values(by="Cophenetic Coefficient")
df_cc


# Creating model using sklearn

# In[84]:


# Define the number of clusters
n_clusters = 3  
# Define the affinity (distance metric)
affinity = 'euclidean' 

# Define the linkage method
linkage = 'ward'  

# Create the Agglomerative Clustering model
HCmodel = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)

# Fit the model to your data
HCmodel.fit(hc_df)


# In[86]:


# creating a copy of the original data
df2 = stock_df.copy()

# adding hierarchical cluster labels to the original and scaled dataframes
hc_df["HC_segments"] = HCmodel.labels_
df2["HC_segments"] = HCmodel.labels_


# Cluster Profiling

# In[88]:


hc_cluster_profile = df2.groupby("HC_segments").mean()


# In[90]:


hc_cluster_profile["count_in_each_segment"] = (
    df2.groupby("HC_segments")["Security"].count().values  
)


# In[91]:


hc_cluster_profile.style.highlight_max(color="lightgreen", axis=0)


# In[94]:


for cl in df2["HC_segments"].unique():
    print("In cluster {}, the following companies are present:".format(cl))
    print(df2[df2["HC_segments"] == cl]["Security"].unique())
    print()


# In[95]:


df2.groupby(["HC_segments", "GICS Sector"])['Security'].count()


# In[96]:


plt.figure(figsize=(20, 20))
plt.suptitle("Boxplot of numerical variables for each cluster")

for i, variable in enumerate(num_col):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(data=df2, x="HC_segments", y=variable)

plt.tight_layout(pad=2.0)


# K-means vs Hierarchical Clustering

# In[71]:


import time
from sklearn.cluster import KMeans, AgglomerativeClustering

# K-means Clustering
start_time = time.time()
kmeans = KMeans(n_clusters=3, random_state=1)  
kmeans.fit(subset_scaled_df)
kmeans_time = time.time() - start_time

# Hierarchical Clustering
start_time = time.time()
hierarchical = AgglomerativeClustering(n_clusters=3)  
hierarchical_labels = hierarchical.fit_predict(subset_scaled_df)
hierarchical_time = time.time() - start_time

print("K-means Time: ", kmeans_time)
print("Hierarchical Clustering Time: ", hierarchical_time)


# - K- means clustering took more time than Hierarchical clustering
# - n_clusters = 3, with a  silhouette score is 0.4644405674779404) has the most distinct clusters
# - for both K-means and Hierarchical Clustering with n_clusters = 3, it appears that both methods achieved the same silhouette score of approximately 0.4644. This indicates that, in terms of the silhouette score, both methods have performed equally well in creating distinct and well-separated clusters for your dataset.

# Insights and Recommendations
# - Based on the silhouette scores for both K-means and Hierarchical Clustering with n_clusters = 3, it appears that both methods achieved the same silhouette score of approximately 0.4644. This indicates that, in terms of the silhouette score, both methods have performed equally well in creating distinct and well-separated clusters for your dataset.
# - Diversified Portfolio Construction: Advise investors to build portfolios that include stocks from each cluster to achieve a diversified investment strategy.
# - Regular Reassessment: Regularly reassess the clusters as market conditions change. The composition and characteristics of clusters might evolve over time, necessitating adjustments in investment strategies.
# - Further Analysis with Additional Data: Consider incorporating additional data points such as dividend yield, market capitalization, or macroeconomic indicators to refine the clustering analysis and investment strategies.
