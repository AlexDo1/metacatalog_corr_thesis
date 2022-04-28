from cProfile import label
import seaborn as sns
from metacatalog import api
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def set_seaborn_style():
    """
    Function to set the plotting style for seaborn.
    Adjust style here for all figures that should go in the final thesis.
    Call function before plotting.
    """
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    
    # Set the font to be serif, rather than sans
    sns.set(font='serif', rc={"font.size":11,"axes.titlesize":11,"axes.labelsize":10, 'legend.fontsize': 9})

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "patch.edgecolor": "w",
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def load_entrygroups(session):
    """
    Return a list with all EntryGroups from mc_corr.
    The LUBW group is returned as an ImmutableResultSet due to its split datasets.
    """
    entry_groups = []
    entry_groups.extend(api.find_group(session, type=1, title='LTZ Augustenberg'))
    entry_groups.extend(api.find_group(session, type=1, title='DWD station Rheinstetten'))
    entry_groups.extend(api.find_group(session, type=1, title='Bühlot Dataset'))
    entry_groups.extend(api.find_group(session, type=1, title='Sap Flow - Hohes Holz'))

    # LUBW gauge network: Split datasets -> get result set to merge Split datasets
    entry_groups.extend(api.find_group(session, type=1, title='LUBW gauge network', as_result=True))

    entry_groups.extend(api.find_group(session, type=2, title='*Eddy*'))
    entry_groups.extend(api.find_group(session, type=4))


    for g in entry_groups:
        if str(type(g)) == "<class 'metacatalog.util.results.ImmutableResultSet'>":
            print(g.group.title)
        else:
            print(g.title)

    return entry_groups


def get_wide_df(session, filter_identifier):
    """
    Load table correlation_matrix as table in wide format: metrics as columns.
    e.g. filter_identifier == 'case01': load case01 entries from correlation_matrix
    """

    sql=f"select * from correlation_matrix where identifier like '%%{filter_identifier}%%' order by left_id, right_id, metric_id"
    df = pd.read_sql(sql, session.bind) 

    # table long to wide format: metrics as columns
    df = pd.pivot_table(df, values=['value'], index=['left_id', 'right_id','identifier'], columns='metric_id').reset_index()

    # get metrics for output column names
    sql = 'SELECT id, symbol FROM correlation_metrics'
    df_metrics = pd.read_sql(sql, session.bind)
    dict_metrics = dict(zip(df_metrics.id, df_metrics.symbol))

    # rename metric_id to metric_name
    df.rename(columns=dict_metrics, inplace=True)

    # Multiindex column names from pivot_table() -> flatten to one level
    col_names = []
    for col in df.columns:
        if col[0] == 'value':
            col_names.append(col[1])
        else:
            col_names.append(col[0])
    df.columns = col_names

    # split up variables in identifier into new columns left_variable, right_variable
    for idx, row in df.iterrows():
        df.loc[idx, 'left_variable'] = df.loc[idx, 'identifier'].split('[')[1].split(']')[0].replace(',', '').split()[0]
        df.loc[idx, 'right_variable'] = df.loc[idx, 'identifier'].split('[')[1].split(']')[0].replace(',', '').split()[1]

    return df


def harmonize_data(left, right):
    """
    Function to harmonize the data from metacatalog like in the metric calculation in 
    metacatalog_corr with harmonize=True
    """
    harmonized_index = right[right.index.isin(left.index)].index
    left = left.loc[harmonized_index]
    right = right.loc[harmonized_index]
    
    left = left.to_numpy()
    right = right.to_numpy()
    
    left = np.hstack(left)
    right = np.hstack(right)
    
    nan_indices = np.logical_not(np.logical_or(np.isnan(left), np.isnan(right)))
    left = left[nan_indices]
    right = right[nan_indices]
                  
    return left, right



from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

def clustering_and_plot_dendrogram(wide_df, metric, compare_to_pearson=False, ax=None, **kwargs):
    """
    This is the main clustering function
    It computes the hierarchical clustering for 1 metric from the wide dataframe created aboce and also plots the corresponding dendrogram. The mean value of the previously defined expected correlation is shown on the x-axis.

    Another possibility: compare mean value of metric in cluster to mean value of pearson in cluster (compare_to_pearson=True)
    
    Paramters:
    wide_df: pd.DataFrame
        Wide dataframe with metrics as columns. Use function get_wide_df()
    metric: str
        Metric for clustering computation and plotting, must be a column in wide_df, eg 'pearson'
    compare_to_pearson: bool
        if True, the mean value of pearson is shown for each column, instead of the mean value
        of expected_correlation
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis to draw dendrogram on.
    """
    
    
    
    
    
    # calculate clustering
    # absolute values of metric column
    X = np.asarray(abs(wide_df[metric]))
    # remove NaN
    X = X[~np.isnan(X)]
    # data has a single feature: reshape
    X = X.reshape(-1, 1)
    # linkage = 'ward': Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach. 
    # (https://scikit-learn.org/stable/modules/clustering.html#:~:text=the%20merge%20strategy%3A-,Ward,-minimizes%20the%20sum)
    clustering = AgglomerativeClustering(n_clusters=4, linkage='ward', compute_distances=True).fit(X)
    
    # dendrogram linewidth
    matplotlib.rcParams['lines.linewidth'] = 2.5

    def plot_dendrogram(model, ax=ax, **kwargs):
        # Create linkage matrix and then plot the dendrogram if no_plot==False
    
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
    
        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)
    
        # Plot the corresponding dendrogram
        if kwargs.get('no_plot', True):
            return dendrogram(linkage_matrix, count_sort='descending', **kwargs)
        else:
            dendrogram(linkage_matrix, count_sort='descending', ax=ax, **kwargs)

    if ax:
        ax.set_title(f"Hierarchical Clustering for {metric}")
    else:
        plt.title(f"Hierarchical Clustering for {metric}")
    
    # no_plot: get clustering dict
    dendrogram_dict = plot_dendrogram(clustering, truncate_mode='lastp', p=4, no_plot=True)

    # drop na if exists
    nan_index = wide_df.loc[pd.isna(wide_df[metric]), :].index
    wide_df.drop(nan_index, inplace=True)

    # create labels for each leaf
    label_list = []
    for ivl in dendrogram_dict['ivl']:
        # convert ivl to int
        ivl = int(ivl.split('(')[1].split(')')[0])
        # label_list: (number of points in cluster, mean value of cluster, expected correlation in cluster)
        for i in range(len(dendrogram_dict['ivl'])):
            if len(wide_df[metric][clustering.labels_ == i]) == ivl:
                cluster_points = ivl
                cluster_mean = round(abs(wide_df[metric])[clustering.labels_ == i].mean(), 3)
                if compare_to_pearson:
                    cluster_pearson_mean = round(abs(wide_df.pearson)[clustering.labels_ == i].mean(), 2)  
                    label_list.append(f"{cluster_points, cluster_mean, cluster_pearson_mean}")
                else:
                    cluster_exp_corr_mean = round(abs(wide_df.expected_corr)[clustering.labels_ == i].mean(), 1) 
                    label_list.append(f"{cluster_points, cluster_mean, cluster_exp_corr_mean}")
    
    # create label dictionary with leaves as keys          
    label_dict = {dendrogram_dict["leaves"][ii]: label_list[ii] for ii in range(len(dendrogram_dict["leaves"]))}
    
    # create leaf label function, returing label for each leaf
    def llf(leaf):
        return label_dict[leaf]
    
    # plot the dendrogram
    plot_dendrogram(clustering, truncate_mode='lastp', p=4, no_plot=False, leaf_label_func=llf, ax=ax)
    
    # ylabel
    if ax:
        ax.set_ylabel('Distance')
    else:
        plt.ylabel('Distance')

    # xlabel
    if compare_to_pearson:
        if ax:
            ax.set_xlabel(f"(Number of points in node, Mean value of {metric}, Average correlation score of Pearson")
        else:
            plt.xlabel(f"(Number of points in node, Mean value of {metric}, Average correlation score of Pearson")
    else:
        if ax:
            ax.set_xlabel(f"(Number of points in node, Average value of {metric.capitalize()}, Average value of expected correlation")
        else:
            plt.xlabel(f"(Number of points in node, Average value of {metric.capitalize()}, Average value of expected correlation")    
    
    plt.tight_layout()
    
    return clustering, label_dict