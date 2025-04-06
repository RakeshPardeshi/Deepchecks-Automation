import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

def get_bin_edges(expected, bins, method="equal_width"):
    if method == "equal_width":
        return np.linspace(min(expected), max(expected), bins + 1)
    
    elif method == "equal_freq":
        return np.percentile(expected, np.linspace(0, 100, bins + 1))
    
    elif method == "adaptive":
        X = np.array(expected).reshape(-1, 1)
        y = np.digitize(expected, bins=np.percentile(expected, np.linspace(0, 100, bins)))
        tree = DecisionTreeClassifier(max_leaf_nodes=bins)
        tree.fit(X, y)
        return np.sort(tree.tree_.threshold[tree.tree_.threshold > 0])
    
    elif method == "kmeans":
        data = np.array(expected).reshape(-1, 1)
        kmeans = KMeans(n_clusters=bins, random_state=42).fit(data)
        return np.sort(kmeans.cluster_centers_.flatten())

    elif method == "domain":
        return bins  # User-defined domain-specific bins (must be passed as a list)
    
    else:
        raise ValueError("Invalid binning method!")

def psi(expected, actual, bins=10, method="equal_width"):
    bin_edges = get_bin_edges(expected, bins, method)
    bin_edges = np.unique(bin_edges)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_perc = expected_counts / np.sum(expected_counts)
    actual_perc = actual_counts / np.sum(actual_counts)

    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)

    psi_values = (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)

    psi_df = pd.DataFrame({
        "Min Bin": bin_edges[:-1],
        "Max Bin": bin_edges[1:],
        "Expected Count": expected_counts,
        "Actual Count": actual_counts,
        "Expected %": expected_perc,
        "Actual %": actual_perc,
        "PSI Value": psi_values
    })

    total_psi = np.sum(psi_values)

    return {
        "Binning Strategy": method,
        "PSI DataFrame": psi_df,
        "Final PSI": total_psi
    }
