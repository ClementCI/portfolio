import hdbscan

from collections import Counter

# ==========================================================
#  HDBSCAN Clustering and filtering
# ==========================================================
def cluster_embeddings(embs, min_cluster_size):
    """
    Runs HDBSCAN with euclidean distance and eom cluster selection method.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(embs)
    return labels
    
def filter_embeddings(labels):
    """
    Filters embeddings by keeping points from the largest cluster.
    """
    counts = Counter(labels)
    largest_cluster_id = max(counts, key=counts.get)                                  # extract largest cluster's id
    main_idx = [idx for idx, label in enumerate(labels) if label==largest_cluster_id] # extract indexes of the main embeddings
    return main_idx
