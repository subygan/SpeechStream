from typing import Tuple, List

import torch

from src.offline_clustering import getCosAffinityMatrix


def calculate_removable_counts(removable_counts_mat: torch.Tensor, remain_count: int, num_clus: int) -> torch.Tensor:
    """
    Calculate removable counts based on the arguments and calculate how many counts should be
    removed from the each cluster. This function has `O(N)` (N = num_clus) time complexity to
    return the desired `removable_counts_mat`.

    Example:

        The original input to `get_merge_quantity` function:
        >>> pre_clus_labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        >>> num_to_be_removed = 3
        >>> min_count_per_cluster = 2

        Histogram: (`min_count_per_cluster`=2 is removed)
        0 |*****
        1 |***
        2 |*

        Inputs:
            >>> removable_counts_mat = [5, 3, 1]
            >>> remain_count = 6
            >>> num_clus = 3

        Interim results:
            >>> diff_counts
            [1, 2, 2]
            >>> gradual_counts
            [3, 4, 2]
            >>> cumsum_counts
            [3, 7, 9]

        Return:
            >>> removable_counts_mat
            [2, 1, 0]

    Args:
        removable_counts_mat (Tensor):
            Tensor containing how many vectors could be removed from each cluster
        remain_count (int):
            Integer value that indicates the number of vectors removed from the total set
        num_clus (int):
            Number of clusters in the given label sequence (cardinality of a label set)

    Returns:
        removable_counts_mat (Tensor):
            Tensor containing the number of vectors should be removed from each cluster
    """
    device = removable_counts_mat.device
    zero_padded_counts = torch.cat(
        [torch.tensor([0]).to(device), removable_counts_mat.sort()[0], torch.tensor([0]).to(device)], dim=0
    )
    removable_count_args = removable_counts_mat.sort(descending=True)[1]

    # Calculate the size difference between clusters
    diff_counts = (zero_padded_counts[1:] - zero_padded_counts[:-1])[:num_clus]
    gradual_counts = torch.arange(num_clus, 0, -1).to(device) * diff_counts
    cumsum_counts = torch.cumsum(gradual_counts, dim=0)
    remain_count_rem = remain_count

    # Find how many remaining counts we can use
    ind: int = 0
    for ind, num in enumerate(cumsum_counts):
        if remain_count < num:
            break

    # Subtract the common values step by step
    if ind > 0:
        for knd in range(ind):
            removable_counts_mat[removable_count_args[: num_clus - knd]] -= diff_counts[knd]
            remain_count_rem -= int(diff_counts[knd].item()) * (num_clus - knd)
    assert remain_count >= 0, "remain_count should never be negative."

    # Add remaining values
    num_labels = remain_count_rem // (num_clus - ind)
    rem_labels = remain_count_rem % (num_clus - ind)
    removable_counts_mat[removable_count_args[: (num_clus - ind)]] -= num_labels
    removable_counts_mat[removable_count_args[:rem_labels]] -= 1
    return removable_counts_mat.int()

def get_merge_quantity(
    num_to_be_removed: int, pre_clus_labels: torch.Tensor, min_count_per_cluster: int,
) -> torch.Tensor:
    """
    Determine which embeddings we need to reduce or merge in history buffer.
    We want to merge or remove the embedding in the bigger cluster first.
    At the same time, we keep the minimum number of embedding per cluster
    with the variable named min_count_per_cluster.

    Constraint:
        - Each cluster should keep the number of vectors over `min_count_per_cluster`.
        - In total, `num_to_be_removed` of vectors should be removed from the total buffer.
        - While merging embeddings, minimize the gap between quantities between clusters.

    Example:
        >>> num_to_be_removed = 3
        >>> pre_clus_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        >>> min_count_per_cluster = 2
        >>> get_merge_quantity(num_to_be_removed, pre_clus_labels, min_count_per_cluster)
        Return:
            torch.tensor([2, 1, 0])
        >>> # Sum should be equal to `num_to_be_removed` which is 3

    Args:
        num_to_be_removed: (int)
            the quantity of the newly obtained embedding from the new stream of input.
        pre_clus_labels: (Tensor)
            the speaker labels of (the history_embedding_buffer_emb) + (the new embeddings to be added)
        min_count_per_cluster: (int)
            Minimum vector quantity for each cluster

    Returns:
        removable_counts_mat: (Tensor)
            Tensor containing the number of vectors should be removed from each cluster
    """
    if num_to_be_removed > pre_clus_labels.shape[0] - 1:
        raise ValueError(f"num_to_be_removed: {num_to_be_removed} should be less than pre_clus_labels length - 1")
    remain_count = pre_clus_labels.shape[0] - num_to_be_removed
    spk_freq_count = torch.bincount(pre_clus_labels)
    num_clus = len(torch.unique(pre_clus_labels))
    if remain_count < min_count_per_cluster * num_clus:
        raise ValueError(f"The remaining embedding vectors should be more than { min_count_per_cluster * num_clus }")

    # Minimum vector counts should be excluded from the removable amount
    min_seg_count = torch.tensor([min_count_per_cluster] * len(spk_freq_count)).to(pre_clus_labels.device)
    min_seg_count_mat = torch.stack((min_seg_count, spk_freq_count)).min(0)[0]

    # Exclude minimum quantities from the removable count matrix
    remain_count -= int(torch.sum(min_seg_count_mat))
    removable_counts_mat = spk_freq_count - min_seg_count_mat

    # Calculate removable counts from `remain_count` variable
    removable_counts_mat = calculate_removable_counts(removable_counts_mat, remain_count, num_clus)
    if int(removable_counts_mat.sum()) != num_to_be_removed:
        raise ValueError("Sum of `removable_counts_mat` is not equal to `num_to_be_removed` variable.")
    if not torch.all(removable_counts_mat >= 0) or not torch.all(spk_freq_count - min_seg_count_mat >= 0):
        raise ValueError(
            f"Every value in `removable_counts_mat` should be always non-negative value but got {removable_counts_mat}"
        )
    return removable_counts_mat


def get_closest_embeddings(affinity_mat: torch.Tensor, n_closest: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the indices of the embedding vectors we want to merge.

    Example:
        >>> n_closest = 2
        >>> affinity_mat = [[1.0, 0.2, 0.8],
                            [0.2, 1.0, 0.4],
                            [0.8, 0.4, 1.0]]
        >>> affinity_mat.sum(0)
        [2.0, 1.6, 2.2]

        # The closest two embedding vectors are at index 0 and 2.

    Args:
        affinity_mat: (Tensor)
            Symmetric affinity matrix of the given embedding vector set.
        n_closest (int):
            The amount of vector counts that are expected to be removed from the set
            Example:
                Input: 10 vectors in a set
                n_closest = 5
                (5+1) vectors are merged into 1 vector
                Output: 5 vectors in a set

    Returns:
        idx_aff_sum (torch.Tensor):
            Indices of the closest `n_closest` embedding vectors
        rest_inds (torch.Tensor):
            Indices of the complementary set of the indices in `idx_aff_sum`
    """
    comb_limit = int(affinity_mat.shape[0] - 1)
    if n_closest > comb_limit:
        raise ValueError(f"Got n_closest of {n_closest}: {n_closest} is bigger than comb_limit {comb_limit}")

    # Take summed values over one axis
    sum_cmat = affinity_mat.sum(0)

    # `n_closest + 1` will become 1 embedding vector after merging
    idx_aff_sum = torch.argsort(sum_cmat, descending=True)[: (n_closest + 1)]
    rest_inds = torch.argsort(sum_cmat, descending=True)[(n_closest + 1) :]
    return idx_aff_sum, rest_inds

def merge_vectors(
    selected_inds: torch.Tensor, emb_ndx: torch.Tensor, pre_cluster_labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge feature (embedding) vectors estimated to be the same cluster label.

    Args:
        selected_inds (Tensor):
            Selected indices for merging
        emb_ndx (Tensor):
            Feature (embedding) vectors
            Dimension: (original vector counts) x (feature dimension)
        pre_cluster_labels (Tensor):
            Original cluster labels before merging

    Returns:
        merged_vecs (Tensor):
            Merged feature vectors that are concatenated
            Dimension: (merged vector counts) x (feature dimension)
        merged_clus_labels (Tensor):
            Cluster labels for the merged feature vectors
            Dimension: (merged vector counts)
    """
    if emb_ndx.shape[0] != pre_cluster_labels.shape[0]:
        raise ValueError("pre_cluster_labels and emb_ndx have mismatch in dimension")
    avg_emb = torch.mean(emb_ndx[selected_inds, :], dim=0)
    merged_clus_labels = pre_cluster_labels[selected_inds]
    selected_inds_list: List[int] = selected_inds.tolist()
    bypass_inds_list: List[int] = []
    for k in range(emb_ndx.shape[0]):
        if k not in selected_inds_list:
            bypass_inds_list.append(k)
    bypass_inds = torch.tensor(bypass_inds_list)
    selected_inds = torch.tensor(selected_inds_list)
    if bypass_inds.shape[0] == 0:
        merged_vecs = avg_emb.unsqueeze(0)
        merged_clus_labels = merged_clus_labels.unsqueeze(0)
    else:
        merged_vecs = torch.vstack((emb_ndx[bypass_inds], avg_emb))
        merged_clus_labels = torch.hstack((pre_cluster_labels[bypass_inds], merged_clus_labels[0]))
    return merged_vecs, merged_clus_labels

def run_reducer(
    pre_embs: torch.Tensor, target_spk_idx: int, merge_quantity: int, pre_clus_labels: torch.Tensor,
):
    """
    Reduce the number of embedding vectors by merging the closest embedding vectors.
        - This merging algorithm is based on the assumption that the closest embeddings
          are the most redundant embedding vectors.
        - The closest embedding vectors are chosen by selecting the highest top-N sum of
          each column in a given affinity matrix.
        - If merge_quantity is N, we choose (N+1) vectors into 1 embedding vector.
          Thus, we reduce N embeddings in the original embedding vector set.

    Example:
        >>> merge_quantity = 1 # We merge 1+1 = 2 embedding vectors
        >>> affinity_mat = [[1.0, 0.2, 0.8],
                            [0.2, 1.0, 0.4],
                            [0.8, 0.4, 1.0]]
        >>> affinity_mat.sum(0)
        [2.0, 1.6, 2.2]

        The first and the third embedding vectors are merged into one embedding vector.
        >>> index_mapping # (bypassed indices, merged indices)
        ([1], [0, 2])

    Args:
        pre_embs (Tensor):
            Potential Embedding vectors to be merged
        affinity_mat (Tensor):
            The affinity matrix of the `pre_embs`
        target_spk_idx (int):
            The targeted speaker index for merging
        merge_quantity (int):
            The count of embeddings to be reduced
        pre_clus_labels (list)
            The original cluster (speaker) index

    Returns:
        merged_embs (torch.Tensor):
            The merged embedding vectors.
        merged_clus_labels (torch.Tensor):
            The cluster (speaker) indices for the merged embedding vectors.
        index_mapping (Tuple[torch.Tensor, torch.Tensor]):
            A tuple containing the indices of the original embeddings that were not merged (`bypassed indices`)
            and the indices of the new merged embeddings (`merged indices`).
    """
    if pre_embs.shape[0] != pre_clus_labels.shape[0]:
        raise ValueError("Dimension mismatch between `pre_embs` and `pre_clus_labels`.")

    target_emb_index = torch.where(pre_clus_labels == target_spk_idx)[0]
    org_size = target_emb_index.shape[0]
    if merge_quantity > 0:
        if merge_quantity > (target_emb_index.shape[0] - 1):
            raise ValueError(
                f"merge_quantity {merge_quantity} should not be larger than target_emb_index length: {target_emb_index.shape[0]-1}"
            )
        total_affinity_mat = getCosAffinityMatrix(pre_embs)

        # Get the lower triangle of the affinity_mat array
        affinity_mat = total_affinity_mat[:, target_emb_index][target_emb_index, :]
        if affinity_mat.shape[0] != target_emb_index.shape[0]:
            raise ValueError(
                "Dimension mismatch between targeted speaker affinity `affinity_mat` and targeted speaker index `target_emb_index`."
            )
        # Get the indices of the closest embedding vectors
        selected_inds, rest_inds = get_closest_embeddings(affinity_mat, merge_quantity)
        spk_cluster_labels, selected_embs = pre_clus_labels[target_emb_index], pre_embs[target_emb_index]

        # Note that we need to return the indices of speaker-specific indices from `target_emb_index`.
        index_mapping = (target_emb_index[rest_inds.sort()[0]], target_emb_index[selected_inds])

        # Merge the embeddings targeted by the 2-dim indices `index_2d`
        merged_embs, merged_clus_labels = merge_vectors(selected_inds, selected_embs, spk_cluster_labels)

        if (org_size - merge_quantity) != merged_embs.shape[0]:
            raise ValueError(
                f"Reducer output {merged_embs.shape[0]} is not matched to the target quantity {org_size - merge_quantity}."
            )

    else:
        merged_embs = pre_embs[target_emb_index]
        merged_clus_labels = pre_clus_labels[target_emb_index]
        index_mapping = (target_emb_index, torch.arange(0))
    return merged_embs, merged_clus_labels, index_mapping