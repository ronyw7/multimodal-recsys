import torch
def get_negative_samples(data, num_users, num_items):
    """
    Generate negative samples for user-business interactions
    Args:
        data: PyG Graph data object containing positive edges
        num_users: Total number of users
        num_items: Total number of items
    Returns:
        tuple: (negative edge indices, negative edge labels)
    """
    # Get positive interactions
    pos_users, pos_items = data.edge_label_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize interaction matrix
    interactions = torch.zeros(
        (num_users, num_items), 
        dtype=torch.bool,
        device=device
    )
    
    # Mark positive interactions
    business_indices = pos_items - num_users  # Adjust business indices
    interactions[pos_users, business_indices] = True
    
    # Find all possible negative interactions
    available_negatives = torch.where(~interactions.reshape(-1))[0]
    
    # Sample random negative interactions
    num_samples = pos_users.size(0)
    sampled_indices = available_negatives[
        torch.randint(
            0, 
            available_negatives.size(0), 
            size=(num_samples,), 
            device=device
        )
    ]
    
    # Convert linear indices to user-business pairs
    sampled_users = sampled_indices // num_items
    sampled_items = (sampled_indices % num_items) + num_users
    
    # Create negative edge tensor
    neg_edges = torch.stack((sampled_users, sampled_items), dim=0)
    neg_labels = torch.zeros(neg_edges.shape[1])
    print(neg_edges.device)
    return neg_edges, neg_labels


