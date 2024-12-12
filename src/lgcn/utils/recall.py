import torch

def recall_at_k(data, model, num_users, k=500):
    """
    Calculate recall@k for recommendations.
    Returns average recall score across all users.
    
    Args:
        data: Graph data object containing edge indices and labels
        model: Neural network model with get_embedding method
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings for users and items
        embeddings = model.get_embedding(data.edge_index)
        user_embeds, item_embeds = embeddings[:num_users], embeddings[num_users:]
        
        # Calculate similarities and initialize truth matrix
        similarities = torch.matmul(user_embeds, item_embeds.t())
        truth = torch.zeros_like(similarities, dtype=torch.bool)
        
        # Get training and supervision edge masks
        train_edges = data.edge_index[:, data.edge_index[0] < num_users]
        sup_edges = data.edge_label_index[:, data.edge_label_index[0] < num_users]
        
        # Mask out training edges from recommendations
        similarities[train_edges[0], train_edges[1] - num_users] = float('-inf')
        
        # Mark ground truth edges
        truth[sup_edges[0], sup_edges[1] - num_users] = True
        
        # Calculate recall
        topk_scores, topk_items = torch.topk(similarities, k, dim=1)
        hits = truth.gather(1, topk_items).sum(dim=1)
        
        # Calculate total relevant items per user
        relevants = torch.bincount(sup_edges[0], minlength=num_users)
        
        # Compute recall, handling users with no relevant items
        recalls = torch.where(
            relevants > 0,
            hits.float() / relevants.float(),
            torch.ones_like(relevants, dtype=torch.float)
        )
        
        return recalls.mean().item()