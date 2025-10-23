import torch
import torch.nn as nn

class GPT2Embeddings(nn.Module):
    """
    Embeddings Layer pour GPT-2
    Combine Token Embeddings + Position Embeddings
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        """
        Args:
            vocab_size (int): Taille du vocabulaire (300 pour votre tokenizer)
            embed_dim (int): Dimension des embeddings (768 pour GPT-2 small)
            max_seq_len (int): Longueur maximale de séquence (1024 pour GPT-2)
        """
        super().__init__()
        
        # Token Embeddings: convertit chaque token ID en vecteur
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Position Embeddings: ajoute l'info de position
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch_size, seq_len] - IDs des tokens
        Returns:
            embeddings: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # 2. Position embeddings (0, 1, 2, ..., seq_len-1)
        positions = torch.arange(0, seq_len, device=input_ids.device)
        position_embeds = self.position_embeddings(positions)
        
        # 3. Addition des deux
        embeddings = token_embeds + position_embeds
        
        return embeddings


# TEST
if __name__ == "__main__":
    # Paramètres de test
    vocab_size = 300
    embed_dim = 64
    max_seq_len = 128
    
    # Créer le module
    embeddings = GPT2Embeddings(vocab_size, embed_dim, max_seq_len)
    
    # Tester avec des tokens aléatoires
    input_ids = torch.randint(0, vocab_size, (2, 10))  # batch=2, seq_len=10
    output = embeddings(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Ça marche!")