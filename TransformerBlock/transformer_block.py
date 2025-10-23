import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Attention.attention import MultiHeadAttention
from FeedForward.feedforward import FeedForward

# ============================================
# TRANSFORMER BLOCK
# ============================================

class TransformerBlock(nn.Module):
    """
    Un bloc Transformer complet pour GPT-2
    
    Architecture :
    1. LayerNorm → Multi-Head Attention → Residual
    2. LayerNorm → Feed-Forward → Residual
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim (int): Dimension des embeddings (768 pour GPT-2 small)
            num_heads (int): Nombre de têtes d'attention (12 pour GPT-2 small)
            dropout (float): Taux de dropout
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Layer Normalization (avant attention)
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Layer Normalization (avant FFN)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network
        self.ffn = FeedForward(embed_dim, dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [seq_len, seq_len] - Masque causal
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # 1. Attention block avec residual connection
        # Pre-LayerNorm (GPT-2 utilise pre-norm, pas post-norm)
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = residual + x  # Residual connection
        
        # 2. Feed-Forward block avec residual connection
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x  # Residual connection
        
        return x


def create_causal_mask(seq_len):
    """Crée un masque causal triangulaire"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


# ============================================
# TESTS
# ============================================

def test_transformer_block():
    """Test du Transformer Block complet"""
    print("\n" + "="*60)
    print("TEST 1: Transformer Block")
    print("="*60)
    
    # Paramètres GPT-2 small
    batch_size = 2
    seq_len = 10
    embed_dim = 768
    num_heads = 12
    
    # Créer le bloc
    block = TransformerBlock(embed_dim, num_heads)
    
    # Input aléatoire
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Masque causal
    mask = create_causal_mask(seq_len)
    
    print(f"✓ Input shape: {x.shape}")
    
    # Forward pass
    output = block(x, mask)
    
    print(f"✓ Output shape: {output.shape}")
    
    # Vérifier que les shapes correspondent
    assert output.shape == x.shape, "Les shapes ne correspondent pas!"
    print(f"✓ Shape correcte: {output.shape}")
    
    # Nombre de paramètres
    num_params = sum(p.numel() for p in block.parameters())
    print(f"\n✓ Nombre de paramètres: {num_params:,}")
    
    # Détails des paramètres
    attention_params = sum(p.numel() for p in block.attention.parameters())
    ffn_params = sum(p.numel() for p in block.ffn.parameters())
    ln_params = sum(p.numel() for p in block.ln1.parameters()) + sum(p.numel() for p in block.ln2.parameters())
    
    print(f"\n📊 Détails des paramètres:")
    print(f"  - Attention:   {attention_params:,} ({attention_params/num_params*100:.1f}%)")
    print(f"  - FFN:         {ffn_params:,} ({ffn_params/num_params*100:.1f}%)")
    print(f"  - LayerNorms:  {ln_params:,} ({ln_params/num_params*100:.1f}%)")
    print(f"  - Total:       {num_params:,}")


def test_residual_connections():
    """Vérifie que les residual connections fonctionnent"""
    print("\n" + "="*60)
    print("TEST 2: Residual Connections")
    print("="*60)
    
    batch_size = 1
    seq_len = 5
    embed_dim = 64
    num_heads = 4
    
    # Créer le bloc
    block = TransformerBlock(embed_dim, num_heads)
    
    # Input simple (identité)
    x = torch.ones(batch_size, seq_len, embed_dim)
    
    # Forward
    mask = create_causal_mask(seq_len)
    output = block(x, mask)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    # L'output devrait être différent de l'input (grâce aux transformations)
    # mais pas trop différent (grâce aux residual connections)
    diff = (output - x).abs().mean().item()
    print(f"\n✓ Différence moyenne input/output: {diff:.4f}")
    print(f"  (Devrait être > 0 mais pas énorme grâce aux residuals)")


def test_layer_norm():
    """Comprendre la Layer Normalization"""
    print("\n" + "="*60)
    print("TEST 3: Layer Normalization")
    print("="*60)
    
    # Créer des données avec des échelles différentes
    x = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])  # [1, 1, 4]
    
    print(f"✓ Input:")
    print(f"  Valeurs: {x.squeeze().tolist()}")
    print(f"  Mean: {x.mean().item():.2f}")
    print(f"  Std: {x.std().item():.2f}")
    
    # Appliquer LayerNorm
    ln = nn.LayerNorm(4)
    x_norm = ln(x)
    
    print(f"\n✓ Après LayerNorm:")
    print(f"  Valeurs: {[f'{v:.3f}' for v in x_norm.squeeze().tolist()]}")
    print(f"  Mean: {x_norm.mean().item():.6f}")
    print(f"  Std: {x_norm.std().item():.6f}")
    print(f"\n💡 La moyenne est ~0 et la variance est ~1 !")


def test_multiple_blocks():
    """Test avec plusieurs blocs empilés (comme dans GPT-2)"""
    print("\n" + "="*60)
    print("TEST 4: Empiler plusieurs blocs")
    print("="*60)
    
    batch_size = 2
    seq_len = 10
    embed_dim = 256
    num_heads = 8
    num_blocks = 3  # On teste avec 3 blocs au lieu de 12
    
    # Créer plusieurs blocs
    blocks = nn.ModuleList([
        TransformerBlock(embed_dim, num_heads)
        for _ in range(num_blocks)
    ])
    
    # Input
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = create_causal_mask(seq_len)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Nombre de blocs: {num_blocks}")
    
    # Passer à travers tous les blocs
    for i, block in enumerate(blocks):
        x = block(x, mask)
        print(f"  Après bloc {i+1}: {x.shape}")
    
    print(f"\n✓ Output final shape: {x.shape}")
    
    # Nombre total de paramètres
    total_params = sum(p.numel() for p in blocks.parameters())
    print(f"✓ Paramètres totaux ({num_blocks} blocs): {total_params:,}")


def test_pipeline_complet():
    """Test du pipeline complet: Embeddings → Transformer Blocks"""
    print("\n" + "="*60)
    print("TEST 5: Pipeline complet")
    print("="*60)
    
    # Simuler des embeddings (comme ceux de votre Embeddings Layer)
    batch_size = 1
    seq_len = 21  # "Bonjour, je teste mon GPT-2!"
    embed_dim = 768
    num_heads = 12
    
    # Embeddings (simulés)
    embeddings = torch.randn(batch_size, seq_len, embed_dim)
    print(f"✓ Embeddings shape: {embeddings.shape}")
    
    # Créer 1 bloc Transformer
    block = TransformerBlock(embed_dim, num_heads)
    
    # Masque causal
    mask = create_causal_mask(seq_len)
    
    # Forward
    output = block(embeddings, mask)
    
    print(f"✓ Après Transformer Block: {output.shape}")
    print(f"\n🎉 Pipeline Embeddings → Transformer Block fonctionne!")


if __name__ == "__main__":
    print("\n🚀 TESTS DU TRANSFORMER BLOCK\n")
    
    # Test 1: Bloc basique
    test_transformer_block()
    
    # Test 2: Residual connections
    test_residual_connections()
    
    # Test 3: Layer Normalization
    test_layer_norm()
    
    # Test 4: Plusieurs blocs
    test_multiple_blocks()
    
    # Test 5: Pipeline complet
    test_pipeline_complet()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("="*60)
    print("\n📁 Fichier refactorisé avec imports depuis Attention/ et FeedForward/")
    print("🎯 Prochaine étape: Refactoriser Model/gpt2_model.py")
    print("="*60 + "\n")
