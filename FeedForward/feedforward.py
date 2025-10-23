import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) pour GPT-2
    
    Architecture simple :
    - Linear: embed_dim → 4 × embed_dim (expansion)
    - GELU activation
    - Linear: 4 × embed_dim → embed_dim (compression)
    """
    def __init__(self, embed_dim, dropout=0.1):
        """
        Args:
            embed_dim (int): Dimension des embeddings (768 pour GPT-2 small)
            dropout (float): Taux de dropout (0.1 par défaut)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = 4 * embed_dim  # 768 × 4 = 3072
        
        # Première couche linéaire (expansion)
        self.fc1 = nn.Linear(embed_dim, self.hidden_dim)
        
        # Deuxième couche linéaire (compression)
        self.fc2 = nn.Linear(self.hidden_dim, embed_dim)
        
        # Dropout pour la régularisation
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # 1. Expansion : 768 → 3072
        x = self.fc1(x)
        
        # 2. Activation GELU
        x = F.gelu(x)
        
        # 3. Dropout
        x = self.dropout(x)
        
        # 4. Compression : 3072 → 768
        x = self.fc2(x)
        
        # 5. Dropout final
        x = self.dropout(x)
        
        return x


# ============================================
# TESTS
# ============================================

def test_feedforward():
    """Test du Feed-Forward Network"""
    print("\n" + "="*60)
    print("TEST 1: Feed-Forward Network")
    print("="*60)
    
    # Paramètres
    batch_size = 2
    seq_len = 10
    embed_dim = 768
    
    # Créer le module
    ffn = FeedForward(embed_dim)
    
    # Input aléatoire (simule la sortie de l'attention)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"✓ Input shape: {x.shape}")
    
    # Forward pass
    output = ffn(x)
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Hidden dimension: {ffn.hidden_dim}")
    
    # Vérifier que les shapes sont correctes
    assert output.shape == x.shape, "Les shapes ne correspondent pas!"
    print(f"✓ Shape correcte: {output.shape}")
    
    # Nombre de paramètres
    num_params = sum(p.numel() for p in ffn.parameters())
    print(f"\n✓ Nombre de paramètres: {num_params:,}")
    
    # Détails des paramètres
    fc1_params = ffn.fc1.weight.numel() + ffn.fc1.bias.numel()
    fc2_params = ffn.fc2.weight.numel() + ffn.fc2.bias.numel()
    
    print(f"\n📊 Détails des paramètres:")
    print(f"  - fc1 (768 → 3072): {fc1_params:,}")
    print(f"  - fc2 (3072 → 768): {fc2_params:,}")
    print(f"  - Total: {num_params:,}")


def test_with_small_dims():
    """Test avec de petites dimensions pour mieux comprendre"""
    print("\n" + "="*60)
    print("TEST 2: FFN avec petites dimensions")
    print("="*60)
    
    # Petites dimensions pour visualiser
    batch_size = 1
    seq_len = 3
    embed_dim = 8
    
    # Créer le module
    ffn = FeedForward(embed_dim, dropout=0.0)  # Pas de dropout pour ce test
    
    # Input simple
    x = torch.ones(batch_size, seq_len, embed_dim)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Hidden dim: {ffn.hidden_dim} (= {embed_dim} × 4)")
    
    # Forward
    output = ffn(x)
    
    print(f"✓ Output shape: {output.shape}")
    print(f"\n🔍 Flux des dimensions:")
    print(f"  Input:  {x.shape} → [{batch_size}, {seq_len}, {embed_dim}]")
    print(f"  fc1:    [{batch_size}, {seq_len}, {embed_dim}] → [{batch_size}, {seq_len}, {ffn.hidden_dim}]")
    print(f"  GELU:   [{batch_size}, {seq_len}, {ffn.hidden_dim}] → [{batch_size}, {seq_len}, {ffn.hidden_dim}]")
    print(f"  fc2:    [{batch_size}, {seq_len}, {ffn.hidden_dim}] → [{batch_size}, {seq_len}, {embed_dim}]")
    print(f"  Output: {output.shape}")


def test_pipeline_complete():
    """Test du pipeline complet: Attention → FFN"""
    print("\n" + "="*60)
    print("TEST 3: Pipeline Attention → FFN")
    print("="*60)
    
    # Simuler la sortie de l'attention
    batch_size = 2
    seq_len = 21  # Comme votre exemple "Bonjour, je teste mon GPT-2!"
    embed_dim = 768
    
    # Output de l'attention (simulé)
    attention_output = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"✓ Attention output: {attention_output.shape}")
    
    # Passer dans le FFN
    ffn = FeedForward(embed_dim)
    ffn_output = ffn(attention_output)
    
    print(f"✓ FFN output: {ffn_output.shape}")
    print(f"\n🎉 Pipeline Attention → FFN fonctionne!")


def compare_parameters():
    """Compare les paramètres Attention vs FFN"""
    print("\n" + "="*60)
    print("COMPARAISON: Attention vs FFN")
    print("="*60)
    
    embed_dim = 768
    
    # Attention (on simule, vous avez déjà le code)
    attention_params = 2_362_368  # Calculé dans attention.py
    
    # FFN
    ffn = FeedForward(embed_dim)
    ffn_params = sum(p.numel() for p in ffn.parameters())
    
    print(f"\n📊 Nombre de paramètres par composant:")
    print(f"  - Multi-Head Attention: {attention_params:,}")
    print(f"  - Feed-Forward Network: {ffn_params:,}")
    print(f"  - Total (1 bloc):       {attention_params + ffn_params:,}")
    
    print(f"\n🔍 Répartition:")
    total = attention_params + ffn_params
    print(f"  - Attention: {attention_params/total*100:.1f}%")
    print(f"  - FFN:       {ffn_params/total*100:.1f}%")
    
    print(f"\n💡 Le FFN contient ~{ffn_params/attention_params:.1f}× plus de paramètres que l'Attention!")


def visualize_gelu():
    """Visualise la fonction GELU vs ReLU"""
    print("\n" + "="*60)
    print("VISUALISATION: GELU vs ReLU")
    print("="*60)
    
    # Créer des valeurs de test
    x = torch.linspace(-3, 3, 13)
    
    # GELU
    gelu_output = F.gelu(x)
    
    # ReLU pour comparaison
    relu_output = F.relu(x)
    
    print("\n📊 Comparaison GELU vs ReLU:\n")
    print("  x     |  GELU  |  ReLU")
    print("--------|--------|-------")
    
    for i in range(len(x)):
        print(f" {x[i]:6.2f} | {gelu_output[i]:6.3f} | {relu_output[i]:6.3f}")
    
    print("\n💡 Observation:")
    print("  - ReLU coupe brutalement à 0 pour x < 0")
    print("  - GELU a une transition douce (valeurs négatives petites mais non nulles)")
    print("  - GELU est préféré dans les Transformers modernes")


if __name__ == "__main__":
    print("\n🚀 TESTS DU FEED-FORWARD NETWORK\n")
    
    # Test 1: FFN basique
    test_feedforward()
    
    # Test 2: Petites dimensions
    test_with_small_dims()
    
    # Test 3: Pipeline complet
    test_pipeline_complete()
    
    # Comparaison avec Attention
    compare_parameters()
    
    # Visualisation GELU
    visualize_gelu()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("="*60)
    print("\n📁 Sauvegardez ce fichier dans: FeedForward/feedforward.py")
    print("🎯 Prochaine étape: Transformer Block (Semaine 5)")
    print("    → Combiner Attention + FFN + LayerNorm + Residual")
    print("="*60 + "\n")