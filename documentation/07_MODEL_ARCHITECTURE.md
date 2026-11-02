# 07 - Model Architecture: UrbanGraphFormer

## 1. Model Overview
`UrbanGraphFormer` is a hybrid spatiotemporal Graph Transformer designed for electrical power grid forecasting.

## 2. Mathematical Formulation
1. **Spatial Representation**: The electrical grid is modeled as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where vertices $\mathcal{V}$ represent buses and edges $\mathcal{E}$ represent transmission lines.
2. **Spatial Attention**:
   $$lpha_{ij} = rac{\exp\left(	ext{LeakyReLU}\left(\mathbf{a}^	op [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_j \,\|\, B_{ij}]ight)ight)}{\sum_{k \in \mathcal{N}_i} \exp\left(	ext{LeakyReLU}\left(\mathbf{a}^	op [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_k \,\|\, B_{ik}]ight)ight)}$$
   where $B_{ij}$ is the branch susceptance.
3. **Temporal Transformer**: Temporal dynamics are captured across a 24-hour sliding window using multi-head self-attention with rotary position embeddings (RoPE).

## 3. Checkpoint Details
- **Weights File**: `urbangraphformer_synth.pt` (307,701 bytes).
- **Execution Target**: Optimized for CPU inference latency (< 15ms per 24-hour horizon).
