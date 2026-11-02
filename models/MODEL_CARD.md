# UrbanGraphFormer Model Card

## Model Details
- **Model Name**: UrbanGraphFormer Synthetic (UGF-Synth)
- **Model Version**: 0.2.0rc3
- **Architecture**: Spatiotemporal Graph Neural Network combining Graph Attention Layers (GAT) and Multi-Head Temporal Self-Attention.
- **Checkpoint File**: `urbangraphformer_synth.pt` (307,701 bytes)
- **Framework**: PyTorch 2.2+

## Intended Use
Predicts 24-hour ahead nodal electrical power demand across synthetic municipal power grid topologies.

## Training Data & Provenance
Pretrained on synthetic load profiles generated using diurnal Gaussian mixture curves, weather covariates (temperature, solar irradiation), and topology-conditioned edge features.
**Synthetic notice**: Trained exclusively on synthetic mathematical models; not calibrated on proprietary utility SCADA feeds.
