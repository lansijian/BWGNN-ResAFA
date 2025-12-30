# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025.12.30

### Added
- ✅ Complete BWGNN model implementation (homogeneous and heterogeneous versions)
- ✅ BWGNN-ResAFA innovation: Residual Adaptive Frequency Attention
  - Homogeneous ResAFA with residual attention, LayerNorm, and weighted concatenation
  - Heterogeneous ResAFA with relation-level attention and max pooling
- ✅ Ablation study implementation
  - `BWGNN_AFA_NoResidual`: Remove residual structure
  - `BWGNN_AFA_NoLayerNorm`: Remove LayerNorm
  - `BWGNN_AFA_Sum`: Use sum instead of concatenation
  - Heterogeneous variants of all ablation models
- ✅ Baseline models: GCN, GAT, GraphSAGE for comparison
- ✅ Comprehensive experimental results on Amazon, YelpChi, and T-Finance datasets
- ✅ Training metrics visualization and analysis tools
- ✅ Complete documentation:
  - Main README with quick start guide
  - Advanced experiment report with ablation analysis
  - Experiment log with detailed iteration records
  - Ablation study documentation
  - Command reference guide

### Results
- ✅ **Amazon**: F1-Macro 92.56%, AUC 98.16% (exceeds paper: 91.94%, 93.95%)
- ✅ **YelpChi**: F1-Macro 76.16%, AUC 89.41% (meets paper: 75.68%, 89.67%)
- ✅ **T-Finance**: F1-Macro 88.75%, AUC 95.96% (matches paper: 88.99%, 95.99%)
- ✅ **ResAFA (Amazon)**: Recall 85.15% (+3.36%), F1-Macro 93.14% (+0.58%)
- ✅ **ResAFA (YelpChi)**: F1-Macro 76.68%, AUC 90.43% (best overall performance)

### Ablation Study Results
- ✅ Validated necessity of residual attention structure
- ✅ Validated importance of LayerNorm for training stability
- ✅ Validated superiority of weighted concatenation over sum aggregation
- ✅ Complete ResAFA outperforms all ablation variants on all metrics

### Fixed
- Fixed heterogeneous graph processing logic (parallel vs sequential)
- Fixed model saving based on validation Macro-F1 (paper implementation)
- Fixed data split consistency across all datasets
- Fixed visualization compatibility issues (matplotlib/Pillow)

### Documentation
- Added comprehensive README with installation and usage instructions
- Added advanced experiment report with ablation analysis
- Added experiment log with detailed iteration process
- Added ablation study documentation
- Added command reference guide
- Added .gitignore for GitHub repository

## [Unreleased]

### Planned
- Support for T-Social dataset (requires high-performance GPU)
- Additional ablation variants (e.g., different attention mechanisms)
- Model distillation for deployment
- Distributed training support for large graphs

