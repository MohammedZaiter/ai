# Presentation outline (about 12-15 slides, export to Presentation.pdf)

Required by the spec: problem formulation, comparative results tables, learning curves, key findings.
All figures are saved automatically in `results/figures/`.

1. **Title**: project title, group number, names
2. **Problem formulation**: CIFAR-10 classification (10 classes, 32x32 RGB), objectives (architecture, hyperparameters, augmentation, transfer learning)
3. **Dataset & pipeline**: 45k / 5k / 10k split, normalization, DataLoaders, sample images
4. **Baseline CNN**: architecture table (layers + output shapes), params, receptive field
5. **Baseline results**: `task1_curves.png` + `task1_confusion_matrix.png`, test accuracy, most confused classes
6. **Task 2 method**: one-factor-at-a-time, list of experiments
7. **Architecture results**: depth / kernel / stride-padding / BN plots + part of the benchmark table (params, receptive field, peak val acc, gap)
8. **Optimization results**: optimizer / lr / scheduler plots
9. **Augmentation results**: augmentation plot, effect on the generalization gap
10. **Benchmark table**: `task2_benchmark.csv` (main rows) + best model vs baseline
11. **Transfer learning setups**: scratch / frozen / finetune_layer4 / finetune_all, trainable parameters
12. **Convergence speed**: `task3_convergence_100pct.png` + epochs to 70% / 80%
13. **Data efficiency**: `task3_data_efficiency.png` (10% vs 100%)
14. **Final comparison**: `task3_comparison.csv` (test acc, macro F1, time/epoch, latency)
15. **Key findings & conclusion**: 4-5 bullet points backed by numbers
