# Defense notes

Every team member must be able to explain **all** of the code and answer these questions.

## 1. Before the defense (checklist)

- [ ] All notebooks: set `TRAIN = False`, then **Kernel → Restart & Run All** without errors
- [ ] `Checkpoints/` contains a `.pth` for every run (task1, t2_*, t3_*)
- [ ] `results/` contains the `.json` histories (the notebooks load them when `TRAIN = False`)
- [ ] `04_demo_inference.ipynb` works with any checkpoint name (try one CNN and one ResNet)
- [ ] Test on the laptop used for the defense (CPU is fine for loading + inference)
- [ ] Notebooks saved **with outputs** in the zip

## 2. Where to make live modifications

| Asked change | Where |
|---|---|
| Learning rate / epochs / batch size | settings cell at the top of each notebook (`learning_rate`, `num_epochs`, `batch_size`) |
| Activation function | `Net.forward` in `utils.py` (`F.relu` → `F.leaky_relu`, `torch.tanh`, `F.gelu`) or `activation="gelu"` in `CustomCNN` |
| Add / remove a layer | `Net.__init__` + `Net.forward` (update the `fc1` input size!) or `conv_channels` / `fc_sizes` of `CustomCNN` |
| Kernel size / padding / stride | `nn.Conv2d(in, out, kernel_size, stride, padding)` or `CustomCNN` arguments |
| Transforms / augmentation | `get_transforms` in `utils.py` (add e.g. `transforms.RandomVerticalFlip()` or `transforms.RandomRotation(30)`) |
| Loss function | `criterion = nn.CrossEntropyLoss(label_smoothing=0.1)`, or class weights `nn.CrossEntropyLoss(weight=...)` |
| Optimizer | `optim.SGD(net.parameters(), lr=0.01, momentum=0.9)` instead of `optim.Adam(...)` |
| Freeze / unfreeze layers | `get_resnet18` in `utils.py` (`for param in model.layer3.parameters(): param.requires_grad = True`) |

Quick test after a change: set `num_epochs = 1` and run the training cell (no need to wait 20 epochs).

**If you change the architecture, the old checkpoint cannot be loaded anymore** (shapes do not match): retrain for 1-2 epochs with a new checkpoint name.

## 3. Theory questions

### Data
- **Why normalize?** Inputs centered at 0 with similar scales → better conditioned gradients, faster and more stable training.
- **Why ImageNet mean/std for ResNet?** The pre-trained filters expect the same input distribution as during their training.
- **Why a validation set and a test set?** Val is used to choose the best epoch/settings, so it is no longer "unseen". The test set is used only once at the end for an unbiased estimate.
- **`shuffle=True` only for train?** Shuffling breaks the order correlations between batches during training; it is useless for evaluation.
- **`num_workers=0`?** Data is loaded in the main process (avoids multiprocessing crashes on Windows); >0 loads the next batches in parallel.

### CNN
- **Conv output size:** `h_out = (h_in + 2p - k) / s + 1`. Max pool 2x2 → divides by 2.
- **Parameters of a conv layer:** `out_channels × (in_channels × k × k) + out_channels (bias)`. E.g. conv1 of Net: 32 × (3×3×3) + 32 = 896.
- **Receptive field:** input region that one output value depends on. `rf += (k-1) × jump`, `jump *= stride`. Grows with depth, kernel size and downsampling.
- **Why several 3x3 instead of one 7x7?** Two 3x3 = receptive field 5x5, three 3x3 = 7x7, but fewer parameters (3×9 = 27 vs 49 per channel pair) and more non-linearities.
- **Padding "same" vs "valid":** same keeps the size (information at the borders is kept); valid shrinks the feature map by k-1.
- **Stride-2 conv vs max pool:** both halve the size; the strided conv *learns* how to downsample (more parameters), max pool keeps the strongest activation (no parameters).
- **Why ReLU?** Non-linear, cheap, no vanishing gradient for positive values. Without an activation, the whole network is one linear function.
- **Why no softmax at the end of the model?** `nn.CrossEntropyLoss` applies log-softmax internally (numerically more stable).

### Training
- **Training step:** `zero_grad()` (gradients accumulate otherwise) → forward → loss → `backward()` (computes gradients with autograd) → `step()` (updates weights).
- **`model.train()` vs `model.eval()`:** changes the behaviour of Dropout (off in eval) and BatchNorm (uses running statistics in eval).
- **`torch.no_grad()`:** no computation graph → less memory and faster during evaluation.
- **Cross-entropy:** `-log(p_correct_class)`. Loss of a random 10-class model ≈ ln(10) ≈ 2.30.
- **Overfitting:** train loss goes down while val loss goes up; big generalization gap. Solutions: augmentation, dropout, weight decay, BatchNorm, early stopping (we keep the best val checkpoint).

### BatchNorm & Dropout
- **BatchNorm:** normalizes each channel over the batch (mean 0, std 1) then applies a learned scale γ and shift β. Stabilizes training, allows higher learning rates, slight regularization.
- **Why `bias=False`-like behaviour with BN?** BN subtracts the mean, so the conv bias is cancelled (we kept the bias for simplicity; it is harmless).
- **Dropout:** randomly sets activations to 0 with probability p during training → the network cannot rely on single neurons. Off at test time.

### Optimizers & learning rate
- **SGD + momentum:** `v = μ v + g`, `w = w - lr × v`. Momentum accumulates past gradients → faster in consistent directions, less oscillation.
- **Adam:** per-parameter adaptive learning rate using running mean of gradients (1st moment) and of squared gradients (2nd moment). Works well with little tuning.
- **Adam vs AdamW:** Adam adds weight decay to the gradient (it gets rescaled by the adaptive term → weaker regularization). AdamW applies the decay directly to the weights ("decoupled"), which regularizes correctly.
- **Learning rate too high:** loss oscillates / diverges. **Too low:** very slow convergence.
- **Schedulers:** StepLR divides lr by 10 at fixed epochs; Cosine decreases it smoothly to ~0. Big lr at the start to learn fast, small lr at the end to converge precisely.

### Augmentation
- **Why?** Creates new variations of the training images without changing the label → the model learns invariances (position, flip, color) → less overfitting.
- **Why no vertical flip on CIFAR-10?** Upside-down cars/animals don't exist in the test set; the augmentation must keep the image realistic.
- **Why no augmentation on val/test?** We want to measure the performance on real, unmodified images.
- **MixUp:** blends two images and their labels with λ ~ Beta(α, α). Strong regularizer, smoother decision boundaries. Train accuracy looks low because the images are mixed.
- **CutMix (alternative):** pastes a patch of one image into another, labels mixed by the patch area.

### Transfer learning
- **Why does it work?** Early layers learn generic features (edges, colors, textures) that are useful for any image dataset; later layers are more task-specific.
- **Feature extraction (frozen):** `requires_grad = False` on the backbone, only the new head is trained. Fast, few trainable parameters, good with little data.
- **Fine-tuning:** unfreeze upper blocks (layer4) or everything, with a **smaller lr** (1e-4) so the pre-trained weights are only adjusted, not destroyed.
- **Why is the new head trainable?** It is created after the freezing loop, so `requires_grad = True` by default.
- **Why keep frozen BatchNorm in eval mode?** Otherwise `model.train()` still updates the running mean/var of the frozen layers with CIFAR statistics.
- **Why only give trainable params to the optimizer?** Frozen params have no gradient; it is cleaner and saves memory.
- **Why resize to 128?** ResNet was trained on 224x224; with 32x32 inputs the feature maps become 1x1 very early and the pre-trained filters do not match the object scale.

### Evaluation
- **Normalized confusion matrix:** each row divided by the number of images of that true class → diagonal = per-class accuracy (recall).
- **Macro F1:** average F1 over classes (all classes count the same).
- **Generalization gap:** train acc - val acc at the best epoch.
- **Latency:** inference time for one image (with `torch.cuda.synchronize()` on GPU because CUDA calls are asynchronous).
- **Checkpoint:** a dictionary with `model_state_dict` (weights of every layer) + the info to rebuild the model. `load_state_dict` needs the exact same architecture.
