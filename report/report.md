# Visual Place Recognition in a Campus Building: A Comparative Study of Classical and Deep-Learning Retrieval Pipelines

**Course:** Computer Vision — Final Project  
**Institution:** IE University  
**Date:** May 2026

---

## Abstract

This report presents an image-retrieval system for Visual Place Recognition (VPR) applied to the IE Tower campus building. Given a query photograph taken anywhere in the building, the system returns the top-K most visually similar gallery images and predicts the corresponding location. Three retrieval pipelines are implemented and evaluated under a shared experimental harness: (1) a classical pipeline based on SIFT local features aggregated with Vector of Locally Aggregated Descriptors (VLAD), (2) a deep-learning pipeline based on frozen DINOv2 ViT-S/14 global embeddings, and (3) a supervised CNN baseline — a small convolutional network trained on gallery labels whose penultimate embedding layer is reused for retrieval. All three methods produce L2-normalized vectors stored in a FAISS flat inner-product index and are evaluated using Top-1 accuracy, Top-5 accuracy, Precision at 5 (P@5), Mean Average Precision (mAP), and per-query index-search latency. The dataset comprises approximately 945 images across 27 indoor locations: 636 gallery images (the full training set) and 309 held-out test queries from a separate collection. On 309 test queries, the DINOv2 pipeline achieves 74.8% Top-1 accuracy and 77.4% mAP; the SIFT+VLAD baseline achieves 53.7% Top-1 accuracy and 58.3% mAP; and the CNN baseline achieves 29.1% Top-1 accuracy and 36.9% mAP. These results confirm that self-supervised Vision Transformer features substantially outperform both classical local-feature aggregation and a small supervised CNN for indoor place recognition at this scale.

---

## 1. Introduction

Visual Place Recognition is the problem of determining where a photograph was taken by matching it against a database of geo-tagged or location-tagged reference images. It is a foundational capability in robot navigation, augmented reality, and pedestrian wayfinding systems. While large-scale outdoor VPR has been studied extensively, indoor environments present unique challenges: repeated textures (corridors, walls, ceilings), variable lighting conditions, a high degree of structural symmetry, and a limited number of training examples per location.

This project constructs a VPR system for the IE Tower building in Madrid. A dataset of approximately 945 photographs was collected across 27 semantically meaningful indoor locations, including common areas, a robotics laboratory, hallways, stairwells, and the adjacent Caleido shopping center. Three methodologically distinct retrieval systems are developed and compared:

1. A **classical pipeline** using SIFT keypoint descriptors aggregated into a compact image-level representation via VLAD, followed by PCA whitening for dimensionality reduction.
2. A **deep-learning pipeline** using the frozen CLS-token embeddings of a DINOv2 Vision Transformer (ViT-S/14), a self-supervised model trained on a large-scale visual similarity objective.
3. A **CNN baseline** using a small convolutional network supervised on gallery location labels; after training, the embedding layer is extracted and used for retrieval in the same FAISS framework.

All three pipelines share the same data loading, index construction, evaluation, and demo components, ensuring a structurally fair comparison. The rest of this report is organized as follows. Section 2 reviews related work. Section 3 describes the dataset. Section 4 details the system architecture. Section 5 describes the feature extraction methodology for each pipeline. Section 6 covers the index and retrieval design. Section 7 defines the evaluation protocol and metrics. Section 8 presents results and analysis. Section 9 discusses limitations and future directions. Section 10 concludes.

---

## 2. Related Work

**Classical Feature-Based Retrieval.** Scale-Invariant Feature Transform (SIFT), introduced by Lowe (2004), detects and describes local image regions that are invariant to scale, rotation, and partial illumination changes. For place recognition, individual keypoint descriptors cannot directly serve as image-level representations and must be aggregated. Bag of Words (BoW) methods quantize each descriptor to a visual vocabulary word and represent an image as a histogram. VLAD, proposed by Jegou et al. (2010), improves on BoW by accumulating the signed residuals between descriptors and their assigned vocabulary centers, preserving richer geometric structure while remaining compact. VLAD representations benefit from intra-normalization (power-law normalization) and final L2 normalization, both of which are applied in this work.

**Deep Learning for Place Recognition.** Self-supervised Vision Transformers have become the state of the art for visual similarity. DINOv2 (Oquab et al., 2023) trains a ViT using self-distillation with no labels, producing features that are directly competitive with supervised models on dense prediction and retrieval tasks. The CLS token of DINOv2, which aggregates global image information through multi-head self-attention, has been shown to be particularly effective for image-level retrieval without any task-specific adaptation. With only approximately 636 gallery images spread across 27 classes, frozen pretrained features are the appropriate choice, as fine-tuning the backbone would risk overfitting on the limited intra-class variation.

**Approximate Nearest Neighbor Search.** Retrieval systems at scale rely on approximate nearest neighbor (ANN) indexes. FAISS (Johnson et al., 2019) provides highly optimized exact and approximate indexes. For a dataset of the size considered here, an exact flat inner-product index is used, as it provides deterministic results and the dataset is small enough that exhaustive search is negligible in latency.

---

## 3. Dataset

### 3.1 Collection

Photographs were collected at named indoor locations within the IE Tower building and the adjoining Caleido commercial complex. Each location corresponds to a semantically distinct area. Images were captured across multiple collection sessions by different team members using mobile phone cameras and stored in JPEG or JPEG-encoded formats. The combined dataset comprises approximately 945 images across 27 location classes.

The folder structure follows a flat convention: each subdirectory of `data/` is named after one location label and contains all training images belonging to that location. Held-out test images follow the same naming convention under a separate `test/` directory. The data preparation script performs light normalization before use: it renames folders with non-standard characters and removes any zero-byte placeholder files.

### 3.2 Splits

The dataset is organized into two physically separate directories rather than a random split of a single pool. The `data/` directory contains the full training set; every image in it is designated as a gallery image. The `test/` directory contains a separately collected set of query images that are never used during fitting or index construction.

This design ensures that gallery images serve a dual role — fitting data-dependent components (such as the VLAD codebook or CNN weights) and forming the indexed database — while test images serve exclusively as evaluation queries. There is no overlap between the two sets.

The split yields **636 gallery images** and **309 test query images** across 27 location classes.

### 3.3 Class Distribution

Table 1 summarizes the aggregate dataset statistics.

**Table 1. Dataset summary.**

| Metric | Value |
|---|---|
| Total location classes | 27 |
| Gallery images (`data/`) | 636 |
| Test query images (`test/`) | 309 |
| Total images | ~945 |
| Average gallery images per class | ~23.6 |
| Average test queries per class | ~11.4 |

The 27 location classes cover a mix of IE Tower building areas (16th floor offices, 5th floor hallway and stairs, bathroom, elevator, gym, robotics lab, entrance balloon installation) and the adjoining Caleido commercial complex (Starbacks, Ocine cinema, Makan, La Desayuneria, Five Guys, New York Burger, Honest Green, Kanbun, Lassal, Aromas, Ecoalf, Bareto, Rosselimac, Santagloria, Masqmenos, ball sculpture, colorful statue, and the Do Eat restaurants on the 4th floor and exterior). Class sizes vary across the dataset; Caleido Starbacks and Caleido Ocine are among the most image-rich classes owing to the variety of viewpoints available at those locations.

---

## 4. System Architecture

The system is organized as a modular library under `src/`, with CLI entry-point scripts under `scripts/` and a Streamlit interactive demo at `src/app.py`. The data flow is as follows:

```
[data/]                 [scripts/prepare_data.py]
  Training photos         Normalize folder names
     |                    Remove 0-byte stubs
     v                    Write data/manifest.csv
  data/manifest.csv       (all images -> gallery split)
                                |
          +-----------------+---+-------------------+
          |                 |                       |
  [Classical track]  [Deep track]          [CNN track]
  SIFT keypoints     DINOv2 ViT-S/14       TinyPlaceCNN
  VLAD aggregation   CLS token (384-d)     trained on gallery
  PCA whitening      L2 normalize          labels (supervised)
  (128-d)                |                 embedding (128-d)
          |              |                       |
          v              v                       v
    L2-norm vector  L2-norm vector          L2-norm vector
          |              |                       |
          +------+--------+-----------+----------+
                 |                    |
     [scripts/build_index.py]    [results/<method>.faiss]
     [src/index.py: FAISS IndexFlatIP]
                          |
            [src/retrieve.py: query pipeline]
                          |
           +--------------+---------------+
           |                              |
  [scripts/run_eval.py]        [src/app.py: Streamlit UI]
  Top-1, Top-5, P@5, mAP       Interactive upload demo
  Index-search latency
  (queries from test/)
```

The shared `Embedder` protocol (`src/features/base.py`) defines the interface that all three feature tracks implement. Any class exposing `fit()`, `embed()`, and `embed_batch()` methods can be plugged into the index and evaluation pipeline without modification, making the comparison structurally fair. The CNN embedder additionally exposes `save()` and `load()` methods so that the trained model weights are persisted to `results/cnn.pt` and reloaded at evaluation time without retraining.

---

## 5. Feature Extraction

### 5.1 Classical Pipeline: SIFT + VLAD

**Keypoint Detection and Description.** The Scale-Invariant Feature Transform is applied to the grayscale version of each image using OpenCV's `SIFT_create`. Up to 500 keypoints are detected per image. Each keypoint is described by a 128-dimensional gradient orientation histogram that is invariant to scale, rotation, and moderate perspective distortion. If no keypoints are detected, a single zero descriptor is substituted to maintain a valid embedding path.

**Codebook Learning.** A visual vocabulary of K = 64 cluster centers is learned from all SIFT descriptors pooled across the gallery split, using Mini-Batch K-Means (batch size 4096, three initializations, random seed 42). The resulting cluster centers form the VLAD codebook.

**VLAD Aggregation.** For each image, every SIFT descriptor is assigned to its nearest codebook center by Euclidean distance. The VLAD vector is constructed as the concatenation of per-cluster residual sums: for cluster c, the contribution is the sum of (descriptor - center\_c) over all descriptors assigned to cluster c. This yields a raw vector of dimension K x 128 = 8192. Intra-normalization is applied by passing each 128-dimensional cluster block through a signed square root (power-law normalization), reducing the influence of bursty visual words. The full vector is then L2-normalized.

**PCA Whitening.** A PCA step is fitted on the gallery VLAD vectors and reduces the representation to min(128, N\_gallery) dimensions with whitening, decorrelating dimensions and equalizing their variance. With 636 gallery images exceeding the 128-component target, PCA produces the full 128 principal components. After projection, the vector is L2-normalized again. The effective embedding dimension is therefore **128**.

**Persistence.** The fitted codebook centers and PCA model are serialized together to `results/classical.pkl` immediately after fitting. At evaluation time, the `.pkl` is reloaded rather than re-fitting from scratch, guaranteeing that the same codebook is used for both gallery embedding and query embedding — eliminating any re-initialization randomness between the two phases.

### 5.2 Deep Pipeline: DINOv2 ViT-S/14

**Model.** DINOv2 ViT-S/14 is a Vision Transformer with patch size 14, trained using self-supervised DINO distillation on a large curated dataset (LVD-142M). It is loaded from the `timm` library (`vit_small_patch14_dinov2.lvd142m`) with pretrained weights, set to evaluation mode, and fully frozen. No fine-tuning is performed, which is appropriate given the small dataset size.

**Preprocessing.** Each image is resized so that its shorter side equals 518 pixels (bilinear interpolation), then center-cropped to 518 x 518. Pixel values are normalized using ImageNet channel statistics (mean [0.485, 0.456, 0.406], standard deviation [0.229, 0.224, 0.225]).

**Embedding.** The preprocessed image is passed through the ViT backbone under `torch.inference_mode()`. The CLS token from the final transformer layer is used as the global image descriptor, producing a 384-dimensional vector that is L2-normalized via `F.normalize`. The CLS token aggregates global context through multi-head self-attention over all patch tokens. The effective embedding dimension is **384**.

**Device Selection.** The implementation detects CUDA availability and falls back to CPU. On Windows without CUDA, inference runs on CPU. On macOS, Metal Performance Shaders (MPS) is deliberately avoided due to a known segmentation fault when co-loading MPS-backed PyTorch and FAISS in the same process.

### 5.3 CNN Baseline: TinyPlaceCNN

**Motivation.** The CNN baseline occupies an intermediate position between the classical and deep tracks: unlike SIFT+VLAD, it learns features directly from pixel data; unlike frozen DINOv2, its feature space is shaped specifically by the gallery label supervision. It also allows evaluation of whether a small network trained from scratch on a limited dataset can compete with a large pretrained model.

**Architecture.** `TinyPlaceCNN` is a lightweight convolutional network comprising four convolutional blocks followed by a projection head and a classification head. Each block consists of a Conv2d layer, Batch Normalization, ReLU activation, and MaxPool2d downsampling. The channel progression is 3 → 32 → 64 → 128 → 256. An `AdaptiveAvgPool2d(1, 1)` collapses the spatial dimensions after the final block. A linear projection layer maps the 256-dimensional pooled features to a 128-dimensional embedding, and a separate linear classification head maps the embedding (after ReLU) to per-class logits. Only the embedding vector — the output of the projection layer — is used during retrieval; the classification head is discarded after training.

**Training.** The model is trained for 12 epochs using the Adam optimizer (learning rate 10⁻³) with cross-entropy loss on all 636 gallery images treated as labeled training examples. Input images are resized to 160 × 160 pixels. Data augmentation during training includes random resized cropping (scale 0.7–1.0), random horizontal flip, and color jitter (brightness, contrast, saturation ±0.2; hue ±0.05), reducing overfitting on the small dataset. At inference, images are resized to 184 pixels on the short side and center-cropped to 160 × 160. Pixel values are normalized with ImageNet channel statistics for both training and inference. The trained weights are serialized to `results/cnn.pt` and reloaded at evaluation time, avoiding retraining costs.

**Embedding.** After training, the embedding vector produced by the projection layer is L2-normalized via `F.normalize` before indexing. The effective embedding dimension is **128**.

**Device Selection.** Device selection follows the same CUDA / CPU fallback logic as the deep pipeline.

---

## 6. Indexing and Retrieval

### 6.1 FAISS Index

Gallery embeddings are stored in a FAISS `IndexFlatIP` (flat inner-product index). Because all vectors are L2-normalized, inner product is equivalent to cosine similarity. The flat index performs exact exhaustive search, which is appropriate for a gallery of 636 images and eliminates the recall loss of approximate quantization methods.

The index is constructed from the gallery embedding matrix, a list of location labels, and a list of image paths. It is serialized to a `.faiss` binary file alongside a `.json` sidecar containing the dimension, labels, and paths. At query time, the index is loaded once and held in memory.

### 6.2 Query Pipeline

Given a query image, the retrieval pipeline executes as follows:

1. Load and preprocess the image using the shared `load_image` utility (RGB conversion, optional long-side bounding at 1024 px).
2. Compute the embedding using the appropriate `Embedder` instance.
3. Call `RetrievalIndex.search(vec, k)`, which executes a single FAISS `search` call and returns the top-K `Hit` objects, each carrying a similarity score, location label, and image path.

Batch embedding (default batch size 16) is used during gallery index construction for efficiency. Single-image embedding is used at query time to reflect the real interactive latency.

### 6.3 Demo Application

The Streamlit application provides an interactive web interface. The user selects a method (deep, classical, or CNN), sets K via a slider, and uploads a query image. The application immediately displays the top-K gallery matches as thumbnails with their location labels and cosine similarity scores. The embedder is cached with `@st.cache_resource` to avoid reloading the model between queries.

---

## 7. Evaluation Protocol

### 7.1 Split Integrity

All evaluation is performed on the held-out query split. Gallery images are never used as queries, and query images are never present in the gallery. This ensures that scores reflect genuine retrieval ability rather than near-duplicate matching.

### 7.2 Metrics

Let Q be the number of query images, H\_i = [h\_{i,1}, ..., h\_{i,K}] the ordered list of K retrieved hits for query i, and t\_i the true location label of query i.

**Top-K Accuracy.** A query is correctly answered at rank K if t\_i appears among the top-K retrieved labels.

$$\text{Top-K Acc} = \frac{1}{Q} \sum_{i=1}^{Q} \mathbf{1}\!\left[t_i \in \{l(h_{i,j})\}_{j=1}^{K}\right]$$

**Precision at K (P@K).** The fraction of the top-K retrieved hits whose label matches the query label, averaged over all queries.

$$\text{P@K} = \frac{1}{Q} \sum_{i=1}^{Q} \frac{|\{j \leq K : l(h_{i,j}) = t_i\}|}{K}$$

**Mean Average Precision (mAP).** Average Precision for a single query is the mean of the precision values at each rank where a correct hit appears, normalized by the number of relevant items in the returned list. mAP is the mean of AP across all queries and rewards systems that place correct results early in the ranking.

$$\text{AP}_i = \frac{1}{R_i} \sum_{k=1}^{K} P_i(k) \cdot \mathbf{1}[l(h_{i,k}) = t_i], \quad \text{mAP} = \frac{1}{Q}\sum_{i=1}^{Q}\text{AP}_i$$

**Query Latency.** Each index search is timed individually (after a warm-up pass) using `time.perf_counter`. Mean and 95th-percentile latency in milliseconds are reported for the FAISS search step alone.

### 7.3 CNN Evaluation Protocol

The CNN baseline is evaluated identically to the other two methods: the saved model weights (`results/cnn.pt`) are loaded, the embedding layer is applied to all test query images from the `test/` directory, and the resulting vectors are searched against the pre-built FAISS index. No retraining is performed at evaluation time. This ensures that the CNN is assessed as a retrieval system rather than as a classifier, making the comparison with the other two methods fair.

---

## 8. Results

### 8.1 Dataset Summary

| Metric | Value |
|---|---|
| Total location classes | 27 |
| Gallery images (`data/`) | 636 |
| Test query images (`test/`) | 309 |
| Total images | ~945 |

### 8.2 Retrieval Results

Table 2 compares all three retrieval methods on the 309 held-out test queries (K = 5). The latency columns reflect FAISS index-search time only; embedding time is separate and is dominated by the model forward pass or CNN inference.

**Table 2. Retrieval evaluation results (K = 5, n = 309 queries).**

| Method | Emb. Dim | Top-1 | Top-5 | P@5 | mAP | Mean Search (ms) | P95 Search (ms) |
|---|---|---|---|---|---|---|---|
| CNN baseline | 128 | 29.1% | 55.7% | 25.4% | 36.9% | 0.019 | 0.025 |
| SIFT + VLAD (classical) | 128 | 53.7% | 70.6% | 44.5% | 58.3% | 0.018 | 0.025 |
| DINOv2 ViT-S/14 (deep) | 384 | **74.8%** | **85.8%** | **62.1%** | **77.4%** | 0.035 | 0.041 |

### 8.3 Analysis

**Ranking of methods.** DINOv2 is the clear winner across all metrics (Top-1 74.8%, mAP 77.4%), followed by SIFT+VLAD (Top-1 53.7%, mAP 58.3%), with the supervised CNN placing last (Top-1 29.1%, mAP 36.9%). The ordering of classical above CNN is counterintuitive given that the CNN receives explicit location-label supervision, and is discussed below.

**Why DINOv2 leads.** The DINOv2 ViT-S/14 processes the entire image through multi-head self-attention, allowing it to integrate global spatial context, object-level features, signage, and scene geometry into a single 384-dimensional descriptor. This representation encodes location-specific semantic content — the distinctive decor of a restaurant, the equipment layout of a robotics laboratory, the architectural character of an atrium — that is highly discriminative even across visually similar spaces. Crucially, these features were learned from a large-scale curated dataset (LVD-142M) and generalize to the IE Tower domain without any fine-tuning.

**Why the classical method outperforms the CNN.** The CNN baseline achieves only 29.1% Top-1 accuracy despite receiving explicit location-label supervision during training. This counterintuitive result is explained by the dataset scale and the nature of the retrieval task. With 636 gallery images split across 27 classes (~23.6 per class on average), training a convolutional network from scratch is highly prone to overfitting — the model memorizes per-instance appearance rather than learning a generalizable embedding geometry. The SIFT+VLAD pipeline, by contrast, uses hand-crafted features that do not require any learning of visual primitives; SIFT descriptors are already invariant to scale, rotation, and moderate illumination changes, and the VLAD aggregation benefits from PCA whitening that decorrelates the embedding dimensions. Importantly, the classical pipeline now persists its fitted codebook and PCA model to disk, ensuring the same parameters are applied at both indexing and query time and eliminating any re-initialization variance. For retrieval specifically, where the embedding geometry must generalize to unseen test views, this regularized classical approach proves more robust than a small supervised network trained from scratch.

**Gap between DINOv2 and classical.** DINOv2 outperforms SIFT+VLAD by 21.1 percentage points in Top-1 accuracy (74.8% vs. 53.7%) and by 19.1 points in mAP (77.4% vs. 58.3%). This is a substantial but not overwhelming gap, reflecting the fact that a well-implemented classical pipeline with proper model persistence can capture meaningful location information in an indoor setting.

**P@5 interpretation.** P@5 for the deep method (62.1%) is lower than Top-5 accuracy (85.8%). This is expected: Top-5 accuracy asks whether the correct class appears anywhere in the top five results, while P@5 asks what fraction of all five results are correct. With 27 classes and a finite gallery, only a subset of the top-5 results will typically share the query's class even for a well-functioning system. The classical method shows a similar spread (P@5 44.5% vs. Top-5 70.6%).

**Index-search latency.** All three methods achieve sub-millisecond FAISS search latency (0.018–0.035 ms mean), which is negligible for any interactive application. Total query latency is dominated by the embedding step: a DINOv2 forward pass on CPU takes on the order of 1–2 seconds per image, while the CNN inference is substantially faster. GPU acceleration would reduce DINOv2 latency to tens of milliseconds and make real-time deployment practical.

---

## 9. Discussion

### 9.1 Limitations

**Statistical reliability.** With 309 test queries, each percentage point of Top-1 accuracy now corresponds to approximately 3.09 queries, making the reported figures substantially more reliable than a 37-query evaluation. Nonetheless, class-level analysis remains noisy for the smallest location classes.

**Class imbalance.** The dataset spans 27 classes of varying size. Aggregate metrics are weighted by class frequency and therefore reflect performance on more image-rich locations (e.g., Caleido Starbacks, Caleido Ocine) more heavily than on less-represented ones. Per-class breakdown is not reported here but would be informative for identifying which locations are hardest to recognize.

**CNN overfitting.** The CNN baseline underperforms even the classical pipeline, which is attributable to overfitting on a small supervised training set (~23.6 images per class). The model architecture and hyperparameters (12 epochs, lr = 10⁻³, no weight decay) were chosen without systematic hyperparameter search, and a more carefully regularized training procedure could potentially improve CNN performance.

**Intra-class viewpoint variation.** Even with more images per class than the prior dataset, some locations may have gallery images concentrated from a narrow range of viewpoints. Queries taken from substantially different angles or under different lighting remain challenging for all methods.

**No spatial re-ranking.** All three pipelines perform global descriptor matching without spatial verification. For the classical method, adding a re-ranking stage based on geometrically consistent SIFT matches (RANSAC homography estimation) between the query and the top-K candidates could improve precision on ambiguous retrievals.

**Embedding latency on CPU.** DINOv2 inference on CPU (approximately 1–2 seconds per image at 518 × 518 resolution) is too slow for real-time use. GPU inference would reduce this to tens of milliseconds.

### 9.2 Design Decisions

**Frozen DINOv2 features.** Fine-tuning a ViT on 636 gallery images would risk overfitting given the limited intra-class variation. The frozen pretrained representation is the standard practice for small-dataset retrieval and the results confirm it is already highly discriminative without adaptation.

**All-gallery training split.** Using every training image as a gallery image (rather than holding out a validation fraction) maximizes the number of reference vectors per class available at retrieval time and ensures that data-dependent components (VLAD codebook, PCA, CNN weights) are fitted on the maximum available signal.

**Shared Embedder protocol with persistence.** The `Embedder` abstract protocol enforces a common interface across all three tracks. All three embedders now implement `save()` and `load()` methods: the deep model uses PyTorch checkpoint serialization, the CNN saves model weights to `.pt`, and the classical pipeline serializes the codebook and PCA model to `.pkl`. This ensures the same fitted parameters are used for gallery indexing and query embedding, eliminating any re-initialization randomness between the two phases.

**Exact FAISS index.** `IndexFlatIP` performs exact exhaustive search. For 636 gallery vectors this is optimal; approximate indexes would introduce recall loss with no latency benefit.

**CNN as retrieval, not classifier.** Despite being trained with a classification objective, the CNN is evaluated exclusively as a retrieval system (embedding similarity search), which is methodologically consistent with the other two methods and reflects the realistic deployment scenario.

### 9.3 Future Directions

**CNN regularization and pretraining.** Rather than training from scratch, initializing the CNN with ImageNet-pretrained weights (transfer learning) and applying stronger regularization (weight decay, dropout, early stopping) would likely improve the CNN baseline substantially.

**Data augmentation for query simulation.** Generating synthetic queries via random crops, brightness perturbation, or perspective distortion would expand the test pool and expose retrieval systems to realistic viewing condition variation without requiring additional data collection.

**Geometric re-ranking for the classical pipeline.** A post-retrieval RANSAC verification stage using the SIFT keypoints of query and top-K candidates could substantially improve the classical method's precision.

**Larger DINOv2 variants.** DINOv2 is available in ViT-B/14 (768-dim) and ViT-L/14 (1024-dim) configurations. With a larger gallery, these larger models may yield further improvements.

**Late fusion.** A weighted combination of SIFT+VLAD and DINOv2 similarity scores could leverage complementary local and global information, potentially outperforming either method alone.

---

## 10. Conclusion

This project implements and evaluates three end-to-end Visual Place Recognition pipelines for an indoor campus environment. The classical pipeline uses SIFT keypoint descriptors aggregated via VLAD with PCA whitening, yielding 128-dimensional embeddings. The CNN baseline uses a small supervised convolutional network trained on gallery labels, whose 128-dimensional penultimate embedding is reused for retrieval. The deep pipeline uses the frozen CLS token of a DINOv2 ViT-S/14, yielding 384-dimensional embeddings. All three pipelines share a FAISS-based retrieval index, a unified evaluation harness, and a Streamlit demo interface.

Evaluated on 309 held-out test queries across 27 indoor locations, DINOv2 achieves 74.8% Top-1 accuracy and 77.4% mAP, the SIFT+VLAD classical pipeline achieves 53.7% Top-1 accuracy and 58.3% mAP, and the CNN baseline achieves 29.1% Top-1 accuracy and 36.9% mAP. Two findings stand out. First, frozen self-supervised Vision Transformer features substantially outperform both alternatives, confirming that large-scale pretraining transfers effectively to indoor place recognition without any fine-tuning. Second, the small supervised CNN underperforms the classical pipeline despite receiving explicit label supervision, illustrating that for retrieval tasks at limited data scale, generalizable feature geometry matters more than task-specific training. The classical pipeline's strong performance is also attributable to proper model persistence — saving the fitted codebook and PCA to disk ensures the same parameters are applied at indexing and query time. Index-search latency is sub-millisecond for all three methods, making the system suitable for interactive deployment given sufficient embedding throughput.

---

## References

Jegou, H., Douze, M., Schmid, C., and Perez, P. (2010). Aggregating local descriptors into a compact image representation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3304-3311.

Johnson, J., Douze, M., and Jegou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.

Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91-110.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. (2023). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*.
