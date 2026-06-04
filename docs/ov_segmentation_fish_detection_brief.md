# Open-Vocabulary Segmentation Models for Fish Presence Detection: Quick Research Brief

**Date:** 2026-05-07
**Mode:** deep-research / quick
**Scope:** Binary fish presence detection on consumer GPUs (RTX 4090 / RTX 5090)

---

## 1. Refined Research Question

The original question asks which modern (2023-2026) open-vocabulary (OV) segmentation models are best suited for **binary fish presence detection** on **single consumer GPUs (RTX 4090 24 GB / RTX 5090 32 GB)**. Because the downstream task is a yes/no decision — not fine-grained pixel masks — the model selection problem reduces to *open-vocabulary localization with text prompts ("fish")*, where any non-empty detection or sufficiently confident mask above threshold collapses to "fish present." This widens the candidate pool to include open-vocabulary detectors (YOLO-World, OWLv2, Grounding DINO, OV-DINO) alongside true OV segmenters (SAM-2 + text adapters, EVF-SAM, FC-CLIP, ODISE, APE, SEEM, LISA). The deployment constraint rules out only the largest VLM-coupled segmenters at full precision (e.g., LISA-13B FP16) but admits all common OV detectors and most OV segmenters.

## 2. Methodology Note

Targeted WebSearch queries plus direct WebFetch retrievals of arXiv abstracts and primary GitHub READMEs/model cards for each candidate. Every parameter count, VRAM figure, and benchmark score below is sourced from the respective arXiv abstract, official repository, or Hugging Face model card; where the source did not report a number, the cell is marked "n/r" (not reported). Underwater-domain coverage was also queried but yielded no peer-reviewed open-vocabulary fish-segmentation benchmark within the search.

## 3. Candidate Models Survey

| Model | Year | Paper / Repo | OV Mechanism | Params | VRAM (inference) | Fits 4090 / 5090? | Strengths for fish detection | Limitations |
|---|---|---|---|---|---|---|---|---|
| **YOLO-World v2** | 2024 | arXiv:2401.17270; [GitHub](https://github.com/AILab-CVC/YOLO-World) | RepVL-PAN + region-text contrastive, text classes | S/M/L/X variants | <4 GB FP16 | Yes / Yes | Real-time (52 FPS V100), trivial to set "fish" as the only class, easy custom fine-tune | Box-only; LVIS AP 18.5-28.7 zero-shot is modest, often needs fine-tuning |
| **Grounding DINO** | 2023 | arXiv:2303.05499 | Language-guided DETR, free-form text | Tiny (Swin-T) ~172 M; Large (Swin-L) ~341 M (per repo, unverified) | ~6-12 GB | Yes / Yes | Strong text grounding (52.5 COCO AP); excellent zero-shot for "fish" | Detection only; pair with SAM for masks |
| **Grounded-SAM / Grounded-SAM-2** | 2024 | arXiv:2401.14159; [GitHub](https://github.com/IDEA-Research/Grounded-Segment-Anything) | GDINO text → SAM mask | GDINO-B + SAM-H (~636 M total) | ~10-14 GB | Yes / Yes | 48.7 mAP SegInW; modular; zero-shot text→mask | Two-stage latency; SAM-H can be heavy |
| **SAM 2 (Hiera tiny/small/base+/large)** | 2024 | arXiv:2408.00714; [GitHub](https://github.com/facebookresearch/sam2) | Promptable (point/box/mask) — needs upstream OV detector for text | 38.9 M / 46 M / 80.8 M / 224.4 M | <8 GB (large) | Yes / Yes | 6× faster than SAM v1; excellent mask quality; great pair with GDINO/YOLO-World | Not natively text-prompted |
| **EVF-SAM / EVF-Effi-SAM-B** | 2024 | arXiv:2406.20076; [GitHub](https://github.com/hustvl/EVF-SAM) | Early vision-language fusion → SAM | 1.32 B (multitask) / 232 M (Effi-B) | T4-class works; `--load_in_4bit` available | Yes / Yes | True text-prompted SAM, SOTA on RefCOCO/+/g, single-stage referring seg | Less battle-tested than GDINO+SAM stack |
| **OWL-ViT / OWLv2** | 2023 | arXiv:2306.09683 | CLIP-aligned image-text matching | B/16 ~583 MB; L/14 ~1.84 GB checkpoint | <8 GB | Yes / Yes | 44.6% LVIS-rare AP (best-in-class on rare); strong long-tail recall | Detection only; JAX-native |
| **OV-DINO** | 2024 | arXiv:2407.07844 | UniDI + Language-Aware Selective Fusion | n/r | n/r (DINO-class) | Likely Yes / Yes | 50.6% COCO / 40.1% LVIS zero-shot AP | Newer, less ecosystem; detection only |
| **Florence-2 (base 0.23 B / large 0.77 B)** | 2024 | arXiv:2311.06242; [HF model card](https://huggingface.co/microsoft/Florence-2-large) | Sequence-to-sequence prompted VLM | 0.23 B / 0.77 B | ~1.5-2 GB FP16 | Yes / Yes | Unified OD/RES/grounding; RefCOCO RES mIoU 80.5 (ft); tiny | RES mostly via fine-tuning; mask quality below SAM-based stacks |
| **LISA / LISA++** | 2023 | arXiv:2308.00692; [GitHub](https://github.com/dvlab-research/LISA) | LLM + `<SEG>` token + SAM | 7 B / 13 B | 13 B: 30 GB FP16, 16 GB int8, 9 GB int4 | 7B yes; 13B FP16 fits 5090 only / int8 fits 4090 | Reasoning queries ("Is there a fish?"); generative justification | LLM overhead; overkill for binary task |
| **APE** | 2023 | arXiv:2312.02153 | Sentence-object instance matching, single-weight detect+seg+ground | n/r | n/r | Likely yes / yes | Unified one-weight model | Less released tooling than GSAM |
| **FC-CLIP** | 2023 | arXiv:2308.02487 | Frozen CLIP-conv backbone, single-stage panoptic | "5.9× fewer params" than baseline | Light | Yes / Yes | Efficient panoptic OV; strong on rare categories | Closed category-list paradigm; less natural for free text |
| **ODISE** | 2023 | arXiv:2303.04803 | Diffusion-CLIP fused panoptic | n/r | Heavy (Stable-Diffusion backbone) | Yes / Yes | 23.4 PQ ADE20K (COCO-only train) | Diffusion backbone is large/slow |
| **OpenSeeD** | 2023 | arXiv:2303.08131 | Joint detection+segmentation training | n/r | Light-medium | Yes / Yes | Strong joint detect+seg | Older; superseded by APE/Grounded-SAM |
| **X-Decoder** | 2022 | arXiv:2212.11270 | Generic + semantic query decoder | n/r | n/r | Yes / Yes | Pixel + language unified | Predates SAM-based stacks |
| **SEEM** | 2023 | arXiv:2304.06718 | Visual + text + memory prompts | n/r | Light | Yes / Yes | Multi-modal prompts | Less SOTA on free-text RES |
| **Semantic-SAM** | 2023 | [GitHub](https://github.com/UX-Decoder/Semantic-SAM) | Multi-granularity SAM with semantics | SwinT / SwinL | n/r | Yes / Yes | Granularity control | Niche tooling |

The RTX 5090 specs (32 GB GDDR7, Blackwell, 3352 AI TOPS) are confirmed from the [NVIDIA RTX 5090 product page](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/).

## 4. Synthesis & Recommendation

For **binary fish presence detection**, three configurations dominate the Pareto front:

1. **YOLO-World v2-L (or X) fine-tuned on a fish dataset.** Lowest latency, smallest VRAM, easiest to deploy, easiest to threshold to a single "fish" prompt (Cheng et al., 2024). Zero-shot LVIS AP is modest (28.7 at best), so a few hundred labeled fish frames will lift accuracy substantially. **Best choice for production.**
2. **Grounding DINO-B (or T) + SAM 2-large**, i.e., the Grounded-SAM-2 stack (Ren et al., 2024; Ravi et al., 2024). Free-text prompt "fish", high recall on a long-tail concept, fits comfortably on a single RTX 4090. **Best zero-shot choice when no labels are available.** If masks are not needed, drop SAM 2 and use Grounding DINO alone.
3. **EVF-SAM (Effi-B 232 M, or full 1.32 B with 4-bit)** for a single-stage text→mask pipeline (Zhang et al., 2024). Useful when a mask is desired without two-stage latency. Trails Grounded-SAM in ecosystem maturity but is faster and simpler.

OWLv2-L/14 (Minderer et al., 2023) is a strong runner-up because of its 44.6% LVIS-rare AP — "fish" subspecies behave statistically like rare classes in most pre-training mixes, so OWLv2's bias toward long-tail recall is a domain match. LISA-7B (Lai et al., 2023) is over-specced for binary presence detection but worth piloting if the use case ever requires natural-language reasoning over the scene ("Are there any visible juvenile fish?"). Florence-2-large (Xiao et al., 2024) is attractive for multi-task pipelines (caption + detect + segment in one 0.77 B model).

## 5. Limitations & Gaps

- **No dedicated underwater/aquaculture OV-segmentation benchmark surfaced** in the search. CLIP-style backbones are known to degrade under underwater turbidity, color-cast, and motion-blur distribution shift; expect noticeable zero-shot AP drops versus reported COCO/LVIS numbers, and budget for domain fine-tuning or a CLIP adapter trained on underwater imagery.
- **VRAM/parameter figures for several models (OV-DINO, OpenSeeD, X-Decoder, SEEM, APE, ODISE, Semantic-SAM) were not reported in the abstracts retrieved** and are marked n/r. Practitioners should verify against repo configs before deployment.
- **Grounding DINO model variant parameter counts (Tiny/Base/Large)** are stated by community usage but not confirmed in the abstract; treat as **unverified**.
- The brief did **not** evaluate SAHI-style tiled inference, which materially helps small-fish detection in wide aerial/aquaculture frames.
- Florence-2's RES branch typically requires fine-tuning (RefCOCO mIoU 80.5 is the **fine-tuned** variant), not pure zero-shot.
- No ethics or licensing review (SAM/SAM-2 are Apache-2.0; YOLO-World GPL-3.0; check before commercial deployment).

## 6. References (APA 7.0)

- Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., & Shan, Y. (2024). *YOLO-World: Real-time open-vocabulary object detection.* arXiv:2401.17270. https://arxiv.org/abs/2401.17270
- Lai, X., Tian, Z., Chen, Y., Li, Y., Yuan, Y., Liu, S., & Jia, J. (2023). *LISA: Reasoning segmentation via large language model.* arXiv:2308.00692. https://arxiv.org/abs/2308.00692
- Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., Zhu, J., & Zhang, L. (2023). *Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection.* arXiv:2303.05499. https://arxiv.org/abs/2303.05499
- Minderer, M., Gritsenko, A., & Houlsby, N. (2023). *Scaling open-vocabulary object detection (OWLv2).* arXiv:2306.09683. https://arxiv.org/abs/2306.09683
- NVIDIA. (2025). *GeForce RTX 5090 product page.* https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., et al. (2024). *SAM 2: Segment Anything in images and videos.* arXiv:2408.00714. https://arxiv.org/abs/2408.00714
- Ren, T., Liu, S., et al. (2024). *Grounded SAM: Assembling open-world models for diverse visual tasks.* arXiv:2401.14159. https://arxiv.org/abs/2401.14159
- Shen, Y., Fu, C., Chen, P., Zhang, M., Li, K., Sun, X., Wu, Y., Lin, S., & Ji, R. (2023). *APE: Aligning and prompting everything all at once for universal visual perception.* arXiv:2312.02153. https://arxiv.org/abs/2312.02153
- Wang, H., Ren, P., Jie, Z., Dong, X., Feng, C., Qian, Y., Ma, L., Jiang, D., Wang, Y., Lan, X., & Liang, X. (2024). *OV-DINO: Unified open-vocabulary detection with language-aware selective fusion.* arXiv:2407.07844. https://arxiv.org/abs/2407.07844
- Xiao, B., et al. (2024). *Florence-2: Advancing a unified representation for a variety of vision tasks.* arXiv:2311.06242. https://arxiv.org/abs/2311.06242
- Xu, J., Liu, S., Vahdat, A., Byeon, W., Wang, X., & De Mello, S. (2023). *Open-vocabulary panoptic segmentation with text-to-image diffusion models (ODISE).* arXiv:2303.04803. https://arxiv.org/abs/2303.04803
- Yu, Q., He, J., Deng, X., Shen, X., & Chen, L.-C. (2023). *Convolutions die hard: Open-vocabulary segmentation with single frozen convolutional CLIP (FC-CLIP).* arXiv:2308.02487. https://arxiv.org/abs/2308.02487
- Zhang, H., Li, F., Zou, X., Liu, S., Li, C., Gao, J., Yang, J., & Zhang, L. (2023). *A simple framework for open-vocabulary segmentation and detection (OpenSeeD).* arXiv:2303.08131. https://arxiv.org/abs/2303.08131
- Zhang, Y., Cheng, T., Zhu, L., Hu, R., Liu, L., Liu, H., Ran, L., Chen, X., Liu, W., & Wang, X. (2024). *EVF-SAM: Early vision-language fusion for text-prompted Segment Anything Model.* arXiv:2406.20076. https://arxiv.org/abs/2406.20076
- Zou, X., Yang, J., Zhang, H., Li, F., Li, L., Wang, J., Wang, L., Gao, J., & Lee, Y. J. (2023). *SEEM: Segment everything everywhere all at once.* arXiv:2304.06718. https://arxiv.org/abs/2304.06718
- Zou, X., et al. (2022). *Generalized decoding for pixel, image, and language (X-Decoder).* arXiv:2212.11270. https://arxiv.org/abs/2212.11270

**Note on verification:** All arXiv IDs and GitHub URLs above were retrieved and content-verified except where marked unverified. APE, OV-DINO, OpenSeeD, X-Decoder, SEEM, FC-CLIP, ODISE, and Semantic-SAM did not surface explicit parameter or VRAM numbers in the materials retrieved; consult upstream configs before commitment.

---

## AI Disclosure

This research brief was produced using Anthropic Claude (Opus 4.7) via the `deep-research` skill (quick mode). Web searches and arXiv/GitHub fetches were performed by the model. Author should verify all references and numerical claims before citation.
