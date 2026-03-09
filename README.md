**[Abstract] delta-LLaVA: Base-then-Specialize Alignment for Token-Efficient Vision-Language Models**
<img width="1280" height="853" alt="gradcam_overlay_bilinear_144_2 (1)" src="https://github.com/user-attachments/assets/5d217aa5-4e00-4327-9584-4d37018588fe" />

Multimodal Large Language Models (MLLMs) combine visual and textual representations to enable rich reasoning capabilities. However, the high computational cost of processing dense visual tokens remains a major bottleneck. A critical component in this pipeline is the visual projector, which bridges the vision encoder and the language model. Standard designs often employ a simple multi-layer perceptron for direct token mapping, but this approach scales poorly with high-resolution inputs, introducing significant redundancy. We present Delta-LLaVA, a token-efficient projector that employs a low-rank DeltaProjection to align multi-level vision features into a compact subspace before further interaction. On top of this base alignment, lightweight Transformer blocks act as specialization layers, capturing both global and local structure under constrained token budgets. Extensive experiments and ablations demonstrate that this base-then-specialize design yields consistent gains across multiple benchmarks with only $144$ tokens, highlighting the importance of token formation prior to scaling interaction capacity. With Delta-LLaVA, inference throughput improves by up to 
55%, while end-to-end training accelerates by nearly 4-5x in pretraining and over 1.5x in finetuning, highlighting the dual benefits of our design in both efficiency and scalability.


This work has been heavily built on top of [TokenPacker](https://github.com/CircleRadon/TokenPacker). Thanks for their great work. 

