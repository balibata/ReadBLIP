# ReadBLIP

This is my own understanding of the BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation, which is a Vision-language pre-training model.



## Background

- Vision-language pre-training has gained significant traction due to its success in numerous multimodal downstream tasks.
- There are two primary challenges in existing methods:
  1. **Model Perspective**: Many methods employ either an encoder-based model or an encoder-decoder model. Encoder-based models are not easily adaptable to text generation tasks like image captioning, while encoder-decoder models haven't been successful for image-text retrieval tasks.
  2. **Data Perspective**: Most high-performing methods, such as CLIP, rely on large datasets with noisy image-text pairs from the web.