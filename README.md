# Read and Understanding

# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation



This is my own understanding of the **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**, which is a Vision-language pre-training model(VLP). And this is the repo for presentation of the course DS-5690 in Vanderbilt University.

## Links

The following links are relevant paper and original github code, and they also provide with a simple example demo on hugging face:

- https://proceedings.mlr.press/v162/li22n/li22n.pdf
- https://github.com/salesforce/BLIP
- https://huggingface.co/spaces/Salesforce/BLIP

## Background

- Vision-language pre-training has gained significant traction due to its success in numerous multimodal downstream tasks.
- There are two primary challenges in existing methods:
  1. **Model Perspective**: Many methods employ either an encoder-based model or an encoder-decoder model. Encoder-based models are not easily adaptable to text generation tasks like image captioning, while encoder-decoder models haven't been successful for image-text retrieval tasks.
  2. **Data Perspective**: Most high-performing methods, such as CLIP, ALBEF and SimVLM, rely on large datasets with noisy image-text pairs from the web. We can obtain better performance on the efficiency of the model by extending the dataset to be trained. Based on the current research results, however, web text with noise is considered as suboptimal source of supervision for visual language learning.

## Related Work

- Vision-language Pre-training: aims to improve performance of downstream vision and language tasks by pretraining the model on large-scale image-text pairs.
- Knowledge Distillation: aims to improve the performance of a student model by distilling knowledge from a teacher model, which has been shown to be effective for image classification and VLP.
- Data Augmentation

## Overview

![image-20231023222807253](C:\Users\14111\AppData\Roaming\Typora\typora-user-images\image-20231023222807253.png)

## Model Architecture





## Demo Exhibition

Website: https://replicate.com/salesforce/blip?prediction=bswoddzbzybiwyizksnkohyzxq

![image-20231024000859307](C:\Users\14111\AppData\Roaming\Typora\typora-user-images\image-20231024000859307.png)

![image-20231024001353865](C:\Users\14111\AppData\Roaming\Typora\typora-user-images\image-20231024001353865.png)

## Video introduction and analysis

https://www.youtube.com/watch?v=X2k7n4FuI7c
