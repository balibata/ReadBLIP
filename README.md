# Read and Understanding

# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation



This is my own understanding of the **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**, which is a Vision-language pre-training model(VLP). And this is the repo for presentation of the course DS-5690 in Vanderbilt University.



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

![image-20231023222807253](E:\DS 5690\Overview.png)



## Model Architecture

- Visual transformer as image encoder, dividing input images into patches and encodes them as a sequence of embeddings
- New Multimodal mixture of encoder-decoder(MED), a multi-task model which can operate in one of the three functionalities:
  - **Unimodal encoder**: separately encodes image and text, appending a CLS token in the beginning of the text input to summarize the sentence.
  - **Image-grounded text encoder**: inject visual information by inserting one additional cross-attention (CA) layer between the self-attention layer and the feed forward network for each transformer block.
  - **Image-grounded text decoder**: replace the bidirectional self-attention layers in the image-grounded text encoder with causal self-attention layers.



## Pre-training Objectives

- **Image-Text Contrastive Loss** (ITC) activates the unimodal encoder. The aim is to align the feature space of the visual transformer and the text transformer by encouraging positive image-text pairs to have similar representations. This has proven to be an effective target for improving vision and language understanding.
- **Image-Text Matching Loss** (ITM) activates the image grounded text encoder. Its purpose is to learn image-text multimodal representation, capturing fine-grained alignment between vision and language. ITM is a binary classification task, given its multimodal characteristics, the model uses ITM headers (linear layers) to predict whether an image-text pair is positive (matching) or negative (mismatching).
- **Language Modeling Loss** (LM) activates the image grounded text decoder. The decoder is designed to generate a textual description of a given image. It optimizes the cross-entropy loss so that the model is trained to maximize the probability of text in an autoregressive manner. In calculating the loss, the authors used a label smoothing of 0.1. In contrast to MLM loss, which is widely used in VLP, LM gives the model the generalization ability to convert visual information into coherent captions.



For efficient pre-training while taking advantage of multitasking learning, the text encoder and text decoder share all parameters except the SA layer. The reason is that the difference between the encoding and decoding tasks is best captured by the SA layer. In particular, the encoder uses bidirectional self-attention to build a representation of the current input token, while the decoder uses causal self-attention to predict the next token. On the other hand, the embedding layer, CA layer, and FFN function similarly between encoding and decoding tasks, so sharing these layers can improve training efficiency while benefiting from multi-task learning.



## CapFilt

![img](E:\DS 5690\CapFilt.png)

The author proposes Captioning and Filtering (CapFilt), which is a new method to improve the quality of textual corpus. The figure above shows an illustration of CapFilt. It introduces two modules: a subtitler for generating subtitles for a given web image, and a filter for removing noisy image-text pairs. Both the subtitler and filter are initialized from the same pre-trained MED model and are individually fine-tuned on the COCO dataset. Fine-tuning is a lightweight process.

Specifically, a subtitler is an image-based text decoder. It is combined with the LM target to decode the text of a given image. Given a web image$I_{w}$, the subtitler generates a composite subtitle $T_s$. The filter is an image-based text encoder. It is combined with the ITC and ITM goals to understand if text matches images. The filter removes noisy text from the original web text $T_w$ and the resultant text ,$T_s$ which is considered noisy text if the ITM header predicts that the text does not match the image. Finally, the authors combined the filtered image-text pairs with human annotation pairs to form a new dataset and used this dataset to pre-train a new model.

![img](E:\DS 5690\CapFilt_eg.png)

## Experiments and Results

![image-20231024021553922](E:\DS 5690\VQA.png)

**Visual Question Answering (VQA):** 

VQA (Antol et al., 2015) requires this model to predict the answer to a given image and question. We do not define VQA as a multi-answer classification task (Chen et al., 2020; Li et al., 2020), we follow Li et al. (2021a) and treat it as an answer generation task, implementing open VQA. As shown in Figure 5(a), during the fine-tuning process, we rearrange the pre-trained model, where image questions are first encoded into multimodal embedding and then given an answer decoder. The VQA model takes the ground real answer as the goal and refines the LM loss. The results are shown in Table 8. Using 14 million images, BLIP outperformed ALBEF's + by 1.64% on the test set. Using 129M images, BLIP achieves better performance than SimVLM, which uses 13× pre-training data and a larger visual backbone with additional convolution stages.

## Future improvement

The authors found that there are several potential directions to further improve BLIP performance: 

- bootstrapping of multiple rounds of data sets; 

- Generate multiple synthetic captions for each image to further expand the pre-training corpus; 

- Simulate model integration by training multiple different subtitlers and filters and combining their power in CapFilt.

## Question 1

Given the significance of data quality in training models, how does BLIP's approach to bootstrapping captions from noisy web data compare to other methods that might clean or curate the data first? What are the advantages and potential pitfalls of BLIP's approach?

### Answer

BLIP's approach is proactive, as it generates synthetic captions and filters out noisy ones, allowing it to utilize vast amounts of web data while ensuring data quality. This method can be more scalable compared to manual curation. However, the quality of synthetic captions and the efficiency of the filtering mechanism are crucial. If the synthetic captions aren't of high quality or if the filtering fails to remove all noisy data, the model could still be trained on misleading information.



## Question 2

BLIP aims to unify vision-language understanding and generation tasks under a single framework. How does this unified approach impact the adaptability and performance of the model across diverse tasks? What challenges might arise from such a unified approach?

### Answer

A unified approach like BLIP's can offer several advantages, such as reduced need for task-specific architectures and potentially better generalization across tasks due to shared knowledge. However, the challenge lies in ensuring that the model doesn't become a "jack of all trades, master of none". Balancing performance across multiple tasks without sacrificing the proficiency in any particular task can be tricky.



## Demo Exhibition

Website: https://replicate.com/salesforce/blip?prediction=bswoddzbzybiwyizksnkohyzxq

![image-20231024001353865](C:\Users\14111\AppData\Roaming\Typora\typora-user-images\image-20231024001353865.png)

## Installation and run

- **Recommended way:**

Install [the Python client](https://github.com/replicate/replicate-python):

```shell
pip install replicate
```

Next, [copy your API token](https://replicate.com/account) and authenticate by setting it as an environment variable:

```shell
export REPLICATE_API_TOKEN=r8_RFU**********************************
```

(This is your `Default` [API token](https://replicate.com/account/api-tokens). Keep it to yourself.)

Then, run the model:

```python
import replicate
output = replicate.run(
    "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
    input={"image": open("path/to/file", "rb")}
)
print(output)
```

- Using Colab to run:

  https://colab.research.google.com/github/salesforce/BLIP/blob/main/demo.ipynb#scrollTo=2b949f9f

  

## Video introduction and analysis

https://www.youtube.com/watch?v=X2k7n4FuI7c

## Links

The following links are relevant paper and original github code, and they also provide with a simple example demo on hugging face:

- https://proceedings.mlr.press/v162/li22n/li22n.pdf
- https://github.com/salesforce/BLIP
- https://huggingface.co/spaces/Salesforce/BLIP

## Reference

- https://arxiv.org/abs/2201.12086
- Chen, Y., Li, L., Yu, L., Kholy, A. E., Ahmed, F., Gan, Z., Cheng, Y., and Liu, J. UNITER: universal image-text representation learning. In *ECCV*, volume 12375, pp. 104–120, 2020.
- Zhu, L. and Yang, Y. Actbert: Learning global-local video text representations. In *CVPR*, pp. 8746–8755, 2020.
