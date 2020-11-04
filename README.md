# FANet: A Feedback Attention Network for Improved Biomedical ImageSegmentation
Authors: [Nikhil Kumar Tomar](https://www.linkedin.com/in/nktomar/), [Debesh Jha](https://www.linkedin.com/in/debesh-jha-071462aa/), Michael A. Riegler, Håvard D. Johansen, Dag Johansen,  Pål Halvorsen and  Sharib Ali



## Abstract
Recently, deep learning based medical image analysis, particularly, biomedical image segmentation has attracted substantial attention in the computer vision community. Even though convolutional neural networks have shown progress, there still exists unleashed opportunities for improvement. While, current networks focus on a systematic one-way epoch wise training, predictions from the previous epoch remains unexplored. In this work, we leverage each epoch information to prune the prediction maps of the subsequent training epoch. In this context, we propose a new architecture called feedback attention network `FANet` that unifies the previous epoch mask with the feature map of the current epoch, similar to a recurrent learning mechanism, which is then used to provide a hard-attention to the learnt feature maps at different convolution layers. We show that our proposed \textit{feedback-attention} model provides a substantial improvement for most segmentation metrics on publicly available biomedical imaging datasets. 
