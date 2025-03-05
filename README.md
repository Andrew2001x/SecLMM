# Implementation of "SecLMM: Towards Secure and Lightweight Inference Service for Large Multimodal Models"

## Overview of the SecLMM Framework
![1740483322842](https://github.com/user-attachments/assets/d67e2c9a-9ada-423e-992e-f73f3824c5c1)
SecLMM has the following salient features:

* SecLMM enables the model owner to delegate LMM-based inference service to multiple secure proxies while fully protecting sensitive requests and model weights.
 
* SecLMM preserves inference accuracy comparable to native design. It optimizes unreasonably computational overhead and opens up the possibility of deploying secure LMM-based inference on resource-constrained servers.
 
* SecLMM can provide guaranteed security against untrusted service providers maliciously delivering incorrect inference results. 	


##Getting Started
###Installation
This section will guide you through setting up the experimental environment on your local machine and downloading the dataset. Please refer to the specific subsections below.

    git clone https://github.com/Andrew2001x/SecLMM.git
    cd SecLMM
    conda env create -n seclmm --file environmnet.yaml

###Dataset
We provide four used datasets, CIFAR-10, CIFAR-100, Tiny-ImageNet and Fairface.

    cd datasets
    sh dataset_u.sh
