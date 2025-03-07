# Implementation of "SecLMM: Towards Secure and Lightweight Inference Service for Large Multimodal Models"

## Overview of the SecLMM Framework
![1740483322842](https://github.com/user-attachments/assets/d67e2c9a-9ada-423e-992e-f73f3824c5c1)
SecLMM has the following salient features:

* SecLMM enables the model owner to delegate LMM-based inference service to multiple secure proxies while fully protecting sensitive requests and model weights.
 
* SecLMM preserves inference accuracy comparable to native design. It optimizes unreasonably computational overhead and opens up the possibility of deploying secure LMM-based inference on resource-constrained servers.
 
* SecLMM can provide guaranteed security against untrusted service providers maliciously delivering incorrect inference results. 	


## Getting Started
### Installation
This section will guide you through setting up the experimental environment on your local machine and downloading the dataset. Please refer to the specific subsections below.

    git clone https://github.com/Andrew2001x/SecLMM.git
    cd SecLMM
    conda env create -n seclmm --file environmnet.yaml

### Dataset
We provide four used datasets, CIFAR-10, CIFAR-100, Tiny-ImageNet and Fairface.

    cd datasets
    sh dataset_u.sh


## Fine-tuning (10-way 5-shot fewshot learning)
Fine-tuning includes four tasks: CIFAR10, CIFAR100, TinyImageNet, and FairFace-Age/Gender/Race. Modify the
--task_name parameter to switch between them.

    cd finetuning
    #run SecLMM fine-tuning process, e.g. for cifar10 with gelu +softmax
    python finetune.py --task_name cifar10  --hidden_act gelu --softmax_act softmax
    #run SecLMM fine-tuning process, e.g. for cifar10 with gelu +approximation
    python finetune.py --task_name cifar10  --hidden_act gelu --softmax_act quant
    #run SecLMM fine-tuning process, e.g. for cifar10 with sig +softmax
    python finetune.py --task_name cifar10  --hidden_act sig --softmax_act softmax
    #run SecLMM fine-tuning process, e.g. for cifar10 with sig +approximation
    python finetune.py --task_name cifar10  --hidden_act sig --softmax_act quant


    
## Inference
Inference includes four tasks: CIFAR10, CIFAR100, TinyImageNet, and FairFace-Age/Gender/Race. Modify the
--task_name parameter to switch between them. 

    cd SecLMM/based_on_tinyclip/src/inference
    cd plaintext-acc

### zeroshot_p

    cd zeroshot
    #run SecLMM plaintext inference process, e.g. for cifar10 with gelu+softmax
    python zeroshot_p.py --task_name cifar10  --hidden_act gelu --softmax_act softmax
    #run SecLMM plaintext inference process, e.g. for cifar10 with gelu+approximation
    python zeroshot_p.py --task_name cifar10  --hidden_act gelu --softmax_act quant
    #run SecLMM plaintext inference process, e.g. for cifar10 with sig +softmax
    python zeroshot_p.py --task_name cifar10  --hidden_act sig --softmax_act softmax
    #run SecLMM plaintext inference process, e.g. for cifar10 with sig +approximation
    python zeroshot_p.py --task_name cifar10  --hidden_act sig --softmax_act quant


### fewshot_p

    cd fewshot
    #run SecLMM plaintext inference process, e.g. for cifar10 with gelu +softmax
    python fewshot_p.py --task_name cifar10  --hidden_act gelu --softmax_act softmax
    #run SecLMM plaintext inference process, e.g. for cifar10 with gelu+quant approximation
    python fewshot_p.py --task_name cifar10  --hidden_act gelu --softmax_act quant
    #run SecLMM plaintext inference process, e.g. for cifar10 with sig +softmax 
    python fewshot_p.py --task_name cifar10  --hidden_act sig --softmax_act softmax
    #run SecLMM plaintext inference process, e.g. for cifar10 with sig + approximation
    python fewshot_p.py --task_name cifar10  --hidden_act sig --softmax_act quant

## Benchmark
Benchmark includes four tasks: CIFAR10, CIFAR100, TinyImageNet, and FairFace-Age/Gender/Race. Modify the
--task_name parameter to switch between them.

    cd SecLMM/based_on_tinyclip/src/benchmark
    cd cipher-acc

### zeroshot_c

    cd zeroshot
    #run SecLMM cipher inference process, e.g. for cifar10 with gelu+softmax 
    python zeroshot_c.py --task_name cifar10  --hidden_act gelu --softmax_act softmax 
    #run SecLMM cipher inference process, e.g. for cifar10 with gelu + approximation
    python zeroshot_c.py --task_name cifar10  --hidden_act gelu --softmax_act quant
    #run SecLMMcipher inference process, e.g. for cifar10 with sig +softmax
    python zeroshot_c.py --task_name cifar10  --hidden_act sig --softmax_act softmax
    #run SecLMM cipher inference process, e.g. for cifar10 with sig +quant approximation
    python zeroshot_c.py --task_name cifar10  --hidden_act sig --softmax_act quant

### fewshot_c

    cd fewshot
    #run SecLMM cipher inference process, e.g. for cifar10 with gelu+softmax 
    python fewshot_c.py --task_name cifar10  --hidden_act gelu --softmax_act softmax 
    #run SecLMM cipher inference process, e.g. for cifar10 with gelu + approximation
    python fewshot_c.py --task_name cifar10  --hidden_act gelu --softmax_act softmax
    #run SecLMMcipher inference process, e.g. for cifar10 with sig +softmax
    python fewshot_c.py --task_name cifar10  --hidden_act sig --softmax_act softmax
    #run SecLMM cipher inference process, e.g. for cifar10 with sig +quant approximation
    python fewshot_c.py --task_name cifar10  --hidden_act sig --softmax_act quant
