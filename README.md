# Introduction
Federated learning (FL) enables edge devices to collaboratively learn a model in a distributed fashion. Many existing researches have focused on improving communication efficiency of high-dimensional models and addressing bias caused by local updates. However, most FL algorithms are either based on reliable communications or assuming fixed and known unreliability characteristics. In practice, networks could suffer from dynamic channel conditions and non-deterministic disruptions, with time-varying and unknown characteristics. To this end, in this paper we propose a sparsity-enabled FL framework with both improved communication efficiency and bias reduction, termed as SAFARI. It makes use of similarity among client models to rectify and compensate for bias that results from unreliable communications. More precisely, sparse learning is implemented on local clients to mitigate communication overhead, while to cope with unreliable communications, a similarity-based compensation method is proposed to provide surrogates for missing model updates. With respect to sparse models, we analyze SAFARI under bounded dissimilarity. It is demonstrated that SAFARI under unreliable communications is guaranteed to converge at the same rate as the standard FedAvg with perfect communications. Implementations and evaluations on the CIFAR-10 dataset validate the effectiveness of SAFARI by showing that it can achieve the same convergence speed and accuracy as FedAvg with perfect communications, with up to 60% of the model weights being pruned and a high percentage of client updates missing in each round of model updates.


## How to Run



## Citation

If this code is useful in your research, you are encouraged to cite our academic paper:
```
@article{mao2023safari,
  title={SAFARI: Sparsity-Enabled Federated Learning with Limited and Unreliable Communications},
  author={Mao, Yuzhu and Zhao, Zihao and Yang, Meilin and Liang, Le and Liu, Yang and Ding, Wenbo and Lan, Tian and Zhang, Xiao-Ping},
  journal={IEEE Transactions on Mobile Computing},
  year={2023},
  publisher={IEEE}
}
```
