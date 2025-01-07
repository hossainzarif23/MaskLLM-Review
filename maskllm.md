# MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models
**Authors: Zarif Hossain, Anamul Hoque Emtiaj, Rakibul Hasan Rafi**
## 1. Overview
The research addresses the **computational inefficiency** of LLMs caused by their massive parameter counts. LLMs have hundreds of billions of parameters, thus makes it challenging to deploy LLMs in real-world application. To solve this problem, research are working on **sparsifying** LLMs. **Sparsity** aims to identify a subset of parameters that attain a comparable quality to the dense model. Network Pruning has been a go through approach for LLM sparsification. This approach compresses pre-trained language models via the removal of redundant parameters. The paper focuses on semi-structured pruning ($N:M$ sparsity) to reduce computational and memory overhead of LLMs while maintaining the model accuracy.
## 2. Pruning Large Language Models
Existing methods can be classified into three categories: Structured Pruning, Unstructured Pruning and Semi-Structured Pruning.
### 2.1 Structured Pruning
- Removes entire components such as neurons, filters, or attention heads, thus facilitates accelaration on GPUs.
- Performance degradation(accuracy) due to removing components(coarse grained removal).
### 2.2 Unstructured Pruning
- Sparsification via zeroing out parameters in LLMs.
- Performance degradation is minimal.
- Irregular sparsity patterns, thus difficult to optimize computations using GPUs
### 2.3 Semi-structured Pruning
- $N:M$ sparsity, which means having no more than $N$ non-zero values within a block of $M$ consecutive parameters. For example, let's consider $2:4$ sparsity. For a $512 \times 512$ linear layer, the weights would be divided into $\frac{512 \times 512}{4} = 65536$ consecutive parameter blocks. In each of the block, only at most $2$ parameters can have non-zero values.
- Computations optimizable using GPUs as sparsity follows a predictable pattern.
## 3. Related Works on Semi-structured Pruning
- **SparseGPT** and **Wanda** utilizes a small calibration datasets to determine $N:M$ sparsity masks, which may fail to represent the diverse knowledge embedded in LLMs pre-trained on massive datasets.
- These approaches generally use heuristic indicators like gradient information or weight magnitude, which may not accurately reflect the impact of pruning on the model’s performance. Thus, current techniques often fail to generalize pruning strategies across different tasks or domains due to their reliance on static importance criteria.
## 4. Methods
**The goal is to learn the optimal mask freezing out all the LLM parameters.**
### 4.1 Mask Selecttion and Non differentiability of Mask Selection
$N:M$ sparsity can be formulated as a mask selection problem with candidate set of $S$ where, $\lvert S \rvert = \binom{M}{N}$. For $2:4$ sparsity, the binary mask M must contain exactly two zeros, resulting in a discrete candidate set $S^{2:4}$. Let's denote $\mathcal{W} \in \mathbb{R}^{1 \times 4}$ as a block of four consecutive parameters and $\mathcal{M}_i^* \in \mathbb{B}^{1 \times 4}$ as the optimal binary mask, indicating which weights should be pruned **(indicated by 0)**.
```math
\mathcal{S}^{2:4} = \left\{ \mathcal{M} \in \mathbb{B}^{1 \times 4} \mid \sum \mathcal{M} = 2 \right\} = \left\{ \hat{\mathcal{M}}_1, \hat{\mathcal{M}}_2, \hat{\mathcal{M}}_3, \hat{\mathcal{M}}_4, \hat{\mathcal{M}}_5, \hat{\mathcal{M}}_6 \right\} 
= \left\{
\begin{bmatrix}
\!1 & \!1 & \!0 & \!0
\end{bmatrix},
\begin{bmatrix}
\!1 & \!0 & \!1 & \!0
\end{bmatrix},
\begin{bmatrix}
\!1 & \!0 & \!0 & \!1
\end{bmatrix},
\begin{bmatrix}
\!0 & \!1 & \!0 & \!1
\end{bmatrix},
\begin{bmatrix}
\!0 & \!1 & \!1 & \!0
\end{bmatrix},
\begin{bmatrix}
\!0 & \!0 & \!1 & \!1
\end{bmatrix}
\right\}
```
For an LLM, there exists a substantial number of parameter blocks, denoted as $\left\{\mathcal{W}_i\right\}$, each requiring the selection of corresponding masks $\left\{\mathcal{M}_i\right\}$. For $N:M$ sparsity, we can define this objective for learning mask selection.
<!-- ```math
\large \left\{\mathcal{M}_i^*\right\} = \argmin_{\left\{\mathcal{M}_i \mid \mathcal{M}_i \in \mathcal{S}^{2:4}\right\}} 
\mathbb{E}_{x \sim p(x)} 
\left[ 
\mathcal{L}_{\mathrm{LM}}(x; \{\mathcal{W}_i \odot \mathcal{M}_i\}) 
\right]
```
But, as we can see, finding the optimal combination of masks as selection from a set of discrete elements is non-differentiable.
### 4.2 Walkaround
Directly determining the exact optimal mask for a block is not feasible as the behavior of pruned LLMs should also depend on the pruning of other parameter blocks. The intuition is to sample masks independently for each block and measure the overall model quality after pruning. Consequently, let's define a categorical distribution with class probability $p_1, p_2, \cdots, p_{\lvert S \rvert}$ such that $\sum_{j} p_j = 1$. During the random sampling phase, if a certain mask achieves good quality during pruning, adjust the categorical distribution by increasing the probability of the sampled mask. With sufficient sampling and updates, the authors conjectured that this process would end with a distribution where the mask with high probability is more likely to maintain good quality after pruning. Thus the objective becomes
```math
\large \left\{p^*(\mathcal{M}_i)\right\} = \argmin_{\left\{p(\mathcal{M}_i)\right\}}
\;\mathbb{E}_{x \sim p(x), \mathcal{M}_i \sim p(\mathcal{M}_i)} 
\left[ 
\mathcal{L}_{\mathrm{LM}}(x; \{\mathcal{W}_i \odot \mathcal{M}_i\}) 
\right]
```
where $p(\mathcal{M}_i)$ refers to the categorical distribution of $i$-th mask $\mathcal{M}_i$. But, drawing samples from a categorical distribution is still non-differentiable.<br>
So, we need to find out a way to do differentially sampling of masks. We can model the differential sampling operation using Gumbel softmax trick, which introduces additional noise variable into the soft sampling process following gumbel distribution. Gumbel distribution is defined as
```math
g_i = -\log(-\log \epsilon_i), \; \epsilon_i \sim U(0,1)
```
Gumbel softmax would lead to a soft and differentiable index $\large \tilde{y} = \left[\tilde{y_1}, \tilde{y_2}, \cdots, \tilde{y_{\lvert S \rvert}}\right]$
```math
\large \tilde{y_{i}} = \frac{\exp((\log p_i + g_i) / \tau)}{\sum_{j} \exp((\log p_j + g_j) / \tau)}
```
The temperature term $\tau$ is a hyper-parameter, controlling the hardness of the sampled index. While $\tau \to 0$, the soft index will be more close to a one-hot vector, resulting in $\tilde{y_i} = y_i$. Now, we can get differentiable mask $\tilde{\mathcal{M}}$ as matrix multiplication
```math
\tilde{\mathcal{M}} = \tilde{y} \times S
```
During evaluation, the masks would be selected based on argmax on $\tilde{y}$.<br>
The authors also don't learn probability directly, instead they learn logits $\pi$ with a scaling factor $\kappa$, which produces probability $\large p_i = \frac{\exp(\pi_i \cdot \kappa)}{\sum_j \exp(\pi_j \cdot \kappa)}$. Scaling factor \kappa balances the relative magnitude of logits and gumbel noise, thus controlling the randomness of sampling.<br>
The authors also introduced sparse weight regularization, which maintains a large magnitude in the remaining weights.
```math
\large \left\{p_{\pi}^*(\mathcal{M}_i)\right\} = \argmin_{\left\{p_{\pi}(\mathcal{M}_i)\right\}}
\;\mathbb{E}_{x \sim p(x), \mathcal{M}_i \sim p(\mathcal{M}_i)} 
\left[ 
\mathcal{L}_{\mathrm{LM}}(x; \{\mathcal{W}_i \odot \mathcal{M}_i\}) 
\right]
- \lambda \sum_i \left\|\mathcal{W}_i \odot \mathcal{M}_i\right\|_{2}^{2} 
```
### 4.3 Transfer Learning
The precomputed masks can be obtained through **SparseGPT** or **Wanda**. For transfer learning, we would need to map the precomputed masks back to class probabilities, then MaskLLM would be able to begin with a good initialization for sampling. Given a prior mask $\mathcal{M}_o$, it's similarity to all candidate masks for $N:M$ sparsity is obtained through
```math
\large sim(\mathcal{M}_o, \mathcal{M}_i) = \mathcal{M}_o \mathcal{M}_i^T - N/2
```
Then we increase the probability of candidate masks based on the similarity.
```math
\large \pi_i^{'} = \pi_i + \sigma(\pi) \cdot sim(\mathcal{M}_o, \mathcal{M}_i) \cdot \alpha
```
## 5. Results
The perplexity and accuracy comparison with Magnitude, SparseGPT and Wanda on various datasets. Authors suspect that the superiority on performance mainly arises from this factor. Difficulty of computing the pruning error, Existing methods use approximated metrics to estimate weight importance where this paper finds weight importance through end-to-end training on large-scale datasets, optimizing language modeling loss function.
<br><br>
<!-- <img src="Comparison.png" width="800" height="600"><br><br> -->
![Alt text](Comparison.png)<br>
Increasing the calibration set beyond 256 does not improve SparseGPT performance, but MaskLLM scales to large datasets.<br>
<!-- <img src="Sample vs Perplexity.png" width="800" height="600"><br><br> -->
![Alt text](Sample_vs_Perplexity.png)<br>
Using prior masks pre-computed by one-shot methods can provide substantial benefits. We can initialize the Gumbel logits with pre-computed masks,which significantly accelerate the training.<br>
<!-- <img src="Transfer Learning.png" width="800" height="200"><br><br> -->
![Alt text](Transfer_Learning.png)<br>
If certain layers are pruned to a small magnitude, the gradients passed to their inputs will also diminish, thereby impeding mask learning and transfer to downstream tasks. Thats why sparse weight regularization provides a superior performance.<br>
<!-- <img src="Regularization.png" width="400" height="200"><br><br> -->
![Alt text](Regularization.png)<br>
To deploy sparse LLMs for a single downstream task, the pre-computed general mask can be picked or an expert mask from scratch can be trained. However, both strategies show a drop in performance compared to the dense LLM as either they loss domain specific parameters for general mask or see limited data from target domain in case of expert mask. If general mask is leveraged as a prior and then transfer learning is applied to learn the masks, the model performance improves.<br>
<!-- <img src="Transfer Learning for downstream tasks.png" width="800" height="200"><br><br> -->
![Alt text](Transfer_Learning_for_downstream_tasks.png)<br>

## 6. Insights and Future Works
- The superior performance is likely due to end to end training with optimizing the language modeling loss function, an well established training method to train LLMs. On the other hand, previous methods applied heuristics as they didn't attempt to work on a learnable optimal masking strategy, which clearly is an inferior strategy.
- Let's say we have a $4 \times 4$ linear layer and we want to apply $2:4$ sparsity. Then, we would have $4$ contiguous blocks of parameters. Let the optimal $2:4$ sparsity mask for this linear layer is $\left[1, 0, 0, 1\right]$ Then, the mask for the linear layer should be 
$\begin{bmatrix}
1 & 0 & 0 & 1 \\
1 & 0 & 0 & 1 \\
1 & 0 & 0 & 1 \\
1 & 0 & 0 & 1 \\
\end{bmatrix}
$<br>
I suspect that, for each row, different columns might be more important but this methodology cannot address this. So, one future work can extend to finding out the optimal masking incorporating this concern.
- The authors did not provide an upper bound on how much of a performance degradation this methodology would cause compared to the dense LLM model. So, one future work can extend to finding out the upper bound of performance deviation estimation given that the optimal masking strategy is trained on a sufficient amount of data with rigorous mathematical analysis. -->