# Alternatives to the transformer and to LLM-scale training, 2017 to October 2026

Research brief, part 2 of 2. Written 12 October 2026.

Verification convention: a citation marked [n] links to the numbered reference list at the end. Each was checked by live search or fetch during writing (October 2026). Anything I could not confirm against a primary or reputable secondary source is marked UNVERIFIED.

## Contents

1. Efficient sequence models
2. Retrieval and memory
3. Yann LeCun's program
4. Energy-based and alternative learning rules
5. Active inference and the free-energy principle
6. Neurosymbolic methods and program synthesis
7. Compositional generalization
8. Category theory and structure
9. Hyperdimensional computing and vector-symbolic architectures
10. Data-efficient language learning
11. Test-time compute and in-context learning theory
12. Other alternatives with traction
13. Synthesis
14. References

## Summary up front

Most of the alternatives that reached production since 2017 cut the cost of running a model on long inputs. None of them has been shown to cut the amount of text needed to train a capable language model by a large factor. The lines that do attack data efficiency (meta-learning for compositionality, program synthesis, JEPA-style prediction in representation space, curriculum and architecture work in the BabyLM challenge) have results on narrow tasks or small budgets, and none has yet produced a general language model that competes with a transformer trained on trillions of tokens. The most useful lesson from the transformer is that its core operation, content-addressed lookup over everything in context, is what every cheaper replacement keeps having to add back.

## 1. Efficient sequence models

### Core idea

Attention compares every token with every earlier token, so the cost of a forward pass grows with the square of the sequence length and the key-value (KV) cache grows linearly with context during generation. Efficient sequence models replace that comparison with a fixed-size state that is updated once per token. Three families converged on the same shape:

- **State-space models (SSMs).** A continuous linear system x'(t) = A x(t) + B u(t), y(t) = C x(t) + D u(t), discretized and unrolled over the sequence [1]. With fixed A, B, C the whole layer is a long convolution and can be trained in parallel.
- **Linear attention.** Replace softmax(QK^T)V with a kernel feature map φ, so that attention becomes φ(q_i)^T S_i with a running state S_i = S_{i-1} + φ(k_i) v_i^T. Katharopoulos et al. showed this makes a causal transformer layer an RNN with constant memory per step [4].
- **Delta-rule memories.** Instead of adding every key-value pair to the state, overwrite the value stored under the current key: S_t = S_{t-1}(α_t(I − β_t k_t k_t^T)) + β_t v_t k_t^T, the gated delta rule [8]. This is the Widrow-Hoff rule applied as fast weights, and it is the form most production hybrids use in 2026.

Dao and Gu proved that these families are views of one object: an SSM is multiplication by a semiseparable matrix, it has an attention-like quadratic dual form (L ∘ QK^T)V, and any kernel attention with a fast recurrent form is an SSM [3].

### Landmark works

| Year | Work | Authors | Link |
|---|---|---|---|
| 2020 | Transformers are RNNs (linear attention) | Katharopoulos, Vyas, Pappas, Fleuret | [4] |
| 2021 | S4: Efficiently Modeling Long Sequences with Structured State Spaces | Gu, Goel, Ré | [1] |
| 2023 | RWKV: Reinventing RNNs for the Transformer Era | Peng et al. | [6] |
| 2023 | RetNet: Retentive Network | Sun, Dong et al. (Microsoft Research, Tsinghua) | [5] |
| 2023 | Mamba: selective state spaces | Gu, Dao | [2] |
| 2024 | Mamba-2: Transformers are SSMs (state space duality) | Dao, Gu | [3] |
| 2024 | DeltaNet, parallelized over sequence length | Yang, Wang, Zhang, Shen, Kim | [7] |
| 2024 | Gated DeltaNet | Yang, Kautz, Hatamizadeh (MIT, NVIDIA) | [8] |
| 2024 | Jamba hybrid | Lieber, Lenz et al. (AI21) | [9] |
| 2025 | MiniMax-01 (lightning attention hybrid) | MiniMax | [13] |
| 2025 | Qwen3-Next (Gated DeltaNet hybrid) | Qwen team, Alibaba | [10] |
| 2026 | Qwen3.8-Flash-Next | Qwen team, Alibaba | [11], [12] |

### Results with numbers

- **S4** reached 91% on sequential CIFAR-10, averaged 80.48% on Long Range Arena where every earlier baseline was under 60%, and was the first model to solve Path-X (length 16,384) at 88%; the paper reports generation 60x faster than a transformer on its language and image tasks [1].
- **Linear attention** was up to 4000x faster than a softmax transformer for autoregressive prediction of very long sequences, at similar quality on image generation and speech recognition [4].
- **RetNet** reported, for a 7B model at 8k context, 8.4x faster decoding and 70% less memory than a transformer with a KV cache, and said it began to beat the transformer above about 2B parameters [5].
- **RWKV** scaled a dense RNN to 14B parameters and reported parity with similarly sized transformers [6].
- **Mamba** made the SSM parameters Δ, B, C functions of the input ("selection"). Mamba-3B beat same-size transformers and matched transformers twice its size; generation throughput was 5x a transformer of similar size [2].
- **Mamba-2** was 2 to 8x faster than Mamba's scan, crossed over FlashAttention-2 at 2K tokens and was 6x faster at 16K. A 2.7B Mamba-2 trained on 300B Pile tokens beat Pythia-6.9B trained on the same data [3].
- **DeltaNet** at 1.3B parameters and 100B tokens beat Mamba and GLA on perplexity and zero-shot tasks; at 3B and 1T tokens it slightly underperformed a matched transformer (PowerLM-3B) [7]. **Gated DeltaNet** beat Mamba-2 and DeltaNet on language modeling, commonsense reasoning, in-context retrieval and long-context benchmarks [8].
- **NVIDIA's controlled study** trained 8B Mamba, Mamba-2 and transformer models on the same data up to 3.5T tokens. Pure SSMs matched or beat transformers on many tasks but lagged on five-shot MMLU, phonebook lookup and long-context reasoning. A hybrid of 43% Mamba-2, 7% attention and 50% MLP layers beat the transformer on all 12 standard tasks (+2.65 points average) and was projected to generate up to 8x faster [17].

### The recall limitation

Two papers explain why pure fixed-state models fall short.

- **Zoology** (Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré; ICLR 2024) trained 17 models from 70M to 1.4B parameters on the Pile. Gated-convolution models trailed attention by up to 2.1 perplexity points, and 82% of that gap came from associative-recall tokens (predicting a token by looking up an earlier pairing), which are only 6.4% of Pile tokens. A 70M attention model beat a 1.4B Hyena model on those tokens (2.41 vs 3.43 perplexity). On their multi-query associative recall task (MQAR), two-layer gated-convolution models needed model dimension at least linear in sequence length, while attention solved it at a constant dimension of 64 [15].
- **Repeat After Me** (Jelassi, Brandfonbrener, Kakade, Malach; ICML 2024) proved that a two-layer transformer can copy strings of length exponential in its size, while any model with a fixed-size latent state is limited by that state. Pretrained transformers "dramatically outperform" pretrained SSMs at copying and retrieving from context [16].

The production record agrees. MiniMax-01 found that pure lightning (linear) attention matched softmax attention on most downstream tasks but failed needle-in-a-haystack retrieval, and so put one softmax layer after every seven linear layers [13]. Its own fitted scaling laws show lightning attention needing different optimal model size and token counts for a compute budget, with no reduction in the total data regime [13].

### Production hybrids in 2026

- **Jamba** (AI21, March 2024): interleaved transformer and Mamba layers with mixture-of-experts, 12B active of 52B total parameters, 256K context, 3x the long-context throughput of Mixtral-8x7B and an 8x smaller KV cache than a vanilla transformer; it fits on one 80 GB GPU at 140K tokens [9].
- **MiniMax-Text-01** (January 2025): 456B total and 45.9B active parameters, 7:1 lightning-to-softmax layers, trained at 1M tokens of context and extrapolating to 4M [13]. Its successor **MiniMax-M2** (October 2025) went back to full attention. The M2 pretraining lead wrote that the hybrid "had clear deficits in complex, multi-hop reasoning tasks" at scale, that linear attention is more sensitive to numerical precision, and that "the race for efficient attention is a race to save compute"; to save tokens instead, "you'd probably bet on other paths" [14].
- **Qwen3-Next-80B-A3B** (September 2025): layout of 12 blocks of (3 Gated DeltaNet layers, then 1 gated attention layer), each followed by a sparse MoE [10].
- **Qwen3.8-Flash-Next** (released 26 August 2026, verified): 125B parameters with 6B active plus 51B n-gram embedding parameters. It keeps the 3:1 Gated DeltaNet to attention ratio and replaces the attention layer with block-level "Qwen Sparse Attention". Alibaba states the hybrid Gated DeltaNet design from Qwen3-Next has been used across the Qwen3.5 through Qwen3.8 series, that training cost is about 1/9 that of Qwen3.7-Plus, and that QSA gives 7.6x prefill and 4.9x decode kernel speedups at 1M tokens [11], [12]. Those are vendor figures without an independent replication I could find.

### Drawbacks and failure modes

- Fixed-size state cannot hold arbitrary amounts of exact context, which shows up as weak copying, lookup, multi-hop retrieval and few-shot format following [15], [16], [17].
- Linear-attention kernels are often memory-bound in practice, and the crossover point against full attention is a few thousand tokens in theory but harder to reach in serving because of prefix caching and speculative decoding [14].
- Every production system that adopted these layers kept some attention: 1 in 8 layers for MiniMax-01, 1 in 4 for Qwen, roughly 1 in 8 for Jamba and 7% in NVIDIA's hybrid.

### Status in 2026

Adopted, as hybrids. Alibaba's Qwen line has used Gated DeltaNet hybrids for a year; AI21 ships Jamba; NVIDIA publishes Mamba-2 hybrids; MiniMax adopted then reverted. Pure SSM and pure linear-attention language models remain research or niche (RWKV community releases).

### Relevance to Andrew's goal

This line meets half of the compute goal: per-token inference cost and memory that do not grow with context. It does nothing measurable for data efficiency. Every model above was trained on hundreds of billions to trillions of tokens, and the scaling-law fits for linear attention sit in the same regime as softmax [13]. The math that made it work (structured matrices, semiseparable decompositions, Householder products, the delta rule as online regression) is a useful toolkit for building any cheaper model. The recall results set a requirement: whatever replaces attention must still do exact content-addressed lookup over context, or it loses in-context learning.

## 2. Retrieval and memory

### Core idea

Separate what a model knows from how it computes. Facts live in an external store (a text database or a table of key-value vectors) that the network queries by similarity, so the weights only need to hold the skill of using what comes back. The common formalism is soft associative lookup: given a query q and stored keys K with values V, return softmax(β q K^T) V. Ramsauer et al. showed this is exactly one update step of a continuous modern Hopfield network with energy E(ξ) = −β^{-1} log Σ_i exp(β x_i^T ξ) + ½ ξ^T ξ + const, which can store a number of patterns exponential in the dimension and retrieve one in a single update [21]. Read this way, transformer attention is already an associative memory over the context, and retrieval models extend the same operation to a corpus.

### Landmark works and results

- **kNN-LM** (Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis; ICLR 2020). Interpolate a trained LM's next-token distribution with a k-nearest-neighbor distribution over (context embedding, next token) pairs from a datastore. On Wikitext-103 perplexity fell to 15.79, 2.86 points better than the base model, with no extra training. A model trained on 100M tokens plus kNN search over 3B tokens beat the same model trained on all 3B tokens. Gains concentrated on rare patterns and facts [18]. The authors concluded that "learning similarity between sequences of text is easier than predicting the next word" [18].
- **RETRO** (Borgeaud et al., DeepMind; ICML 2022). A transformer that retrieves neighbor chunks from a 2-trillion-token database with a frozen BERT retriever and reads them through chunked cross-attention. It matched GPT-3 and Jurassic-1 on the Pile with 25x fewer parameters [19].
- **Memorizing Transformers** (Wu, Rabe, Hutchins, Szegedy; ICLR 2022). An approximate kNN lookup into a non-differentiable memory of past (key, value) pairs. Language modeling improved steadily as memory grew to 262K tokens, and on code and math the model used functions and theorems defined only at test time [20].
- **Modern Hopfield networks** (Ramsauer et al., 2020, ICLR 2021). Beyond the theory, Hopfield layers set state-of-the-art on several multiple-instance-learning problems, immune repertoire classification and small UCI tabular datasets where deep learning usually does poorly [21].
- **Reproduction and limits of RETRO** (Wang, Ping et al., NVIDIA; EMNLP 2023). Reproduced RETRO up to 9.5B parameters with a 330B-token database. RETRO beat GPT on knowledge-intensive tasks, produced less repetition and slightly higher factual accuracy, but was "on par with GPT on other tasks" [22].
- **Memory Layers at Scale** (Berges et al., Meta, December 2024). Trainable key-value lookup layers inside the network, scaled to 128B memory parameters and 1T training tokens. Memory-augmented models beat dense models with more than twice the compute and beat compute- and parameter-matched MoE, with the largest gains on factual tasks [23].
- **Titans** (Behrouz, Zhong, Mirrokni, Google; January 2025). A neural long-term memory whose weights are updated at test time by gradient descent on an associative loss, with "surprise" measured by the gradient; the authors show the update is equivalent to mini-batch gradient descent with momentum and weight decay, and generalizes the forgetting gates of Mamba-2 and DeltaNet. Titans scaled past 2M tokens of context with better needle-in-haystack accuracy than baselines [24].
- **N-gram embedding tables in production.** Qwen3.8-Flash-Next adds 51B parameters of n-gram lookup embeddings that can sit in host memory and be prefetched, which Alibaba describes as a "local-pattern memory" at little compute cost [12]. This is retrieval-as-parameters in a shipping model.

### Drawbacks and failure modes

- Retrieval helps facts and rare strings much more than reasoning. NVIDIA's reproduction found no gain on non-knowledge tasks [22]; Meta's memory layers also gained most on factual tasks [23].
- RETRO's headline perplexity depends partly on overlap between evaluation text and the retrieval database; the RETRO paper itself analyzes this leakage and reports results by overlap level [19].
- Datastores are large. kNN-LM stores one vector per training token; RETRO's database is 2T tokens. The cost moves from FLOPs to storage and approximate search.
- The retriever's notion of similarity is fixed by a separate encoder (frozen BERT in RETRO), so the model cannot easily learn what to look up.

### Status in 2026

Retrieval-augmented generation is standard practice at inference time, layered onto ordinary LLMs. Retrieval during pretraining in the RETRO style is niche; I found no frontier model that discloses RETRO-style pretraining. Learned memory as parameters (memory layers, n-gram tables) is entering production through Meta's research and Alibaba's Qwen3.8-Flash-Next [12], [23]. Test-time-trained memories (Titans and the delta-rule family) are the active research front.

### Relevance to Andrew's goal

This is the clearest existing evidence that parameters and training text can be traded for an explicit store. kNN-LM beat a model trained on 30x more tokens by searching those tokens instead of training on them [18], and RETRO matched models 25x larger [19]. The weakness is that the gains are in knowledge, and the underlying model still needs large-scale training to learn language. The Hopfield result matters for the "direct representation" hypothesis: content-addressed memory has a clean energy-function mathematics, and attention is one instance of it. A system that learns concepts once and stores them in an addressable memory, instead of spreading them across weights by gradient descent, would follow from this line.

## 3. Yann LeCun's program: JEPA, world models and energy-based models

### Core idea

LeCun's 2022 position paper proposes an agent built from a configurable world model, a cost module, an actor and short-term memory, trained mainly by self-supervised prediction [25]. The central component is the Joint-Embedding Predictive Architecture (JEPA). An encoder maps an observed part x and a target part y to representations s_x and s_y; a predictor, given a latent variable z describing the relation between them, predicts s_y from s_x. Training minimizes a distance between the predicted and actual representation, D(s_y, Pred(s_x, z)), which can be read as an energy E(x, y) that is low for compatible pairs. Predicting in representation space lets the model drop unpredictable detail (exact pixels, exact wording) and keep what is predictable. The hard part is preventing collapse, where the encoder maps everything to a constant and the energy is trivially zero. The 2022 paper lists regularizers that limit the information in latent variables (quantization, rank minimization, sparsity, noise) [25]. LeJEPA (Balestriero and LeCun, November 2025) gives a cleaner answer: it proves that an isotropic Gaussian embedding distribution minimizes downstream prediction risk and enforces it with a single regularizer, SIGReg, removing stop-gradients and teacher-student networks [30].

### Landmark works and results

- **A Path Towards Autonomous Machine Intelligence** (LeCun, 2022, OpenReview) [25].
- **I-JEPA** (Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas; CVPR 2023). From one image block, predict the representations of other blocks. A ViT-Huge/14 trained on ImageNet on 16 A100 GPUs in under 72 hours (under 1,200 GPU hours), over 10x more efficient than a ViT-H/14 trained with MAE pixel reconstruction, and converging in about 5x fewer iterations [26].
- **V-JEPA** (Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas; 2024). Feature prediction on 2M videos with no text, negatives or reconstruction. A ViT-H/16 reached 81.9% on Kinetics-400, 72.2% on Something-Something-v2 and 77.9% on ImageNet-1K with a frozen backbone, and lost less accuracy than pixel-reconstruction models when labels were scarce [27].
- **V-JEPA 2** (Assran et al., Meta, June 2025). Pretrained on over 1M hours of internet video; 77.3% top-1 on Something-Something-v2 and 39.7 recall-at-5 on Epic-Kitchens-100 action anticipation. An action-conditioned predictor (V-JEPA 2-AC, 300M parameters) post-trained on under 62 hours of unlabeled robot video planned pick-and-place zero-shot on Franka arms in two new labs, taking 16 seconds per action against 4 minutes for the Cosmos video-generation world model [28].
- **LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero; March 2026). The first JEPA trained stably end-to-end from raw pixels with two loss terms; 15M parameters, trainable on a single GPU in a few hours, planning up to 48x faster than world models built on foundation encoders [31].
- **H-JEPA** (Zhang, Terver, Rabbat, LeCun, Balestriero; AMI Labs, NYU, INRIA, Brown; arXiv 2610.06805, October 2026). A hierarchy of JEPAs, each predicting farther ahead in its own latent space; on Visual AntMaze a three-level hierarchy raised planning success from 18% to 73% with less planner compute. Higher levels drop fast-changing detail and keep slower task-relevant state [32].
- **RoboJEPA** (Meta FAIR, October 2026). An 8B-parameter action-conditioned JEPA predictor on frozen V-JEPA 2.1 features, trained on 23 robot datasets; its prediction error follows a power law in compute that extrapolated beyond the fitted range [33].
- **LLM-JEPA** (Huang, LeCun, Balestriero; ICLR 2026). The first JEPA objective for language models, added to the usual next-token loss. It reports outperforming standard training objectives "by a significant margin" on NL-RX, GSM8K, Spider and RottenTomatoes across Llama 3, OpenELM, Gemma 2 and OLMo models, and less overfitting [29]. The objective needs two views of the same content (for example a natural-language description and its regex or SQL), and the paper calls the lack of JEPA-style LLMs "a testimony of the challenge in designing such objectives for language" [29].

### The argument against autoregressive LLMs

LeCun's slides from 2023 and 2024 state: "Auto-Regressive LLMs are doomed. They cannot be made factual, non-toxic, etc. They are not controllable." The argument: if e is the probability that any generated token leaves the set of correct answers, an answer of length n is correct with probability (1 − e)^n, which "diverges exponentially. It's not fixable (without a major redesign)" [34]. The same slides say autoregressive LLMs "have a constant number of computational steps between input and output" and do not really reason or plan [34]. He made the same (1 − e)^n argument in a LinkedIn post on 27 March 2023, where commenters disputed the math [35]. My assessment: the argument assumes per-token errors are independent and unrecoverable. Reasoning models that spend variable compute on long chains of thought and correct themselves, which arrived in 2024 and 2025, weaken both assumptions in practice, though they do not answer his separate point about learning from text alone.

### Departure from Meta and AMI Labs (verified)

- LeCun announced his departure from Meta in November 2025 after 12 years and founding FAIR. He told WIRED that Meta "had to basically catch up with the industry on LLMs and kind of do the same thing that other LLM companies are doing, which is not my interest" [36].
- On 9 and 10 March 2026, Advanced Machine Intelligence (AMI Labs), based in Paris with offices in New York, Montreal and Singapore, announced a $1.03B (about €890M) seed round at a $3.5B valuation, co-led by Cathay Innovation, Greycroft, Hiro Capital, HV Capital and Bezos Expeditions [36], [37], [38]. LeCun is executive chairman and remains an NYU professor; Alexandre LeBrun is CEO; Saining Xie is chief science officer; Michael Rabbat leads world models [38].
- Direction: "a new breed of AI systems that understand the world, have persistent memory, can reason and plan, and are controllable and safe," built on JEPA world models, sold first to industrial partners (WIRED names Toyota and Samsung) before broader use [36], [37]. In an October 2026 interview LeCun said AMI has 50 to 60 staff, is in a research phase with no short-term product plan, and will publish research first [39]. Its published output so far is LeWorldModel and H-JEPA, both robotics and planning work [31], [32]. I found no AMI language model.

### Drawbacks and failure modes

- Almost all of the evidence is in vision, video and control. The only language result (LLM-JEPA) is a fine-tuning or pretraining add-on to a next-token model on tasks with paired views, at modest scale [29].
- Collapse prevention took years of heuristics (EMA teachers, stop-gradients, VICReg terms) before LeJEPA's SIGReg, and end-to-end training from pixels was only stabilized in 2026 [30], [31].
- V-JEPA 2 still needed 1M hours of video. JEPA lowers compute per unit of representation quality, and its few-shot efficiency appears downstream, after large-scale pretraining [27], [28].
- Planning with these models uses sampling-based optimizers (cross-entropy method) that cost seconds per action [28].

### Status in 2026

Active and funded, with a dedicated company. Mature in vision representation learning (I-JEPA, V-JEPA 2 are widely cited); early in planning; nearly absent in language.

### Relevance to Andrew's goal

This is the program closest to Andrew's hypothesis that ideas have a direct abstract representation: the training target is a representation, and the model is meant to discard surface form. The label-efficiency findings (V-JEPA losing less accuracy with few labels [27]; LeJEPA in-domain pretraining beating DINOv2 transfer from one-shot to full supervision on Galaxy10 [30]) are evidence of data efficiency downstream of pretraining. For language, the question that is still open is what the "two views" of a sentence are, so that a JEPA can predict one from the other without reconstructing words. Paraphrase, translation, and text paired with code or with perception are candidate views. The energy-based framing, D(s_y, Pred(s_x, z)) with a regularized latent, is a real mathematical alternative to likelihood over tokens, and LeJEPA's isotropic-Gaussian result is one of the few theorems about what a good representation distribution is.

## 4. Energy-based and alternative learning rules

### Core idea

Backpropagation needs a separate backward pass, symmetric weights and storage of every activation. Local learning rules try to compute weight updates from quantities available at each synapse, which would allow learning on analog or neuromorphic hardware without moving activations through memory. Three lines are active:

- **Predictive coding (PC).** Each layer predicts the activity of the layer below; the network minimizes a sum of squared prediction errors F = Σ_l ½ ||x_l − f(W_l x_{l+1})||², first by relaxing activities, then by updating weights with a local Hebbian rule. Whittington and Bogacz (2017) showed that for certain parameters the PC weight change converges to the backpropagation gradient [40].
- **Equilibrium propagation (EP).** An energy-based network settles to a fixed point (free phase), is weakly nudged toward the target by adding β·C to the energy (nudged phase), and updates weights from the difference of local statistics between the two equilibria. As β → 0 the update equals the gradient of the loss (Scellier and Bengio, 2017) [41].
- **Forward-forward (FF).** Hinton (2022) replaces the backward pass with two forward passes, one on real ("positive") data and one on negative data, and trains each layer to give high "goodness" (for example, the sum of squared activities) on positive data and low goodness on negative data [42].

### Results with numbers

- **FF**: a 4-layer, 2000-unit network reached 1.36% test error on permutation-invariant MNIST after 60 epochs, against about 1.4% for backpropagation in about 20 epochs. On CIFAR-10 with locally connected layers, FF had 41% to 46% test error against 37% to 39% for backpropagation, and Hinton notes that backpropagation reduces training error much faster [42].
- **EP**: the original 2017 paper trained networks with 1 to 3 hidden layers on MNIST [41]. Laborieux, Ernoult, Scellier, Bengio and Grollier (2021) found that finite nudging biases the gradient; a symmetric three-phase estimate cut CIFAR-10 test error from 86.6% to 11.7%, within 0.6 points of backpropagation through time on the same convolutional network [43].
- **PC**: the PCX benchmark (Pinchetti et al., 2024) found PC matches backpropagation for networks up to about 7 layers (VGG-7), including on CIFAR-100 and Tiny ImageNet, but performance falls as depth grows (VGG-5 > VGG-7 > VGG-9 > ResNet), the reverse of backpropagation. The authors trace the failure to energy imbalances across layers that make gradients exponentially small in deeper models [44].

### Drawbacks and failure modes

- None of these has trained a model at the scale of even a small language model. The largest published comparisons are CIFAR-scale and Tiny-ImageNet-scale vision [43], [44].
- Settling to equilibrium (EP, PC) costs many inference iterations per example, so on GPUs these methods are slower than backpropagation; the benefit only appears on hardware that settles physically.
- FF needs a source of negative data and learns more slowly [42].

### Status in 2026

Niche research, mainly in computational neuroscience and neuromorphic hardware groups. I found no language-model result from any of these rules that approaches backpropagation-trained baselines.

### Relevance to Andrew's goal

These rules attack the hardware cost of training, and only if paired with analog or in-memory hardware. They do not reduce the data needed: they approximate the same gradient on the same data, and at best match backpropagation. The mathematics is worth knowing because it ties learning to energy minimization, the same frame as Hopfield memory (Section 2) and JEPA (Section 3), but the evidence base is small networks.

## 5. Active inference and the free-energy principle

### Core idea

Friston's free-energy principle (FEP) says that any system that maintains itself separate from its environment (across a "Markov blanket") behaves as if it minimizes variational free energy, F = E_q[log q(s) − log p(o, s)] = KL[q(s) || p(s | o)] − log p(o), an upper bound on surprise. Perception updates the belief q; action changes observations o to fit predictions ("active inference"); planning chooses policies that minimize expected free energy, which mixes goal-seeking with information-seeking. Friston and colleagues' 2023 Physics Reports paper derives it from a Langevin description of random dynamical systems and calls the result "a Bayesian mechanics that can be read as a physics of sentience" [45].

### Evidence, stated plainly

- **Supporting.** Isomura, Kotani, Jimbo and Friston (Nature Communications, 2023) cultured rat cortical neurons, fed them mixtures of two hidden sources, and found that a free-energy model fitted on sessions 1 to 10 predicted synaptic changes with under 4% error and over 80% of neuronal responses at session 100; drugs that changed excitability disrupted inference as the model's priors predicted [46]. This is a real quantitative test, at the scale of a dish of neurons.
- **Engineering.** VERSES AI's AXIOM (Heins et al., 2025, with Friston as co-author) is an object-centric active-inference agent using expanding Bayesian mixture models and no gradient descent. It learned simplified pixel games within 10,000 interaction steps, beat deep RL baselines in that low-data regime on the authors' own Gameworld 10k benchmark, and used an order of magnitude fewer parameters [47]. The games were designed by the authors with single-color sprites, and the paper says AXIOM has not been run on standard complex control tasks [47].
- **Critical.** Bruineberg, Dołęga, Dewhurst and Baltieri (Behavioral and Brain Sciences, 2022) argue that the literature confuses Markov blankets as a tool of Bayesian inference ("Pearl blankets") with a metaphysical boundary between agent and world ("Friston blankets"), and that the second use is not justified by the mathematics [48]. A second criticism, widely repeated but which I did not trace to one primary source in this pass (UNVERIFIED attribution), is that the FEP is unfalsifiable as a principle, since almost any behavior can be described as minimizing free energy under some generative model. Friston's group frames it as a normative principle in the same way [45]; the falsifiable claims live in specific process models such as the one Isomura tested [46].

### Status in 2026

Influential in theoretical neuroscience; niche in AI engineering, with one commercial company (VERSES) building on it. I found no language model built on active inference that competes with standard LMs.

### Relevance to Andrew's goal

Active inference's strongest engineering result, AXIOM, is a data-efficiency result: competitive play within 10,000 interactions, with priors about objects built into the model [47]. That success comes from hand-chosen structural priors (objects, piecewise-linear motion), which is close to the expert-system pattern Andrew wants to avoid. The useful idea is the expected-free-energy objective, which makes an agent seek the data that would most reduce its uncertainty. A language learner that chose what to read or ask on that basis might need less text; I found no demonstration of this for language.

## 6. Neurosymbolic methods and program synthesis

### Core idea

Represent a solution as a program in a symbolic language and use a neural network to guide the search for it. A program found from a few examples generalizes exactly wherever its logic applies, which is the data-efficiency property Andrew wants. The search space grows exponentially with program length, so the open problems are which primitives to start with and how to keep the search tractable. Two answers emerged: learn the library of primitives (DreamCoder), or let a large language model propose programs and a verifier check them (AlphaProof, LLM-guided synthesis on ARC).

### Landmark works and results

- **DreamCoder** (Ellis, Wong, Nye, Sablé-Meyer, Morales, Hewitt, Cary, Solar-Lezama, Tenenbaum; PLDI 2021). A wake-sleep loop: in "wake" it searches for programs solving tasks given by one or a few examples; in "sleep" it refactors solved programs with E-graph matching to grow a library of reusable abstractions, and trains a neural search policy on replayed and imagined tasks. Across eight domains, including planning, inverse graphics and equation discovery, learning library and policy together solved more problems faster [49]. The formalism is Bayesian: find the library L maximizing P(L) Π_tasks Σ_programs P(task | program) P(program | L).
- **ARC** (Chollet, "On the Measure of Intelligence", 2019). Defines intelligence as skill-acquisition efficiency given priors and experience, and introduces the Abstraction and Reasoning Corpus: grid puzzles, each with a few input-output demonstrations, built on priors close to innate human ones (objectness, counting, basic geometry) [50]. Chollet argues that "unlimited priors or unlimited training data allow experimenters to 'buy' arbitrary levels of skills" [50].
- **ARC Prize 2024** (ARC-AGI-1, Kaggle, one P100 GPU, 12 hours, no internet). Winner: the ARChitects at 53.5% with a NeMo-Minitron-8B model, heavy augmentation and test-time training (TTT); MindsAI scored 55.5% but did not open-source and was ineligible. Ryan Greenblatt reached 42% on the public leaderboard by having GPT-4o write and debug thousands of Python programs per task; Akyürek et al. reached 53% on the public evaluation set with an 8B model and TTT [51]. In December 2024 OpenAI's o3, trained on 75% of the public training set, scored 75.7% on the semi-private set at $26 per task and 87.5% at 172x the compute ($4,560 per task) [52].
- **ARC Prize 2025** (ARC-AGI-2). 1,455 teams. First place NVARC at 24.03% for about $0.20 per task: a synthetic-data-driven ensemble of an ARChitects-style TTT model and Tiny Recursive Model components. Second, the ARChitects at 16.53% with a 2D-aware masked-diffusion language model and recursive self-refinement. Third, MindsAI at 12.64% with test-time fine-tuning and augmentation ensembles. Chollet's technical report names the "refinement loop" (per-task iterative optimization guided by feedback, in program space or weight space) as the theme of the year and says frontier reasoning remains "fundamentally constrained to knowledge coverage" [53], [54].
- **Tiny Recursive Model** (Jolicoeur-Martineau, Samsung SAIL Montreal, October 2025). A single 2-layer network of 7M parameters applied recursively, trained on about 1,000 examples, reached 44.6% on ARC-AGI-1 and 7.8% on ARC-AGI-2, above DeepSeek R1 (15.8%, 1.3%), o3-mini-high and Gemini 2.5 Pro in the same table, with under 0.01% of their parameters. It improved on the 27M-parameter Hierarchical Reasoning Model (40.3%, 5.0%). It also reached 87.4% on Sudoku-Extreme [55].
- **ARC Prize 2026 and ARC-AGI-3** (launched 25 March 2026). ARC-AGI-3 is interactive: hundreds of game-like environments where agents must explore, infer goals and plan. Over $2M in prizes; submissions due 2 November 2026, results 4 December 2026 [56], [57]. Third-party trackers report, as of 11 October 2026, a top Kaggle ARC-AGI-3 public score of 62.96 (Yi-Chia Chen), with NVARC's 2025 members in the top ten, and a top frontier-model score of 62.7 for OpenAI's GPT-6 Astra on the official harness [58], [59]. I could not read these figures on arcprize.org itself, so treat them as UNVERIFIED against the official source.
- **AlphaGeometry** (Trinh, Wu, Le, He, Luong; Nature, January 2024). A language model trained from scratch on 100 million synthetic theorems and proofs proposes auxiliary constructions; a symbolic deduction engine (deductive database plus algebraic reasoning) does the rest. It solved 25 of 30 recent olympiad geometry problems against 10 for the previous best method, with no human demonstrations. Symbolic deduction alone solved 14; the language model's constructions added 11 [60].
- **AlphaProof** (Hubert et al., Google DeepMind; Nature, November 2025). An AlphaZero-style RL agent that searches for proofs in Lean, trained on millions of auto-formalized problems, with test-time RL that generates and learns from problem variants. At IMO 2024 it solved three of five non-geometry problems including P6 (solved by five human contestants), and with AlphaGeometry 2 on P4 scored 28 of 42, the silver range. Problems were formalized by hand, candidate answers came from Gemini 1.5 Pro, and each proof took 2 to 3 days of compute [61].
- **The counterpoint**: at IMO 2025 an advanced Gemini Deep Think scored 35 of 42 (gold) working end-to-end in natural language within the 4.5-hour limit, with no formal system [62].

### Drawbacks and failure modes

- Hand-designed domain-specific languages do not transfer. DreamCoder's library learning works per domain; nobody has shown it learning the primitives of open-ended language.
- The best ARC results come from neural models with test-time training and from large LLMs, with program synthesis as one ingredient. The 2024 and 2025 Kaggle winners were both test-time-trained neural models [51], [53].
- Chollet's own report warns that frontier ARC scores reflect "knowledge coverage" and new forms of contamination [54].
- AlphaGeometry and AlphaProof rely on a domain where a symbolic checker exists (Euclidean deduction, Lean). Language has no such verifier.
- A year after AlphaProof's IMO silver, a pure language model took gold without formal tools, inside the time limit [62]. On competition mathematics the formal pipeline's lead over plain language models lasted about a year.

### Status in 2026

ARC is the reference benchmark for few-shot abstraction, with four frontier labs reporting scores [54]. Program-synthesis ideas survive as components (verifiers, refinement loops, synthetic-data generators). AlphaProof-style formal provers are active at Google DeepMind and in the Lean community. DreamCoder-style library learning is academic research.

### Relevance to Andrew's goal

This line holds the clearest evidence that small models with the right inductive bias and search can beat large ones on few-shot abstraction: TRM's 7M parameters against DeepSeek R1's 671B, about 100,000 times larger, on ARC [55], and AlphaGeometry learning from synthetic data with no human proofs [60]. The pattern that works is a learned proposer plus an exact checker plus iteration. Its limit for Andrew's purpose is that it needs a formal target language. Program synthesis says what an idea might be (a short program in a learned library, compressing many examples), and DreamCoder's library learning is the closest existing mathematics to "discover the basis that ideas are written in". It has not been made to work on natural language.

## 7. Compositional generalization

### Core idea

Systematic compositionality is the ability to understand new combinations of known parts: once you learn "dax", you understand "dax twice" and "sing and dax". Fodor and Pylyshyn argued in 1988 that neural networks lack it. The formal statement is that meaning is a homomorphism from syntax to semantics, ⟦f(a, b)⟧ = F(⟦a⟧, ⟦b⟧), so a learner that has the parts and the combination rules should need no new data for new combinations. Benchmarks test this by holding out combinations from training.

### Landmark works and results

- **SCAN** (Lake and Baroni, ICML 2018). Commands such as "jump twice and walk left" map to action sequences. Seq2seq RNNs generalized when test commands differed little from training, but when a primitive seen only in isolation had to be composed with others, they "fail spectacularly". The authors suggest that "lack of systematicity might be partially responsible for neural networks' notorious training data thirst" [63].
- **COGS** (Kim and Linzen, EMNLP 2020). Semantic parsing of an English fragment with systematic gaps (new combinations of known structures and words). Transformers and LSTMs scored 96% to 99% in distribution and 16% to 35% on the generalization set, with ±6 to 8 points of variance across random seeds [64].
- **Meta-learning for Compositionality, MLC** (Lake and Baroni, Nature, October 2023). A standard transformer, with no symbolic machinery, is trained on a stream of episodes; each episode has a new randomly generated grammar for a pseudolanguage, a few study examples and queries. To do well across episodes the network must learn how to learn compositional rules. On the same few-shot instruction-learning task given to people, the best run was 100% systematic when choosing its top response; sampling from it gave systematic outputs 82.4% of the time against 80.7% for humans, and it reproduced human error types such as one-to-one translations. Among seven models compared, only MLC had both the systematicity of symbolic models and the flexibility of neural ones. It also improved accuracy on SCAN and COGS few-shot splits [65].

### Drawbacks and failure modes

- MLC's meta-training distribution is generated by the authors' grammars. The network learns to compose in the kind of system it was meta-trained on, and the paper states that it models adult compositional skill and leaves aside how adults acquire it [65].
- Benchmarks like SCAN and COGS are small artificial languages. Large pretrained LLMs improved on many compositional benchmarks, which makes it hard to separate compositional skill from coverage of similar patterns in pretraining data.

### Status in 2026

Active academic line (Lake at NYU, Baroni, Linzen and others). MLC's recipe, training on many small generated tasks so the model learns a learning procedure, overlaps with how in-context learning and synthetic-task pretraining are now studied (Section 11), and with NVARC's synthetic-data approach on ARC (Section 6).

### Relevance to Andrew's goal

This line bears most directly on Andrew's question. The 2018 and 2020 results show that ordinary training gives networks a shallow, data-hungry form of generalization. The 2023 result shows that the same architecture becomes systematic and few-shot when the training distribution is built to reward composition. The mathematics that matters is the meta-learning objective, min_θ E_{grammar G} E_{examples from G}[−log p_θ(query output | query, study examples)], where the expectation runs over randomly generated grammars G, so the network is rewarded for inferring rules from a few examples. That puts the inductive bias in the data distribution. A program for Andrew's goal could take this further: generate structured micro-languages whose meanings have known algebraic form, meta-train a small network to infer and compose them, then test how little real text it needs to learn English.

## 8. Category theory and structure

### Core idea

Category theory describes systems by how parts compose. Three programs apply it to learning:

- **DisCoCat and quantum NLP.** Coecke, Sadrzadeh and Clark (2010) noticed that Lambek's pregroup grammar and the tensor calculus of quantum mechanics share the same compact-closed category structure. Word meanings are tensors whose order is set by grammatical type (a noun is a vector in N, a transitive verb a tensor in N ⊗ S ⊗ N), and grammar contracts them into a sentence vector in S. The sentence meaning is a functor F from the grammar category to vector spaces, so composition is guaranteed by construction, and all sentence meanings live in one space S where an inner product compares them [74]. Since quantum circuits are a restricted family of tensor networks, the same diagrams can run on a quantum computer [66], [67]. DisCoCirc extends this from sentences to whole texts as circuits [68].
- **Categorical deep learning.** Gavranović, Lessard, Dudzik, von Glehn, Araújo and Veličković (ICML 2024 position paper) propose the universal algebra of monads valued in a 2-category of parametric maps as a single theory that covers both the constraints a model must satisfy (such as equivariance) and the implementation of architectures such as RNNs [69].
- **Geometric deep learning.** Bronstein, Bruna, Cohen and Veličković (2021) apply Klein's Erlangen programme: derive architectures from the symmetry group G of the domain, so that layers are G-equivariant, f(g·x) = g·f(x). Convolutional networks (translations), graph networks (permutations) and transformers (permutations of tokens, broken by position encodings) fall out as cases [70].

### Results

- **lambeq** (Kartsaklis et al., Cambridge Quantum, 2021): an open-source library that parses sentences with a CCG parser, turns them into string diagrams and then into tensor networks or parameterized quantum circuits [66].
- **QNLP in practice** (Lorenz, Pearson, Meichanetzidis, Kartsaklis, Coecke; JAIR 2023): the first NLP experiments on noisy quantum hardware with datasets above 100 sentences. Models trained and converged on simple sentence-classification tasks, and results matched expectations for each task [67].
- **QDisCoCirc on trapped ions** (Duneau, Bruhn, Matos, Laakkonen, Saiti, Pearson, Meichanetzidis, Coecke; Quantinuum, 2024): components trained classically on small texts, then composed into larger texts evaluated on Quantinuum's H1-1. Accuracy did not decay on texts larger than those seen in training, while transformer, LSTM and GPT-4 baselines performed near chance on the larger stories. The dataset was a handcrafted toy question-answering task about actors moving in directions [68]. A companion paper proves question answering in QDisCoCirc is BQP-hard, so a classical simulation would need super-polynomial resources under standard conjectures [71].
- **Categorical deep learning** has produced a framework and a company. Symbolica raised a $31M Series A led by Khosla Ventures in April 2024, citing the DeepMind co-authored paper as the basis for transformer alternatives [72]. In February 2026 Symbolica announced an ARC-AGI-2 public-evaluation score of 85.28% using recursive subagents [73]; I could not confirm the method or whether it used category-theoretic models or frontier LLMs underneath, so treat its relation to categorical deep learning as UNVERIFIED.
- **Geometric deep learning** is the part with broad results: CNNs, graph neural networks and equivariant networks are standard in vision, chemistry and physics. Its principle (build in the symmetry, save the data needed to learn it) is the best-tested data-efficiency argument in deep learning [70].

### Drawbacks and failure modes

- DisCoCat-family results are on toy datasets of hundreds of sentences or handcrafted stories [67], [68]. Nobody has shown a DisCoCat model competitive with a statistical language model on open text. The approach needs a parser, so its meanings are only as good as its grammar, and word tensors for high-order types (a ditransitive verb is a fourth-order tensor) are expensive.
- Quantum hardware is noisy and small; the claimed advantage applies to a model-native task defined within QDisCoCirc, and the authors note the hardness result does not cover question answering in other NLP models [71].
- Categorical deep learning, as of the 2024 paper, recovers existing architectures; I found no peer-reviewed result where a category-derived architecture beat standard models on a language benchmark.
- Language has no clean symmetry group comparable to rotations for molecules, which limits how far geometric deep learning carries into text.

### Status in 2026

DisCoCat and QNLP: niche, carried by Quantinuum and Oxford-affiliated groups. Categorical deep learning: early-stage, with one funded startup. Geometric deep learning: adopted in science domains, absorbed into standard practice.

### Relevance to Andrew's goal

DisCoCat is the most explicit existing statement of Andrew's hypothesis: meaning has an algebraic basis, grammar is a map that composes word meanings, and the composition law is fixed by mathematics. QDisCoCirc's result, trained on small texts and generalizing to larger ones where GPT-4 failed, is the kind of data efficiency he wants, on a toy task [68]. What failed is learning the word tensors and the grammar from raw text at scale; the framework assumes a parser and hand-built types. A possible next step is to keep the functorial composition law and learn the types and the category from data, which would join this line to DreamCoder-style library learning (Section 6). Geometric deep learning's lesson applies to any new design: find the invariances of the domain and build them in, because each one removes a factor of data.

## 9. Hyperdimensional computing and vector-symbolic architectures (HDC/VSA)

### Core idea

Represent every symbol as a random vector in a very high-dimensional space (thousands of dimensions), where random vectors are nearly orthogonal. Three operations build structure: **bundling** (superposition, a + b, which stays similar to both), **binding** (a ⊗ b, for example elementwise multiplication of bipolar vectors or circular convolution in holographic reduced representations, which produces a vector dissimilar to both and is invertible), and **permutation** (ρ(a), for sequence order). A record such as {color: red, shape: square} becomes color ⊗ red + shape ⊗ square, and unbinding with color returns a noisy copy of red that a nearest-neighbor cleanup memory restores. The family includes Smolensky's tensor product representations, Plate's holographic reduced representations, multiply-add-permute and binary spatter codes [75].

### Results after 2017

- **Survey and consolidation.** Kleyko, Rachkovskij, Osipov and Rahimi's two-part survey in ACM Computing Surveys (2022) organizes the models and data encodings [75].
- **In-memory hardware.** Karunaratne et al. (IBM Research Zurich and ETH; Nature Electronics, 2020) ran a complete HDC system on two memristive crossbars with 760,000 phase-change memory devices, matching software accuracy on language identification, news classification and gesture recognition. HDC's tolerance to device noise is the reason it suits analog hardware [76].
- **Neuro-vector-symbolic reasoning.** Hersche, Zeqiri, Benini, Sebastian and Rahimi (Nature Machine Intelligence, 2023) used a neural front end that outputs VSA vectors and a probabilistic reasoning back end on Raven's progressive matrices: 87.7% on RAVEN and 88.1% on I-RAVEN, with reasoning two orders of magnitude faster than the symbolic search in prior neuro-symbolic systems [77].
- **Comparative benchmark.** Vergés, Heddes, Nunes, Kleyko and Givargis (Artificial Intelligence Review, 2025) benchmarked HDC classifiers on 121 UCI datasets. HDC generalized from few training instances and held accuracy until about 40% of memory bits were flipped, but "these developments could often be surpassed by off-the-shelf methods" [78].
- **Integration with deep learning.** Carzaniga, Hersche, Schindler and Rahimi (NeSy 2025) report that there are still "no clear guidelines" for using VSAs inside trained networks, and give practical rules (MAP and Hadamard binding 3 to 4x faster than HRR) [79]. A 2025 OpenReview submission uses VSA encodings as probes to read concepts out of LLM embeddings, reaching 0.94 binary accuracy against target encodings [80].

### Drawbacks and failure modes

- Capacity is limited: a bundle of n items in dimension d is recoverable only while n stays well below d, and noise grows with nesting depth.
- Encodings are designed by hand; the representation of "red" carries no meaning until something learns it. HDC on its own has no account of how to learn the symbol vectors from raw text.
- On standard classification it is matched or beaten by ordinary methods [78].

### Status in 2026

Niche but steady: edge and in-memory hardware (IBM Zurich, ETH), cognitive modeling, and neuro-symbolic hybrids. Not used in any production language model I could find.

### Relevance to Andrew's goal

VSA gives an exact algebra for the operations Andrew's hypothesis needs: binding roles to fillers, superposing facts, and recovering parts from wholes, all in fixed-width vectors at linear cost. Attention's key-value lookup and the delta-rule state S += v k^T (Section 1) are close relatives of binding and unbinding, which suggests VSA is a description of what trained networks already do as much as an alternative to them. The gap is learning: VSA says how to compose given symbols, and has no data-efficient way to acquire the symbols. Pairing VSA's algebra with a learner that discovers the role and filler vectors, as NVSA does for vision [77], is an open path for language.

## 10. Data-efficient language learning

### The size of the gap

- **Chinchilla** (Hoffmann et al., DeepMind; NeurIPS 2022) fit loss over 400+ models and found that compute-optimal training scales parameters N and tokens D equally, N_opt ∝ C^0.50 and D_opt ∝ C^0.50, about 20 tokens per parameter. Chinchilla itself was 70B parameters trained on 1.4 trillion tokens, and beat the 280B Gopher trained on 300B tokens at the same compute [83].
- Frontier practice went past Chinchilla because inference is cheaper for smaller, longer-trained models. Llama 3 405B was pretrained on 15.6T tokens with 3.8×10^25 FLOPs [84]; Qwen3 on about 36T tokens across 119 languages, including trillions of synthetic tokens generated by earlier Qwen models [85].
- **Data constraints** (Muennighoff et al., JMLR 2025): with fixed data and compute, up to 4 epochs of repeated data is nearly as good as unique data, after which "the value of adding compute eventually decays to zero" [86]. This is the formal statement that the current recipe is data-bound.
- The BabyLM organizers' framing: children acquire language from under 100 million words, while LLMs "typically require 3 or 4 orders of magnitude more data and still do not perform as well as humans on many evaluations" [81].

### BabyLM challenge (2023, 2024, 2025)

Budgets: Strict track 100M words and Strict-Small 10M words, at or below the organizers' figure of under 100 million words of input for a child acquiring language [81], plus a Loose track in 2023, a multimodal track from 2024 and an interaction track in 2025.

- **2023** (Warstadt et al., CoNLL): 31 papers. Winner of both Strict and Strict-Small: ELC-BERT (Charpentier and Samuel), built on LTG-BERT, a transformer with extra layer normalization, GEGLU feed-forward layers, DeBERTa-style disentangled attention and scaled initialization; ELC-BERT feeds each layer a learned weighted sum of all previous layers. It was trained for over 450 epochs (Strict) and over 2,000 epochs (Strict-Small). LTG-BERT models beat the Llama 2 and RoBERTa-base skylines on overall score and on every test suite except (Super)GLUE. Strict-track aggregate: ELC-BERT 0.74 against 0.60 for the OPT-125M baseline. Curriculum learning, the most common approach, was "largely unsuccessful" [81].
- **2024** (Hu et al., CoNLL): 31 submissions. Winner: GPT-BERT, a single transformer trained with a hybrid masked and causal objective, beating masked-only and causal-only models [82], [87]. No submission beat the baselines in the multimodal track, and the organizers found "a strong relationship between training FLOPs and average performance" [82].
- **2025** (Charpentier, Choshen, Cotterell, Gul, Hu; first BabyLM workshop): a compute cap was added and an interaction track where a student learns from feedback by a teacher (such as Llama 3.1 8B Instruct). Winners included a diffusion masked language model (Strict NLP score), an instruction-tuned model, a mixture-of-experts model (MoEP, Strict-Small: macro average 42.3 against 37.4 for the best baseline) and a reinforcement-learning interactive model (BLM). In the Strict track, no submission beat last year's GPT-BERT baselines on the aggregate metrics. Curriculum learning again won nothing. Some Strict and interaction models reached BLiMP grammaticality scores comparable to Llama 70B, while all stayed below Llama 70B and humans on GLUE. FLOPs correlated with performance only in the interaction track [88].

### TinyStories

Eldan and Li (Microsoft Research, 2023) generated a synthetic corpus of short stories using only words a 3- to 4-year-old understands, written by GPT-3.5 and GPT-4. Models under 10M parameters, or with one transformer block, trained on it produced fluent, consistent multi-paragraph stories with near-perfect grammar, while 125M-parameter general models such as GPT-2 small "can rarely generate coherent and consistent English text beyond a few words" [89]. The result suggests that the breadth of the corpus, more than model size, sets how much capacity coherent text requires.

### Drawbacks and failure modes

- The BabyLM gains come from known transformer techniques (better normalization, hybrid objectives, many epochs, distillation from teachers). The 2023 winners trained for hundreds to thousands of epochs over the same words, so they are data-efficient and compute-hungry [81].
- At 100M words, models match larger models on grammaticality tests (BLiMP) but stay far behind on knowledge and reasoning (GLUE and beyond) [88]. Grammar is learnable from little data; broad world knowledge has not been learned from that little text.
- TinyStories depends on a large teacher model to produce simplified data, so the knowledge was paid for elsewhere [89].
- Curriculum learning, the most intuitive "learn like a child" idea, was the most common approach in 2023 and 2025 and won nothing in either year [81], [88].

### Status in 2026

BabyLM is an established annual workshop with about 30 submissions a year. Synthetic and teacher-generated data, of which TinyStories was an early demonstration, is now standard in frontier pretraining (Qwen3's trillions of synthetic tokens [85]).

### Relevance to Andrew's goal

BabyLM is the closest thing to a controlled experiment on Andrew's question, and its answer so far is limited. Three years of community effort produced transformers that learn grammar well from 10M to 100M words and beat much larger models on grammar tests, with architectural and objective changes as the winning ideas; curricula won nothing. I found no model trained on 100M words that is useful as a general language model. The gap that remains is knowledge and reasoning. My reading is that "learning from less text" will need another source of structure (perception, interaction, simulation or built-in compositional priors); three years of better ways to read the same text have not closed that gap. The interaction track (learning from a teacher's feedback) and the GPT-BERT result (two objectives on the same data beat one) are the most promising openings.

## 11. Test-time compute and in-context learning theory

### Core idea

A trained transformer learns from its prompt without any weight change. Two mechanistic accounts explain parts of this:

- **Induction heads** (Olsson et al., Anthropic, 2022): pairs of attention heads that implement [A][B] … [A] → [B], copying what followed an earlier occurrence of the current token. They appear at the same point in training as a sudden jump in in-context learning, visible as a bump in the loss curve. The evidence is causal for small attention-only models and correlational for large models with MLPs [90].
- **In-context learning as gradient descent** (von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov, Vladymyrov; ICML 2023): a single linear self-attention layer can be set so its output equals one step of gradient descent on a least-squares loss over the in-context examples, ΔW = −η Σ_i (W x_i − y_i) x_i^T. Self-attention-only transformers trained on regression tasks found weights matching this construction, and deeper models learned a curvature correction that beats plain gradient descent. The authors read induction heads as a special case [91].
- **The limit of that account.** Shen, Mishra and Khashabi (ICML 2024 position paper) tested the claim on LLaMA-7B trained on natural data and found that in-context learning and gradient descent respond differently to demonstration order and change output distributions differently; "the equivalence between ICL and GD remains an open hypothesis" [92].
- **Test-time compute.** Snell et al. (Google DeepMind and Berkeley, August 2024) found that allocating inference compute adaptively per prompt (search against a process reward model, or iterative revision) was more than 4x more efficient than best-of-N, and in a FLOPs-matched comparison a small model with test-time compute beat a 14x larger model on problems where the small model already had some success. On the hardest problems, extra pretraining was still better [93]. Reasoning models since late 2024 (o-series, Deep Think) apply this at scale; o3's 87.5% on ARC-AGI-1 cost 172x the compute of its 75.7% run [52].

### What this means for any replacement

The delta rule S_t = S_{t-1} + β_t (v_t − S_{t-1} k_t) k_t^T (Section 1) is exactly one step of online gradient descent on the loss ½||S k_t − v_t||². Titans makes the same step with momentum and weight decay [24]. The gradient-descent reading of attention and the delta-rule recurrences are therefore one idea: the sequence layer is a learner that fits a key-to-value map to the context. Attention fits it exactly by keeping every pair (a nonparametric learner with memory linear in context); recurrent layers fit it approximately in a fixed-size state (a parametric learner). The recall results in Section 1 measure the cost of that approximation.

Any replacement for the transformer must therefore provide, inside its forward pass:

1. a content-addressable store of the context with exact recall when needed (induction heads, associative recall [15], [90]);
2. a fast inner learner that adapts to the examples in the prompt (in-context learning [91]);
3. a way to spend more computation on harder inputs (test-time compute [93]).

### Status in 2026

Mechanistic interpretability of in-context learning is an active field at Anthropic, Google DeepMind and in academia. Test-time compute is adopted in every frontier reasoning model.

### Relevance to Andrew's goal

In-context learning is already few-shot learning, and it is the transformer's most data-efficient behavior: a pretrained model learns a new task from a handful of examples. The expensive part is the pretraining that produces a good inner learner. This reframes Andrew's goal as two separate targets: make the outer learning (pretraining) need less text, and keep or improve the inner learner. MLC (Section 7) and the in-context regression studies show that the inner learner can be shaped by what the outer training distribution contains. Test-time compute trades inference cost for model size, which works against Andrew's compute goal at inference but helps it at training.

## 12. Other alternatives with traction

### Kolmogorov-Arnold Networks (KAN)

- **Idea.** The Kolmogorov-Arnold theorem says any continuous multivariate function is a sum of compositions of univariate functions, f(x) = Σ_q Φ_q(Σ_p φ_{q,p}(x_p)). KANs put a learnable spline on every edge instead of a fixed activation on every node, with no linear weights (Liu, Wang, Vaidya, Ruehle, Halverson, Soljačić, Hou, Tegmark; April 2024) [94].
- **Results.** On small AI-for-science tasks, smaller KANs matched or beat larger MLPs at function fitting and PDE solving, with faster empirical scaling (test RMSE roughly ∝ G^-3 in grid size for a toy case, against a theoretical G^-4) and readable learned functions used to rediscover formulas in mathematics and physics [94].
- **Failure.** A parameter- and FLOP-matched comparison (Yu, Yu and Wang, July 2024) found KAN beat MLP only on symbolic formula fitting; MLP won on machine learning, vision, NLP and audio. Giving an MLP B-spline activations erased KAN's formula advantage, and KAN forgot more in continual learning [95]. Splines are also slower per FLOP on GPUs.
- **Status.** Niche, in scientific regression and interpretability. No language model of note uses KAN layers.
- **Relevance.** Low for language. KAN's interpretable univariate pieces suit problems that have a compact formula; text does not appear to be one.

### Liquid neural networks

- **Idea.** Liquid time-constant networks (Hasani, Lechner, Amini, Rus, Grosu; AAAI 2021) are continuous-time RNNs whose time constant depends on the input: dx/dt = −[1/τ + f(x, I; θ)] x + f(x, I; θ) A. They are stable, bounded and expressive on time series [96].
- **What shipped.** Liquid AI's LFM2 (July 2025; technical report November 2025) is far from the original LTC. A hardware-in-the-loop search over SSMs, linear attention, liquid-S4 and CfC blocks chose gated short convolutions for most layers plus a minority of grouped-query attention layers (10 convolution and 6 attention blocks in the 16-block design). LFM2 reports 2x faster decode and prefill than Qwen3 on CPU [97], [98]. The report states that adding linear-attention or state-space operators "does not improve aggregate quality" under on-device budgets once a few attention layers handle long-range retrieval [98].
- **Status.** Adopted for edge models by one company. The liquid ODE idea survives as branding and search-space options more than in the shipped layers.
- **Relevance.** A compute result for small on-device models, and another independent finding that a few attention layers are needed for retrieval.

### Spiking and neuromorphic language models

- **Idea.** Neurons communicate with binary, event-driven spikes, so hardware only spends energy when a spike fires.
- **Results.** SpikeGPT (Zhu, Zhao, Eshraghian; 2023) trained 45M to 260M-parameter spiking language models with an RWKV-style linear attention, reporting competitive results with about 5x less energy on neuromorphic hardware [99]. SpikingBrain (BICLab, Chinese Academy of Sciences; September 2025) converted Qwen2.5-7B into a 7B linear model and a 76B hybrid-linear MoE with adaptive spiking neurons, trained on MetaX GPUs with about 150B tokens of continual pretraining (the authors say about 2% of typical pretraining data), reaching performance comparable to open transformer baselines, over 100x faster time-to-first-token at 4M tokens and 69% activation sparsity [100].
- **Caveat.** SpikingBrain's data efficiency is inherited: it starts from a Qwen2.5 checkpoint that was pretrained on trillions of tokens, so the "2% of the data" figure measures the cost of converting an already trained model [100]. Its long-context speed comes from linear attention, the Section 1 mechanism.
- **Status.** Research and national-hardware programs; no spiking LM competes at the frontier.
- **Relevance.** Energy per inference on special hardware. No evidence for learning language from less text.

### Diffusion language models

- **Idea.** Instead of generating left to right, start from a fully masked sequence and iteratively unmask tokens in parallel, training on a masked-token likelihood bound. The model sees both directions.
- **Results.** LLaDA (Nie et al., Renmin University and Ant Group; NeurIPS 2025) trained an 8B masked diffusion model from scratch on 2.3T tokens with 0.13M H800 GPU hours. It beat LLaMA2 7B on nearly all of 15 tasks and was on par with LLaMA3 8B (trained on 15T tokens) in in-context learning; it removed the "reversal curse", beating GPT-4o on reversing Chinese poem lines [101]. Inception Labs' Mercury Coder reached 1,109 (Mini) and 737 (Small) tokens per second on H100s, measured by Artificial Analysis, up to 10x faster than speed-optimized frontier models at comparable coding quality [102]. In BabyLM 2025 a diffusion masked LM won the Strict-track NLP score [88], and the ARChitects' second place in ARC Prize 2025 used a masked-diffusion model [53].
- **Drawbacks.** LLaDA cannot use a KV cache, so each denoising step reprocesses the sequence [101]. Quality at the frontier scale is not established in public.
- **Status.** Adopted commercially for fast generation (Inception Labs), and Google DeepMind released an experimental Gemini Diffusion demo on 20 May 2025, which it said generated text "significantly faster than our fastest model so far, while matching its coding performance" [105].
- **Relevance.** A data-efficiency signal worth noting: LLaDA matched LLaMA3 8B in-context performance with 2.3T tokens against 15T, though the comparison mixes data quality and other differences [101]. Bidirectional training also removes a failure (the reversal curse) that comes from left-to-right factorization.

### Large Concept Models (Meta, December 2024)

- **Idea.** Treat a sentence as a "concept" and model language as a sequence of sentence embeddings. Text is split into sentences, each encoded by the frozen SONAR encoder (200 languages for text, 76 for speech input); a transformer predicts the next sentence embedding; SONAR decodes it back to text. The authors tried MSE regression, two diffusion variants and quantized SONAR codes [103].
- **Results.** Ablations at 1.6B parameters on about 1.3T tokens; one diffusion ("Two-Tower") design scaled to 7B parameters on 2.7T tokens in the arXiv version (Meta's publication page says 7.7T). The 7B model was evaluated on summarization and summary expansion and showed zero-shot generalization to many languages, beating same-size LLMs on the multilingual comparison [103]. MSE regression failed in a revealing way: when many next sentences are plausible, it predicts their average, which "may not correspond to any relevant point in SONAR space"; diffusion did better [103].
- **Drawbacks.** The paper lists fragile SONAR embeddings and imperfect local geometry as limitations. Much of the cross-lingual transfer comes from SONAR's alignment, which the authors acknowledge, and the multilingual comparison uses one baseline family [103], [104]. Diffusion in continuous space struggles on math and code, where exact tokens matter [104].
- **Status.** Research release with open code; I found no production successor.
- **Relevance.** This is the most direct test of "ideas have a direct representation" at scale, and the result is mixed. Operating on sentence meanings works well enough to summarize and to transfer across languages for free, which supports the hypothesis. It did not learn from less data (2.7T tokens) or beat token models on general tasks, and the averaging failure shows that a point in a fixed embedding space is a poor representation of an idea that has many valid expressions. A distribution or region per concept, or a learned concept space trained jointly (JEPA-style, Section 3), would be the next step.

## 13. Synthesis

### What to learn from the transformer

The transformer won on a combination of properties, and every serious replacement since 2017 has had to keep most of them or add them back.

1. **Parallel training over the sequence.** RNNs lost to attention because they trained one step at a time. Every successful efficient model (S4, Mamba-2, RetNet, RWKV, DeltaNet) has a parallel or chunked training form, usually by writing the recurrence as a structured matrix product [1], [3], [5], [7]. A replacement that needs sequential training will not be trained at scale.
2. **Content-addressable recall over the whole context.** This is the property cheaper models most often lose. Zoology traced 82% of the gap between gated convolutions and attention to associative-recall tokens [15]; Jelassi et al. proved fixed-state models cannot copy long strings [16]; MiniMax, NVIDIA, Qwen, AI21 and Liquid AI all kept some attention layers for retrieval [9], [10], [13], [17], [98]; MiniMax went back to full attention after finding multi-hop reasoning deficits at scale [14]. Attention is a modern Hopfield memory with exponential capacity [21].
3. **In-context learning.** The forward pass contains a learner: induction heads copy patterns [90], and at least in regression settings attention implements gradient descent on the prompt [91]. The delta-rule recurrences make this explicit as online regression (Section 11). A replacement must provide a fast inner learner, and its quality depends on exact recall (point 2).
4. **A generic objective that scales.** Next-token prediction needs no labels and keeps improving with data and compute along predictable curves [83]. Alternatives with better objectives on paper (JEPA, energy-based learning, local learning rules) have struggled with collapse, stability or depth until recently [30], [44].
5. **Hardware fit.** Dense matrix multiplies on GPUs. Linear attention's memory-bound kernels, spiking networks and equilibrium methods all lose much of their theoretical advantage on current hardware [14], [43].
6. **Variable computation at inference.** Not native to the architecture, but added through chain-of-thought and search, and now central to frontier results [52], [93].

The weaknesses to fix are equally specific: quadratic cost in context length, a KV cache that grows with context, autoregressive left-to-right factorization (the reversal curse, which LLaDA's bidirectional training removed [101]), and a pretraining appetite of 15 to 36 trillion tokens [84], [85].

### Which directions attack data efficiency, and which only attack compute

| Direction | Attacks compute | Attacks data | Evidence for data efficiency in language |
|---|---|---|---|
| SSMs, linear attention, delta-rule hybrids (Sec. 1) | Yes: linear time, constant-memory inference | No | None; same token budgets and scaling regime [13], [14] |
| Retrieval and memory (Sec. 2) | Partly: fewer parameters for the same perplexity | Partly: knowledge moves to a store | kNN-LM beat a model trained on 30x more tokens [18]; gains concentrate on facts [22] |
| JEPA and world models (Sec. 3) | Yes in vision (I-JEPA 10x cheaper than MAE) | Yes downstream (label efficiency) | Minimal: LLM-JEPA on paired-view fine-tuning tasks [29] |
| Local learning rules (Sec. 4) | Only on special hardware | No | None |
| Active inference (Sec. 5) | Yes (no gradient descent in AXIOM) | Yes, in games with built-in object priors | None |
| Program synthesis, neurosymbolic (Sec. 6) | Yes: 7M-parameter TRM | Yes, few-shot by construction | Only in formal domains (ARC, geometry, Lean) |
| Meta-learned compositionality (Sec. 7) | Small models | Yes, systematic few-shot generalization | Artificial languages; human-level on the tested task [65] |
| Category theory, DisCoCat (Sec. 8) | Unclear (quantum hardware) | Yes in principle (compositional generalization) | Toy datasets only [67], [68] |
| HDC/VSA (Sec. 9) | Yes: cheap, noise-tolerant hardware | Few-shot in classification | None for open text |
| BabyLM methods (Sec. 10) | No; often many epochs | Yes, the explicit target | Grammar from 10M to 100M words; knowledge and reasoning lag [81], [88] |
| Test-time compute (Sec. 11) | Trades inference for parameters | No | Not applicable |
| KAN, liquid, spiking (Sec. 12) | Yes, in niches | No | None |
| Diffusion LMs (Sec. 12) | Yes, parallel decoding | Possibly | LLaDA near LLaMA3 8B with 2.3T vs 15T tokens, confounded [101] |
| Large Concept Models (Sec. 12) | Shorter sequences of concepts | Not shown | Trained on 2.7T tokens [103] |

Read across the sections, the evidence divides in two. The compute-only directions are mature and in production. The data-efficiency directions work where the target has formal structure that can be built in or generated (grids, geometry, proofs, artificial languages, physical objects) and have not yet worked on open-ended English. The one shared ingredient in the data-efficient successes is a structural prior that the learner did not have to discover from text: symmetry in geometric deep learning, objects in AXIOM, a DSL in program synthesis, grammar types in DisCoCat, or a distribution of generated compositional tasks in MLC and NVARC.

### Where new mathematics might plausibly come from (opinion)

Everything in this subsection is my judgment, drawn from the evidence above.

1. **Learn the algebra, do not hand it in.** DisCoCat has the right shape for Andrew's hypothesis (meaning as a functor from a grammar category to vector spaces [74]) and the wrong source for its parts (a hand-built grammar and parser). DreamCoder learns a library of abstractions by compressing solutions [49]. A model that learns the category itself, its objects (types) and morphisms (composition operations), by compression over text, with composition enforced as a structural law, joins these two lines. The mathematical question is what family of categories is rich enough for language and still learnable from little data. Categorical deep learning [69] supplies a vocabulary; nobody has supplied the learning algorithm.

2. **Make the inductive bias a distribution over languages.** MLC [65] and the ARC winners [53] show that a standard network becomes a systematic few-shot learner when trained on many generated tasks whose structure varies. The outer data then teaches how to learn, and the inner learner (in-context learning) does the few-shot work. The open mathematics is the design of the task generator: a prior over grammars and meanings that is broad enough that English is a typical sample, so that real text is needed only to identify which language this is. This is the most practical route to "far less text" that current evidence supports.

3. **Represent an idea as a region or a distribution.** LCM's regression failure, averaging plausible next sentences into a point that decodes to nothing [103], and JEPA's collapse problem [25], [30] are the same problem: a single vector cannot stand for an idea with many valid expressions. LeJEPA's isotropic-Gaussian result is a theorem about the right embedding distribution [30]. Energy functions (Hopfield [21], JEPA [25]) and VSA superposition [75] give two ways to represent sets of meanings in one object. A concept space where an idea is a basin of an energy function, and composition is binding, would combine Sections 2, 3 and 9.

4. **Keep exact recall, compress everything else.** The efficient-model literature converged on a split: a compressed state that learns by online regression (delta rule, Titans) for most layers, plus a small amount of exact content-addressed lookup [8], [13], [24]. A data-efficient model will still need that lookup, because in-context learning depends on it. The mathematics of the compressed part (structured matrices, semiseparable decompositions, Householder products [3], [7]) is ready to reuse.

5. **Two views of the same meaning are the training signal.** The JEPA question for language (what to predict, in what space) and the LLM-JEPA requirement for paired views [29] point to a concrete experiment: train on aligned pairs (paraphrase, translation, text and code, text and perception) with a representation-space objective, and measure how much raw text is then needed. GPT-BERT's finding that two objectives on the same 100M words beat either alone [87] is weak evidence in the same direction.

A caution applies to all five. Rich Sutton's "bitter lesson" (2019) is that "general methods that leverage computation are ultimately the most effective", and that building in "how we think we think does not work in the long run" [106]; he names symmetries and objects among the things not to build in. The pattern recurs in this report: AlphaProof's formal pipeline was matched a year later by a plain language model [62]; MiniMax's linear hybrid lost to full attention at scale [14]; structured priors (KAN, HDC) lost to generic networks on general benchmarks [78], [95]. Sutton's prescription, build in "only the meta-methods that can find and capture this arbitrary complexity" [106], fits options 1 and 2 above (learn the algebra, learn the prior over languages) better than any fixed hand-written structure. A new mathematical basis has to show that it keeps improving with scale, or it will be overtaken by the next larger transformer. The test that would separate a real advance from a clever prior is a learning curve: performance against training tokens, compared with a transformer at matched compute, extending past 100M words.

## 14. References

All links were retrieved by live search in October 2026. Entries marked "secondary" or "third-party" are not primary sources and are used only where noted in the text.

1. Gu, Goel, Ré. Efficiently Modeling Long Sequences with Structured State Spaces (S4). 2021. https://arxiv.org/abs/2111.00396
2. Gu, Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. 2023. https://arxiv.org/abs/2312.00752
3. Dao, Gu. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2). ICML 2024. https://arxiv.org/abs/2405.21060
4. Katharopoulos, Vyas, Pappas, Fleuret. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020. https://proceedings.mlr.press/v119/katharopoulos20a.html
5. Sun, Dong, Huang, Ma, Xia, Xue, Wang, Wei. Retentive Network: A Successor to Transformer for Large Language Models. 2023. https://arxiv.org/abs/2307.08621
6. Peng et al. RWKV: Reinventing RNNs for the Transformer Era. 2023. https://arxiv.org/abs/2305.13048
7. Yang, Wang, Zhang, Shen, Kim. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 2024. https://proceedings.nips.cc/paper_files/paper/2024/file/d13a3eae72366e61dfdc7eea82eeb685-Paper-Conference.pdf
8. Yang, Kautz, Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. 2024. https://arxiv.org/abs/2412.06464
9. Lieber, Lenz et al. (AI21). Jamba: A Hybrid Transformer-Mamba Language Model. 2024. https://arxiv.org/abs/2403.19887
10. Qwen team. Qwen3-Next-80B-A3B-Instruct model card. 2025. https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
11. Qwen team. Qwen3.8-Flash-Next model card and repository. August 2026. https://huggingface.co/Qwen/Qwen3.8-Flash-Next and https://github.com/qwenlm/qwen3.8-flash-next
12. Alibaba Cloud Community. Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost-Efficiency. 27 August 2026. https://www.alibabacloud.com/blog/qwen-3-8-flash-next-a-new-architecture-towards-ultimate-cost-efficiency_603501
13. MiniMax. MiniMax-01: Scaling Foundation Models with Lightning Attention. 2025. https://arxiv.org/abs/2501.08313
14. MiniMax (M2 pretraining lead). Why Did M2 End Up as a Full Attention Model? 29 October 2025. https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model
15. Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré. Zoology: Measuring and Improving Recall in Efficient Language Models. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/file/448fc91f669c15d10364ee01d512cc10-Paper-Conference.pdf
16. Jelassi, Brandfonbrener, Kakade, Malach. Repeat After Me: Transformers are Better than State Space Models at Copying. ICML 2024. https://proceedings.mlr.press/v235/jelassi24a.html
17. Waleffe et al. (NVIDIA). An Empirical Study of Mamba-based Language Models. 2024. https://research.nvidia.com/publication/2024-06_empirical-study-mamba-based-language-models
18. Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis. Generalization through Memorization: Nearest Neighbor Language Models. ICLR 2020. https://arxiv.org/abs/1911.00172
19. Borgeaud et al. (DeepMind). Improving Language Models by Retrieving from Trillions of Tokens (RETRO). ICML 2022. https://proceedings.mlr.press/v162/borgeaud22a.html
20. Wu, Rabe, Hutchins, Szegedy. Memorizing Transformers. ICLR 2022. https://research.google/pubs/memorizing-transformers/
21. Ramsauer et al. Hopfield Networks is All You Need. 2020 (ICLR 2021). https://arxiv.org/abs/2008.02217
22. Wang, Ping et al. (NVIDIA). Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.482/
23. Berges et al. (Meta). Memory Layers at Scale. December 2024. https://ai.meta.com/research/publications/memory-layers-at-scale/
24. Behrouz, Zhong, Mirrokni (Google). Titans: Learning to Memorize at Test Time. 2025. https://arxiv.org/abs/2501.00663
25. LeCun. A Path Towards Autonomous Machine Intelligence. 2022. https://openreview.net/pdf?id=BZ5a1r-kVsf
26. Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA). CVPR 2023. https://arxiv.org/abs/2301.08243
27. Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas. Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA). 2024. https://arxiv.org/abs/2404.08471
28. Assran et al. (Meta). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. 2025. https://arxiv.org/abs/2506.09985
29. Huang, LeCun, Balestriero. LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures. ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac818ec2da1c976b267de32f59709c69-Abstract-Conference.html
30. Balestriero, LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. November 2025. https://arxiv.org/abs/2511.08544
31. Maes, Le Lidec, Scieur, LeCun, Balestriero. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. March 2026. https://arxiv.org/abs/2603.19312
32. Zhang, Terver, Rabbat, LeCun, Balestriero. H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning. arXiv 2610.06805, October 2026. https://h-jepa.com/ and https://github.com/kevinghst/H-JEPA
33. Meta FAIR (correspondence Zholus, Bohg). RoboJEPA: Scaling Robotic Latent World Models. October 2026. https://arxiv.org/html/2610.10515
34. LeCun. Objective-Driven AI (slides, Harvard CMSA, 28 March 2024). https://cmsa.fas.harvard.edu/media/lecun-20240328-harvard_reduced.pdf
35. LeCun. LinkedIn post on autoregressive LLMs as exponentially diverging processes, 27 March 2023. https://www.linkedin.com/posts/yann-lecun_i-have-claimed-that-auto-regressive-llms-activity-7045908925660950528-hJGk
36. Zeff. Yann LeCun Raises $1 Billion to Build AI That Understands the Physical World. WIRED, 10 March 2026 (secondary). https://www.wired.com/story/yann-lecun-raises-dollar1-billion-to-build-ai-that-understands-the-physical-world/
37. LeCun. Unveiling our new startup Advanced Machine Intelligence (AMI Labs). LinkedIn, 10 March 2026. https://www.linkedin.com/posts/yann-lecun_unveiling-our-new-startup-advanced-machine-activity-7437016101080883200-0Mr9
38. Abdullahi. Yann LeCun, Meta's Former AI Chief, Launches $1B Startup Focused on 'World Models'. eWeek, 10 March 2026 (secondary). https://www.eweek.com/news/meta-ai-yann-lecun-ami-world-models-startup/
39. LeCun interview at AIM 2026, transcript (secondary). October 2026. https://ceointerviews.ai/interview/2961732/
40. Whittington, Bogacz. An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity. Neural Computation, 2017. Verified via index entry https://exa.ai/library/publication/1q76nrfw1js (publisher DOI 10.1162/NECO_a_00949 not fetched)
41. Scellier, Bengio. Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation. Frontiers in Computational Neuroscience, 2017. https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full
42. Hinton. The Forward-Forward Algorithm: Some Preliminary Investigations. 2022. https://arxiv.org/abs/2212.13345
43. Laborieux, Ernoult, Scellier, Bengio, Grollier. Scaling Equilibrium Propagation to Deep ConvNets by Drastically Reducing Its Gradient Estimator Bias. Frontiers in Neuroscience, 2021. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.633674/full
44. Pinchetti et al. Benchmarking Predictive Coding Networks, Made Simple. 2024. https://arxiv.org/abs/2407.01163
45. Friston, Da Costa, Sajid, Heins, Ueltzhöffer, Pavliotis, Parr. The free energy principle made simpler but not too simple. Physics Reports, 2023. https://www.sciencedirect.com/science/article/pii/S037015732300203X
46. Isomura, Kotani, Jimbo, Friston. Experimental validation of the free-energy principle with in vitro neural networks. Nature Communications, 2023. https://www.nature.com/articles/s41467-023-40141-z
47. Heins et al. (VERSES AI). AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models. 2025. https://arxiv.org/abs/2505.24784
48. Bruineberg, Dołęga, Dewhurst, Baltieri. The Emperor's New Markov Blankets. Behavioral and Brain Sciences 45, 2022. https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/emperors-new-markov-blankets/715C589A73DDF861DCF8997271DE0B8C
49. Ellis et al. DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning. PLDI 2021. https://dl.acm.org/doi/10.1145/3453483.3454080
50. Chollet. On the Measure of Intelligence. 2019. https://arxiv.org/abs/1911.01547
51. Chollet, Knoop, Kamradt, Landers. ARC Prize 2024: Technical Report. https://arxiv.org/abs/2412.04604
52. Chollet. OpenAI o3 Breakthrough High Score on ARC-AGI-Pub. ARC Prize blog, 20 December 2024. https://arcprize.org/blog/oai-o3-pub-breakthrough
53. Knoop. ARC Prize 2025 Results and Analysis. ARC Prize blog, 5 December 2025. https://arcprize.org/blog/arc-prize-2025-results-analysis
54. Chollet et al. ARC Prize 2025: Technical Report. January 2026. https://arxiv.org/abs/2601.10904
55. Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny Networks. October 2025. https://arxiv.org/abs/2510.04871
56. ARC Prize. ARC Prize 2026 competition page. https://arcprize.org/competitions/2026
57. Kamradt. Announcing ARC-AGI-3. ARC Prize blog, 25 March 2026. https://arcprize.org/blog/arc-agi-3-launch
58. ARC Explainer (third-party tracker). ARC-AGI-3 leaderboard snapshot, 11 October 2026. https://arc.markbarney.net/arc-leaderboards
59. LLMLearner (third-party tracker). ARC-AGI-3 model rankings, October 2026. https://llmlearner.com/rankings/arc-agi-3
60. Trinh, Wu, Le, He, Luong. Solving olympiad geometry without human demonstrations (AlphaGeometry). Nature, 2024. https://www.nature.com/articles/s41586-023-06747-5
61. Hubert et al. (Google DeepMind). Olympiad-level formal mathematical reasoning with reinforcement learning (AlphaProof). Nature, November 2025. https://www.nature.com/articles/s41586-025-09833-y
62. Luong, Lockhart. Advanced version of Gemini with Deep Think officially achieves gold-medal standard at the International Mathematical Olympiad. Google DeepMind blog, 21 July 2025. https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/
63. Lake, Baroni. Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks (SCAN). ICML 2018. https://proceedings.mlr.press/v80/lake18a.html
64. Kim, Linzen. COGS: A Compositional Generalization Challenge Based on Semantic Interpretation. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.731/
65. Lake, Baroni. Human-like systematic generalization through a meta-learning neural network. Nature, 2023. https://www.nature.com/articles/s41586-023-06668-3
66. Kartsaklis et al. lambeq: An Efficient High-Level Python Library for Quantum NLP. 2021. https://arxiv.org/abs/2110.04236
67. Lorenz, Pearson, Meichanetzidis, Kartsaklis, Coecke. QNLP in Practice: Running Compositional Models of Meaning on a Quantum Computer. JAIR, 2023. https://www.jair.org/index.php/jair/article/view/14329
68. Duneau, Bruhn, Matos, Laakkonen, Saiti, Pearson, Meichanetzidis, Coecke. Scalable and interpretable quantum natural language processing: an implementation on trapped ions. 2024. https://arxiv.org/abs/2409.08777 and https://neurips.cc/virtual/2024/107076
69. Gavranović, Lessard, Dudzik, von Glehn, Araújo, Veličković. Position: Categorical Deep Learning is an Algebraic Theory of All Architectures. ICML 2024. https://proceedings.mlr.press/v235/gavranovic24a.html
70. Bronstein, Bruna, Cohen, Veličković. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. 2021. https://arxiv.org/abs/2104.13478
71. Laakkonen, Meichanetzidis, Coecke. Quantum Algorithms for Compositional Text Processing. 2024. https://arxiv.org/abs/2408.06061
72. Tong. Symbolica raises $31 mln to develop AI systems to compete with OpenAI. Reuters, 9 April 2024 (secondary). https://www.reuters.com/technology/symbolica-raises-31-mln-develop-ai-systems-compete-with-openai-2024-04-09/
73. Symbolica AI. "Today we set a new ARC-AGI-2 public eval SOTA: 85.28%." LinkedIn, 12 February 2026 (vendor claim). https://www.linkedin.com/posts/symbolica-ai_today-we-set-a-new-arc-agi-2-public-eval-activity-7427546074229137408-28WU
74. Coecke, Sadrzadeh, Clark. Mathematical Foundations for a Compositional Distributional Model of Meaning. 2010. https://arxiv.org/abs/1003.4394
75. Kleyko, Rachkovskij, Osipov, Rahimi. A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I. ACM Computing Surveys, 2022. https://dl.acm.org/doi/10.1145/3538531
76. Karunaratne, Le Gallo, Cherubini, Benini, Rahimi, Sebastian. In-memory hyperdimensional computing. Nature Electronics, 2020. https://www.nature.com/articles/s41928-020-0410-3
77. Hersche, Zeqiri, Benini, Sebastian, Rahimi. A neuro-vector-symbolic architecture for solving Raven's progressive matrices. Nature Machine Intelligence, 2023. https://www.nature.com/articles/s42256-023-00630-8
78. Vergés, Heddes, Nunes, Kleyko, Givargis. Classification using hyperdimensional computing: a review with comparative analysis. Artificial Intelligence Review, 2025. https://link.springer.com/article/10.1007/s10462-025-11181-2
79. Carzaniga, Hersche, Schindler, Rahimi. Practical Lessons on Vector-Symbolic Architectures in Deep Learning-Inspired Environments. NeSy 2025. https://proceedings.mlr.press/v284/carzaniga25a.html
80. Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures. OpenReview submission, 2025 (authors not recorded in the retrieved copy). https://openreview.net/pdf?id=WxM7lIoGBb
81. Warstadt et al. Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora. CoNLL 2023. https://aclanthology.org/2023.conll-babylm.1.pdf
82. Hu et al. Findings of the Second BabyLM Challenge. CoNLL 2024. https://aclanthology.org/2024.conll-babylm.1/
83. Hoffmann et al. (DeepMind). Training Compute-Optimal Large Language Models (Chinchilla). NeurIPS 2022. https://proceedings.neurips.cc/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf
84. Llama Team (Meta). The Llama 3 Herd of Models. 2024. https://arxiv.org/abs/2407.21783
85. Qwen Team. Qwen3 Technical Report. 2025. https://arxiv.org/abs/2505.09388
86. Muennighoff et al. Scaling Data-Constrained Language Models. JMLR 26, 2025. https://www.jmlr.org/beta/papers/v26/24-1000.html
87. Charpentier, Samuel. GPT or BERT: why not both? BabyLM 2024. https://aclanthology.org/2024.conll-babylm.24/
88. Charpentier, Choshen, Cotterell, Gul, Hu et al. Findings of the Third BabyLM Challenge: Accelerating Language Modeling Research with Cognitively Plausible Data. 2025. https://aclanthology.org/2025.babylm-main.28/
89. Eldan, Li. TinyStories: How Small Can Language Models Be and Still Speak Coherent English? 2023. https://arxiv.org/abs/2305.07759
90. Olsson et al. (Anthropic). In-context Learning and Induction Heads. 2022. https://arxiv.org/abs/2209.11895
91. von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov, Vladymyrov. Transformers Learn In-Context by Gradient Descent. ICML 2023. https://proceedings.mlr.press/v202/von-oswald23a.html
92. Shen, Mishra, Khashabi. Position: Do pretrained Transformers Learn In-Context by Gradient Descent? ICML 2024. https://proceedings.mlr.press/v235/shen24d.html
93. Snell, Lee, Xu, Kumar. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. 2024. https://arxiv.org/abs/2408.03314
94. Liu, Wang, Vaidya, Ruehle, Halverson, Soljačić, Hou, Tegmark. KAN: Kolmogorov-Arnold Networks. 2024. https://arxiv.org/abs/2404.19756
95. Yu, Yu, Wang. KAN or MLP: A Fairer Comparison. 2024. https://arxiv.org/abs/2407.16674
96. Hasani, Lechner, Amini, Rus, Grosu. Liquid Time-constant Networks. AAAI 2021. https://ojs.aaai.org/index.php/AAAI/article/view/16936
97. Liquid AI. Introducing LFM2: The Fastest On-Device Foundation Models on the Market. 10 July 2025. https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models
98. Liquid AI. LFM2 Technical Report. November 2025. https://arxiv.org/abs/2511.23404
99. Zhu, Zhao, Eshraghian. SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks. 2023. https://arxiv.org/abs/2302.13939
100. BICLab. SpikingBrain Technical Report: Spiking Brain-inspired Large Models. September 2025. https://arxiv.org/abs/2509.05276
101. Nie et al. Large Language Diffusion Models (LLaDA). NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/file/48b383b24230e0e6e649d9c98dae4d8c-Paper-Conference.pdf
102. Inception Labs. Mercury: Ultra-Fast Language Models Based on Diffusion. 2025. https://arxiv.org/abs/2506.17298
103. LCM team (Meta). Large Concept Models: Language Modeling in a Sentence Representation Space. December 2024. https://arxiv.org/abs/2412.08821 and https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/
104. Pith (third-party review). Review of Large Concept Models. https://pith.science/paper/2412.08821
105. Google. Gemini Diffusion: Google DeepMind's experimental research model. 20 May 2025. https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/
106. Sutton. The Bitter Lesson. 13 March 2019. Archived copy: https://web.archive.org/web/20251006184925/http:/www.incompleteideas.net/IncIdeas/BitterLesson.html

### Items not verified

- ARC-AGI-3 scores as of 11 October 2026 (top Kaggle 62.96; GPT-6 Astra 62.7) come from third-party trackers [58], [59]; I could not read them on arcprize.org.
- Whether Symbolica's February 2026 ARC-AGI-2 result [73] used category-theoretic models.
- The attribution of the "FEP is unfalsifiable" criticism to a specific primary source (Section 5).
- Qwen3.8-Flash-Next's training-cost and speedup figures are vendor claims without independent replication [11], [12].
- Author lists marked "et al." were not all checked in full; the first authors and venues were.
