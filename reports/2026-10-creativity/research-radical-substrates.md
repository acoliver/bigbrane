# Radical substrates, curricula and a child-scale experiment

Research note, October 2026. Citation markers like [12] refer to the numbered reference list at the end. Items marked UNVERIFIED were not confirmed against a primary source during this pass. Opinion is labelled as opinion.

## Summary

**Headline.** Nothing in the literature replaces vectors for learning language from child-scale data today, and the radical alternatives that work (sampling, programs, solvers) win at inference and composition on top of a small learned model. The test Andrew wants is feasible: two to four times a six-year-old's input lands on BabyLM's 100M-word track, which has published GPT-2 small and best-recipe baselines to beat.

- **The "1982" paper.** The finite-state argument is Chomsky 1956 ("Three models", IRE) and 1957 (*Syntactic Structures*), extended to learnability by Miller and Chomsky 1963. The real 1982 papers on this ladder are Pullum and Gazdar ("Natural languages and context-free languages") and Bresnan, Kaplan, Peters and Zaenen ("Cross-serial dependencies in Dutch"), both about context-free grammars. Content match to Chomsky: high. Date match: low.
- **Non-vector substrates.** Sampling explains human reasoning limits and SMC lets small LMs beat models 8x larger on constrained tasks; program induction learns visual concepts from one example; hyperdimensional codes identify languages cheaply; assembly calculus parses simple sentences with hand-built lexicons. None has a child-scale language benchmark result.
- **Units.** BPE beat character and phoneme units at BabyLM scale; speech-first learners are near chance on child-scale audio; byte patching (BLT) pays off at large scale.
- **Structure first.** Formal-language pre-pretraining saved 33% of tokens at 1B parameters; nesting or crossing dependencies transfer, a copy language hurts. No direct evidence for school mathematics first.
- **Hybrids.** Neural proposer plus exact checker works (PAL, Logic-LM, AlphaGeometry, FunSearch, SMC). The weak point is translation into the checker.
- **Child word budgets.** By age six: about 7M to 100M words, central 27M (LENA) to 39M (Hart and Risley working class). 2x: 14M to 200M (central 54M to 78M). 4x: 29M to 400M (central 108M to 156M). BabyLM 10M is the low end of one child; BabyLM 100M is about 4x the LENA mean.
- **Baselines.** At 100M words: GPT-2 small BLiMP 75.1, EWoK 51.2; best recipe GPT-BERT BLiMP 86.1. At 10M: GPT-2 small BLiMP 67.3; GPT-BERT 81.2. World knowledge stays near chance (max 58.4%).
- **Development.** Curricula, child-directed register and teacher feedback gave small or no gains; grounding taught concrete word meanings from 61 hours of one child's video.
- **Proposed experiment.** 50M and 100M words of the BabyLM corpus, 10-pass cap, GPT-2 small and GPT-BERT baselines against a hybrid (small LM translator plus probabilistic world model plus SMC), a structure-first arm, and the best non-vector proposal, evaluated on Zorro, BLiMP, EWoK, entity tracking, wug tests, a text fast-mapping set and a new child-question set. A win is matching GPT-2 small's 100M-word scores with 50M words at equal or lower FLOPs, or beating both baselines on child questions by 10 points.
- **Compute.** GPT-2 small at 1B words seen is about 10^18 FLOPs, about 115 days on 8 CPU threads at this machine's measured rate; measure GPU throughput before committing. The 10M pilot fits locally.

## 0. The "1982" finite-state paper

Andrew remembered a paper "from 1982" explaining why a hidden Markov or finite-state approach to language does not work. I found no well-known 1982 paper whose main claim is that. The argument he describes comes from Chomsky in the 1950s; 1982 is the year of two papers on the next rung up the hierarchy (context-free grammars), and those are the most likely source of the date.

**Candidates, ranked by how well they match the remembered content.**

1. **Chomsky (1956), "Three models for the description of language", IRE Transactions on Information Theory IT-2(3), 113 to 124** [1]. The abstract states that "no finite-state Markov process that produces symbols with transition from state to state can serve as an English grammar," and that n-order statistical approximations to English "do not come closer, with increasing n, to matching the output of an English grammar." The argument rests on nested dependencies (if...then, either...or) that can embed without bound. This is the original source of the claim. Match on content: high. Match on date: none.
2. **Chomsky (1957), *Syntactic Structures*, chapter 3** [2]. A book-length restatement of the 1956 argument for linguists, plus the "colorless green ideas sleep furiously" example aimed at statistical models. Same content match; date 1957.
3. **Miller and Chomsky (1963), "Finitary models of language users", Handbook of Mathematical Psychology vol. II, 419 to 491** [3]. Extends the argument from grammar to learning: a k-limited Markov model of English would have too many parameters for a child to estimate from the input available. This is the version closest to Andrew's concern with data efficiency. The specific parameter counts commonly attributed to this chapter are UNVERIFIED in this pass; I confirmed the chapter, its page range and its treatment of Markov sources, but not the numbers.
4. **Pullum and Gazdar (1982), "Natural languages and context-free languages", Linguistics and Philosophy 4, 471 to 504** [4]. A real 1982 paper on the formal power needed for natural language. It argues the other way: it reviews published proofs that natural languages are not context-free and finds them flawed (summary from the paper's well-known thesis; the full text was not fetched in this pass). It does not argue about finite-state models. If Andrew's memory merged "1982" with "formal-language-class arguments," this is the likely source of the year.
5. **Bresnan, Kaplan, Peters and Zaenen (1982), "Cross-serial dependencies in Dutch", Linguistic Inquiry 13(4), 613 to 635** [5]. Shows that Dutch cross-serial verb constructions cannot be given correct tree structures by any context-free grammar ("not strongly context free"), using a pumping lemma on tree automata. Also a 1982 argument that a weaker formalism fails, one rung above finite-state.
6. Later and related: **Shieber (1985)** gave the Swiss German argument that natural language is not even weakly context-free [6]. **Pereira (2000)** answered the 1957 "colorless green ideas" argument directly: a class-based aggregate bigram model (16 classes, trained by EM on newspaper text) assigns the grammatical order about 2 x 10^5 times the probability of the reversed order, so the claim that statistical models must treat both as equally remote depended on unsmoothed maximum-likelihood estimates [7].

**Confidence.** Content match to Chomsky 1956/1957: high (my estimate: about 80% that this is the argument he means). Year match: low. My best guess for the "1982" is a conflation with Pullum and Gazdar 1982 or Bresnan et al. 1982 (my estimate: combined about 40%), or a secondary source from the 1980s retelling Chomsky. If he recalls an author, the list above should settle it.

**Relevance.** The finite-state argument concerns unbounded embedding and exact recognition. It says nothing about whether a finite device can approximate the bounded-depth language humans produce; human center-embedding beyond two or three levels is rare and hard to process (a standard psycholinguistic observation, UNVERIFIED in this pass). Transformers with fixed context are finite devices in this sense, and the argument did not stop them. The stronger lesson for a data-efficient learner is Miller and Chomsky's: a model whose parameters grow with the number of distinct contexts cannot be estimated from child-scale input, so the learner needs structure (hierarchy, sharing, compositional reuse) that lets one observation inform many contexts. Pereira's reply shows the same thing from the other side: smoothing and class sharing are what made the statistical approach work.

## 1. Non-vector and sampling-based representations

The question here is whether something other than dense real vectors and matrix multiplication could carry linguistic knowledge. Each entry gives the idea, what it has shown, its cost, and what it means for us.

### 1.1 The sampling hypothesis in cognition

**Idea.** The brain does not store or compute probabilities; it draws samples from an implicit distribution and acts on a few of them.

**Evidence.** Vul, Goodman, Griffiths and Tenenbaum (2014) show that when samples cost time, deciding from one or a few samples maximizes long-run reward. For two-alternative choices with symmetric utility, one sample is optimal whenever the action-to-sample time ratio is below 10; even with 32 alternatives, few samples are optimal across a wide range of costs. A meta-analysis of probability-matching studies found people use more samples when stakes are higher (r(25) = .61) [8]. Sanborn and Chater (2016) argue that a sampler with finite samples produces the classic reasoning errors (conjunction fallacy, base-rate neglect, unpacking effects) and does best in context-rich problems, which is where people do well too [9].

**Drawbacks.** These are theories of inference at decision time. Neither paper says how the generative model being sampled is learned, and learning is our bottleneck. Sampling-based estimates converge at the Monte Carlo rate (error falls as one over the square root of the sample count), so precise probabilities are expensive.

**Relevance.** This argues for a learner that represents hypotheses (parses, meanings, referents) as a small population of discrete samples and accepts approximate answers. It does not by itself remove vectors: something must score a sample, and in every working system that score is a learned function.

### 1.2 Particle filters and sequential Monte Carlo for parsing and generation

**Levy, Reali and Griffiths (2009), NeurIPS.** Incremental parsing with a particle filter over PCFG parses. With about 20 particles the parser usually builds an interpretation up to the disambiguating word of a garden-path sentence and then fails, matching human difficulty; with 1,000 particles it approximates the exact posterior. The model gave the first rational account of the "digging-in" effect, confirmed in a human experiment [10]. Cost scales linearly with particles and sentence length, which is the human profile. Limitation: the grammar was supplied by hand, and the model did not learn it.

**Lew, Zhi-Xuan, Grand and Mansinghka (2023), SMC steering.** Language generation tasks are written as probabilistic programs that call an LLM; SMC replaces decoding. At a cost similar to beam search it handles infilling, syntactic constraints and "prompt intersection." The LLaMPPL library shares a key-value cache across particles to keep cost down [11]. The workshop paper reports preliminary results only.

**Loula et al. (2025), ICLR.** The follow-up evaluated SMC with grammar-constrained proposals, weight correction and expensive potentials on four domains (Python data science, text-to-SQL, goal inference, molecule synthesis). In three of four domains a small model with full SMC beat models over 8x larger, and SMC with 5 to 10x fewer particles beat methods that check constraints only at the end. Llama 3.1 70B with SMC beat Codex-002 (175B, code-tuned) [12].

**Relevance.** This is the strongest evidence that a sampling layer can buy capability per parameter. All of it is inference-time: the language knowledge still lives in a pretrained transformer. For a child-scale learner, SMC is a candidate for the reasoning and parsing layer on top of a small learned model, and for learning itself (SMC over grammars or programs) it is untested at our scale.

### 1.3 Probabilistic programming languages

**Idea.** Write the generative model as a program with random choices; a generic engine does inference. Church (Goodman et al. 2008) introduced this for cognitive models [25]; Gen (Cusumano-Towner et al. 2019, PLDI) added programmable inference and outperformed earlier systems "sometimes by multiple orders of magnitude" on tracking, pose and time-series structure [13]; Pyro (Bingham et al. 2019) built on PyTorch for deep probabilistic models [26]. The best-known data-efficiency result is Lake, Salakhutdinov and Tenenbaum (2015): handwritten characters represented as stroke programs learned from one example reached human-level one-shot classification and beat the deep learning models of the time [14].

**Drawbacks.** Inference cost grows with the hypothesis space; structure learning (inferring the program itself) is the hard case and is usually limited to small domains. Lake et al. built in strong priors about strokes; no comparable program-induction system learns a language's lexicon and grammar from raw child-scale text.

**Relevance.** This is the most developed alternative to vectors for data-efficient concept learning. Its weak point is open-vocabulary language; its strong point is the meaning and reasoning side, which is where BabyLM models fail (Section 7: EWoK near chance).

### 1.4 Neural sampling in the brain

Berkes, Orbán, Lengyel and Fiser (2011, Science) recorded visual cortex in awake ferrets across development. The distribution of spontaneous activity grew closer to the average activity evoked by natural scenes with age (Spearman rho = -0.70) and matched activity evoked by natural scenes better than activity evoked by gratings or noise. This is what a sampling-based internal model predicts if spontaneous activity samples the prior [15]. It is evidence that cortex may represent uncertainty by sampling. The data are from vision, no linguistic test exists, and the study does not specify a learning algorithm.

### 1.5 Boltzmann machines and energy-based models

**Idea.** Define an energy over whole configurations and learn by comparing data with model samples. Mnih and Hinton (2007) built a factored restricted Boltzmann machine language model: on a 10M-word set the temporal FRBM reached perplexity 127.3 against 116.2 for a Kneser-Ney 6-gram, and mixing it with the n-gram gave a 21% perplexity reduction over the best n-gram. The deterministic log-bilinear model from the same paper was faster to train and performed as well or better, and it is the line that led to modern neural LMs [16]. Deng et al. (2020) trained a sequence-level energy model on the residual of an autoregressive LM with noise-contrastive estimation; it lowered perplexity (for example CC-News test 14.89 to about 12.1 for the largest joint model) and humans preferred its generations [17].

**Drawbacks.** The partition function makes likelihood intractable; training needs sampling (contrastive divergence) or noise-contrastive tricks; Deng et al. only made it work as a residual on a normal LM.

**Relevance.** Energy-based scoring of whole utterances is a natural partner for SMC (the energy is the weight). At child scale the RBM results were roughly n-gram level on 10M words, which is far below a modern 10M-word BabyLM model.

### 1.6 Stochastic computing

**Idea.** Represent a number in [0,1] as the fraction of 1s in a random bit-stream; an AND gate multiplies two independent streams. Alaghi and Hayes (2013) survey the field: circuits are tiny and fault-tolerant, but precision costs bit-stream length exponentially (4 to 8 bits of precision means 16 to 256 bits), correlated streams give wrong answers (p AND p gives p where p squared is correct), and use has been limited to low-precision control, image processing and LDPC decoding [18].

**Relevance.** Stochastic computing is a hardware substrate for sampling and has no learning theory of its own. It would matter only if the learner tolerates 4 to 8 bits of precision, which low-precision neural network work suggests is plausible for inference. No language results found.

### 1.7 Hyperdimensional and sparse binary codes

**Idea.** Represent items as random vectors of about 10,000 dimensions (binary or bipolar). Binding is elementwise multiply, bundling is addition, sequence order is a fixed permutation. Learning is a single counting pass with no gradient.

**Evidence.** Joshi, Halseth and Kanerva (2016) encoded letter n-grams this way and identified the language of single Europarl sentences across 21 languages at 97.8% with tetragrams, training each language vector from about a million bytes in about 14.5 seconds on a laptop [19].

**Drawbacks.** This is still a vector substrate, with different algebra. Language identification is a bag-of-n-grams task; I found no hyperdimensional result that matches even a small neural LM on syntax or question answering. Capacity of a bundle degrades as more items are superposed.

**Relevance.** Cheap, one-pass, and exactly compositional for binding. A plausible substrate for a symbolic or episodic memory inside a hybrid; weak evidence as a full language learner.

### 1.8 Graph, hypergraph and categorical representations

**Idea.** Meaning as a structure built by typed composition. DisCoCat (Coecke, Sadrzadeh and Clark 2010) lifts pregroup grammar type reductions to tensor contractions, so sentence meaning is computed from word meanings along the parse, in a diagrammatic calculus [20]. lambeq turns sentences into circuits that run on quantum hardware [21]. The 2010 paper set up the framework and left "a practical implementation for future work"; its authors note a limitation that sentence meanings do not compose into text.

**Drawbacks.** Tensor orders grow with grammatical type (a transitive verb is an order-3 tensor), which is expensive; experiments have been small. I did not find a graph-rewriting or hypergraph-rewriting language learner with published results on a standard benchmark. Wolfram-style hypergraph rewriting has no language-learning evaluation that I could find.

**Relevance.** Strong on composition, weak on learning from raw text. Useful as a constraint on how a hybrid composes meanings, if the grammar can be learned.

### 1.9 Hebbian assemblies and cellular automata

**Assembly calculus.** Papadimitriou, Vempala, Mitropolsky, Collins and Maass (2020, PNAS) define operations on neural assemblies (project, associate, merge) realizable by random networks with Hebbian plasticity and inhibition, prove the system can in principle compute anything, and propose an architecture for syntax [22]. Mitropolsky, Collins and Papadimitriou (2021) built a parser in this model that parses "reasonably nontrivial" simple English sentences and a small Russian version [23]. The lexicon and grammar are hand-wired; no benchmark comparison exists. Cost per operation is a few rounds of sparse random firing, which is cheap on sparse hardware and slow when simulated densely.

**Cellular automata.** Zhang et al. (ICLR 2025) pretrained GPT-2 models on elementary cellular automata rules. Models trained on more complex rules (Wolfram classes III and IV) did better on downstream ARC-style reasoning and chess move prediction; the most chaotic rules hurt, which the authors call an "edge of chaos" sweet spot [24]. The learner was a transformer, so the result bears on what to pretrain on (see Section 3) and says nothing about substrate.

**Relevance.** Assemblies are the closest thing to a non-vector, non-backprop language substrate with formal guarantees. Today they parse, they do not learn language from data. A useful research proposal would be learning the lexicon and a small grammar inside the assembly model from CHILDES utterances, scored on Zorro (Section 6).

### 1.10 Section verdict

No substrate in this list has matched a small transformer on any child-scale language benchmark. The sampling and program-based approaches show the clearest data-efficiency wins, but only in narrow domains (handwritten characters, constrained generation) or at inference time on top of a pretrained LM. (Opinion.) Vectors are cheap to learn with gradients, which is a likely reason they won; the alternatives above are mostly cheaper to run or to reason with and harder to learn. The promising place for "new mathematics" is the inference and composition layer and the learning objective, with a small vector model kept as the perception and scoring component until a non-vector learner beats it on Zorro or BLiMP at 10M words.

## 2. Non-token units

### 2.1 Bytes and learned patches

**ByT5 (Xue et al. 2022, TACL).** A standard T5 transformer on raw UTF-8 bytes, with minimal changes, is competitive with token-level models less sensitive to noise, and better on spelling and pronunciation tasks; the cost is longer sequences, which the authors characterize in parameters, training FLOPs and inference speed [27].

**MegaByte (Yu et al. 2023, NeurIPS).** Fixed-size byte patches: a large global transformer over patches and a small local model within each patch. Self-attention becomes sub-quadratic and feedforward layers run per patch. A 1.5B MegaByte model generated 40% faster than a 350M standard transformer while reaching better perplexity at equal compute [28].

**Byte Latent Transformer (Pagnoni et al. 2024; ACL 2025).** Patch boundaries come from the entropy of a small byte LM's next-byte prediction, so predictable stretches become long patches and hard spots get more compute. First FLOP-controlled scaling study of byte models, up to 8B parameters and 4T bytes. BLT matched Llama 3 at equal training FLOPs while allowing up to about 50% fewer inference FLOPs (average patch of 8 bytes against 4.4 bytes per Llama 3 token). BPE models were better at small training budgets; BLT overtook them at about 2.5x to 3x the compute-optimal budget [29].

**Drawbacks and relevance.** These are vector models with a different input unit. The gains are noise tolerance, character-level skill and inference compute at large scale. At child scale the BabyLM 2024 tokenization studies point the other way: Goriely et al. found standard BPE scored highest on the BabyLM text benchmarks (character tokenization did not significantly lower BLiMP), with phoneme input helping only on phonological tasks [30, 45]. Entropy patching is still worth testing as a mathematical idea: a segmenter driven by predictive surprise is close to how chunking theories say children segment speech (Section 2.4).

### 2.2 Morphemes, syllables and phonemes

In BabyLM 2024, grapheme- and phoneme-aware tokenization "did not seem to help language models on the base BabyLM evaluation tasks" but helped tasks that need sound or spelling knowledge [45]. In Goriely et al.'s GPT-2 runs on the 100M BabyLM corpus, converting text to phonemes lowered BLiMP by 3.5%, GLUE by 1.5% and BLiMP Supplement by 11.3%; character tokenization did not significantly lower BLiMP; and character-level phoneme models set the best BabySLM scores reported to that date (syntactic 94.9, lexical 89.6) [30]. So phoneme input costs a few points on text benchmarks, and the large losses come from learning from raw audio (Section 2.3). I did not find a BabyLM result where morpheme segmentation beat BPE on BLiMP at equal data; treat any claim of that kind as UNVERIFIED.

### 2.3 Speech-first and textless learning

**GSLM (Lakhotia et al. 2021, TACL).** A discrete speech encoder (CPC, wav2vec 2.0 or HuBERT, quantized to 50 to 200 units), a transformer LM on the unit sequence, and a speech decoder, all without text or labels. Trained on 6,000 hours of LibriLight audiobooks, generated speech had "recognizable phonemes and words and locally acceptable syntax," and automatic metrics showed language-level quality lagged acoustic quality [31]. For scale: 6,000 hours of speech at a typical reading rate is roughly tens of millions of words (my estimate, not from the paper), which is in the child range.

**TWIST (Hassid et al. 2023).** Warm-starting a speech LM from a pretrained text LM beat cold-start speech LMs "across the board," trained on about 150,000 hours of speech [32]. Speech models borrow text-learned structure to work well.

**BabySLM (Lavechin et al. 2023, Interspeech).** The test closest to our question. A speech LM (STELA) trained on 1,024 hours of child-centered long-form recordings (SEEDLingS) showed no lexical or syntactic knowledge (chance level). Text LMs trained on transcripts of the same kind of input were "largely above chance." Trained on 1,024 hours of clean audiobooks, the same speech model improved strongly on the lexical task but reached only 52.8% on syntax [33].

**Relevance.** Learning from audio the way a child does is currently far worse than learning from transcripts at the same scale. A speech-first learner would spend most of its effort on acoustic invariance that text gives away for free. For a data-efficiency experiment, text transcripts of child-directed speech are the fair and feasible input; speech is a later step.

### 2.4 Sentences and concepts as units

**Large Concept Models (Meta, 2024).** Autoregressive prediction in the SONAR sentence-embedding space (frozen encoder and decoder, 200 languages). Variants: MSE regression, diffusion, quantized SONAR. 1.6B models on about 1.3T tokens, scaled to 7B (the arXiv abstract gives 2.7T training tokens; Meta's page says 7.7T; I could not resolve which). Diffusion variants did best; plain MSE regression averaged plausible continuations into points that decode poorly. The 7B model showed zero-shot multilingual summarization above same-size LLMs [34].

**Drawbacks.** Relies on a frozen encoder trained on vast parallel data; nothing about it is data-efficient. Sentence-level units hide syntax, which is what child-scale evaluations probe.

### 2.5 Chunking theories from cognition

Christiansen and Chater (2016, BBS), "The Now-or-Never bottleneck": memory for raw input is lost within moments, so the language system must "eagerly" recode input into chunks, pass them up a hierarchy of levels, and resolve ambiguity "Right-First-Time." Learning is learning to process: acquisition means learning which chunks to build, and constructions in grammar correspond to chunks [35].

**Relevance.** The paper states a design rule and gives no algorithm. The rule maps onto three concrete choices: (1) a learned, surprise-driven segmenter (BLT's entropy patching is an engineering version); (2) a small fixed working memory at each level, which argues against long-context attention as the main memory; (3) online, single-pass learning. It also predicts that a learner with a small number of hypotheses kept alive (a particle filter with few particles, Section 1.2) is a realistic model of human processing limits.

### 2.6 Section verdict

The token unit is a minor factor at child scale: BPE still won the BabyLM tokenization comparisons, and speech-first learners are near chance at child scale. (Opinion.) The unit that matters for data efficiency is the chunk the learner builds for itself, at several levels, and that is a question about the learning objective.

## 3. Mathematics or formal-language-first pretraining

### 3.1 Results

**Papadimitriou and Jurafsky (2020), "Learning music helps you read" (EMNLP).** LSTMs pretrained on 100M-token non-linguistic corpora, then tested on Spanish with only the embeddings retrained. Mean Spanish perplexity by pretraining corpus: random uniform 513.7, random Zipfian 493.2, MIDI music 256.2, Java code 139.1, nesting parentheses 171.0, flat (non-nesting) paired parentheses 170.3; pretraining on Spanish itself gave 52.3 and on other natural languages 61 to 118 [36]. So structured non-language data closed a large part of the gap between random and natural-language pretraining, and paired dependencies without recursion transferred as well as recursion did.

**Papadimitriou and Jurafsky (2023), "Injecting structural hints" (Findings of EMNLP).** Transformers pretrained on synthetic languages then fine-tuned on English, Japanese and Basque. Crossing (non-context-free) dependencies were a better inductive bias than recursive nesting; mixing 1% crossing tokens into a nesting language improved downstream perplexity by about 5 points and 10% by about 9 points. A Zipfian vocabulary distribution helped even with no token correspondence [37].

**Hu, Petty, Shi, Merrill and Linzen (2025), "Between circuits and Chomsky" (ACL).** "Pre-pretraining" transformers on formal languages before natural language. Languages with hierarchical dependencies (k-Dyck, k-Shuffle Dyck) helped; the copy language ww hurt. Pre-pretraining reduced loss more than an equal amount of extra natural language. A 1B Pythia model on 1.63B tokens of C4 reached the baseline's final loss after 1.10B total tokens with k-Shuffle Dyck pre-pretraining: a 33% token saving, with only about 0.03B formal-language tokens, so each formal token replaced about 17 natural tokens. Attention heads learned during pre-pretraining stayed important for syntactic evaluations [38].

**Procedural data.** Shinnick et al. (2025) pretrained small transformers on outputs of simple algorithms (k-Dyck, sets, stacks, identity, cellular automata). Different rules put different, composable structure into different layers; transferring only attention layers from Identity pretraining raised a retrieval task from 18.8% to 99.0%; procedural pretraining beat OpenWebText pretraining on multiplication [48]. Krishna, Bigham and Lipton (2021) found that pretraining a T5-small on nonsense text with summarization-like tasks recovered on average 95% of T5's ROUGE-1 gain on CNN-DailyMail and Rotowire, and 37% on XSum and Rotten Tomatoes; the authors read this as evidence that much of pretraining's benefit comes from a better initialization for optimization [49]. Zhang et al. (2025) found GPT-2 models pretrained on complex (class IV) cellular automata transferred best to reasoning tasks [24].

**Code.** Aryabumi et al. (2024) trained 470M to 2.8B models with and without code. Adding code gave up to 8.2% relative improvement in natural-language reasoning, 4.2% in world knowledge and 6.6% in generative win rate, compared with text-only pretraining [39].

### 3.2 What transferred and how much data it saved

- What transferred: dependency structure (matched pairs, crossing links, hierarchy) and a Zipfian frequency profile; procedural skills such as retrieval and stack-like memory, mostly living in attention weights.
- How much: the only direct data-saving number at LM scale is Hu et al.'s 33% fewer natural-language tokens for the same loss at 1B parameters. The 2020 and 2023 papers report perplexity gains on a fixed budget and give no data-savings figure. I found no published result for formal-language pre-pretraining run inside BabyLM at 10M or 100M words; treat any such number as UNVERIFIED.
- What did not transfer: world knowledge. Nothing here teaches that ice is cold. EWoK-style knowledge (Section 7) is exactly where child-scale models fail.

### 3.3 Should we teach mathematics first?

(Opinion, grounded in the above.) The evidence supports teaching structure first, in small doses: a few percent of the training budget in formal languages with hierarchical and crossing dependencies, then natural language. The evidence does not support "mathematics" in the school sense first. Arithmetic and algebra are formal languages with particular structure, and they are worth including as some of the procedural data, but the transfer comes from the dependency pattern. Children also do the opposite: they learn language for years before formal arithmetic. A math-first curriculum is a cheap experiment to add (Implications) and should be compared against a Dyck-style pre-pretraining arm at the same token count.

## 4. Composed or hybrid systems

The pattern that works in every success below: a neural model proposes or translates, and an exact component (interpreter, solver, deduction engine, evaluator, or probabilistic inference) checks or computes. The pattern's weak point is always the translation step.

**LLM plus interpreter (PAL, Gao et al. 2023, ICML).** The LLM writes Python for a word problem and the interpreter computes the answer. Across 13 math, symbolic and algorithmic tasks it beat much larger models; with Codex it beat PaLM with chain-of-thought on GSM8K by 15 points absolute [53].

**LLM plus symbolic solver (Logic-LM, Pan et al. 2023, Findings of EMNLP).** The LLM translates a logic problem into solver input (logic programming, first-order logic, constraint satisfaction, SAT); a deterministic solver answers; solver error messages drive self-refinement. Average gains of 39.2% over standard prompting and 18.4% over chain-of-thought across five datasets, with the gap widening as reasoning depth grows. Failure mode: when the translation does not parse, the system falls back to chain-of-thought, and more refinement rounds raised executability without raising accuracy, so translation errors that still parse are the limit [54].

**LLM plus deduction engine trained on synthetic data (AlphaGeometry, Trinh et al. 2024, Nature).** A language model trained from scratch on 100M synthetic theorem-proof pairs proposes auxiliary constructions; a symbolic engine does the deduction. It solved 25 of 30 olympiad geometry problems; the symbolic engine alone solved 14 and the best hand-heuristic system 18. With 20% of the training data it still solved 21 [51]. The data cost was 100,000 CPU workers for 72 hours of synthetic generation. This is the clearest case of a hybrid trained on generated formal data instead of human text.

**LLM plus program search and evaluator (FunSearch, Romera-Paredes et al. 2024, Nature).** An evolutionary loop: an LLM mutates programs, an evaluator scores them. It found a larger cap set in dimension 8 than previously known and raised the asymptotic lower bound on cap set capacity to 2.2202, plus better online bin-packing heuristics [52]. The evaluator does the trusting; the LLM supplies variety.

**LLM plus probabilistic program (Wong et al. 2023, "From word models to world models").** An LLM (Codex, few-shot) translates sentences into Church code in a "probabilistic language of thought"; Bayesian inference over the program answers questions. Demonstrated on probabilistic, relational, physical and social reasoning, with physics simulators and planners as modules [50]. The paper is a framework with worked examples and no benchmark numbers; the translator is a large pretrained model, so it says nothing yet about data efficiency.

**Neural plus SMC.** Covered in Section 1.2: SMC over a small LM's outputs with constraint potentials let small models beat models 8x larger in three of four domains [12].

**Dual-process designs.** The successes above are dual-process systems in practice: a fast associative proposer (System 1) and a slow exact checker (System 2). I did not find a dual-process system trained from scratch at child scale with published results; treat any such claim as UNVERIFIED.

**Relevance to "small LLM plus something radically different".** Yes, composition works, and it lets a smaller neural part reach the accuracy of a larger one, in domains with an exact checker. The open problem for us is that child-level language tasks rarely have an exact checker. The one place a checker exists at child scale is the world model: a six-year-old's questions about objects, quantities, containment, causation and agents can be answered by a small simulator or probabilistic program, if the learner can translate the question into it. (Opinion.) That suggests the hybrid to test: a child-scale LM as translator, a probabilistic program or simulator as world model, SMC as the glue, all trained or built under the same word budget as the baseline.

## 5. Child language input numbers

### Sources and their raw figures

**Hart and Risley (1995; summarized by the authors in 2003).** Forty-two Kansas families were recorded one hour a month from about 10 months to 3 years. Average words heard per hour: 616 (welfare), 1,251 (working class), 2,153 (professional). The authors extrapolated to a 100-hour waking week and a 5,200-hour year: 3.2M, 6.5M and 11.2M words per year, and 13M, 26M and 45M by age four [40]. Their count covers speech in the presence of the child from the primary caregiver, sampled one hour a month, so it is an extrapolation from a narrow window.

**Gilkerson et al. (2017), LENA.** 3,213 twelve-hour recordings of 329 children aged 2 to 48 months. Adult word counts were "age independent after early infancy" [41]. The LENA technical report on the same norms gives a mean adult word count of 12,297 per 12-hour day, SD 6,462 [42]. LENA counts every adult word the device picks up near the child, including overheard speech, but it covers only the 12 recorded hours.

**Sperry, Sperry and Miller (2019).** Forty-two children aged 18 to 48 months in five US communities. Counting all caregivers and bystander talk changes the picture: middle-class primary caregivers spoke 1,491 words per hour to the child and all caregivers 1,777; children in the low-income Black Belt community heard 3,203 words per hour of ambient speech, above Hart and Risley's professional mean of 2,153. The authors conclude the data "do not support Hart and Risley's claim" of a 30-million-word gap [43].

**Frank (2023), "Bridging the data gap".** A soft upper bound of about 10^6 words per month (6 x 10^7 by age five) and a lower bound for language-poor environments of about 10^5 words per month (6 x 10^6 by age five) [44]. Frank's own framing: LLMs see four to five orders of magnitude more language than children.

The BabyLM organizers cite Gilkerson et al. for the claim that children hear fewer than 100M word tokens by age 13 [45].

### Words heard by age six, and the experiment budget

The table extends each source to 72 months. Where the source gives an hourly rate, the extension uses Hart and Risley's 5,200 waking hours per year; where it gives a daily rate, 365 days. These extensions are my arithmetic, not figures the authors published, and they assume flat input from birth (LENA supports flatness after early infancy; nothing supports it for the first months).

| Source and condition | Rate | By age 6 | 2x | 4x |
|---|---|---|---|---|
| Frank lower bound | 10^5 / month | 7.2M | 14.4M | 28.8M |
| LENA mean minus 1 SD | 5,835 / day | 12.8M | 25.6M | 51.1M |
| Hart and Risley, welfare | 3.2M / year | 19.2M | 38.4M | 76.8M |
| LENA mean | 12,297 / day | 26.9M | 53.9M | 107.7M |
| Hart and Risley, working class | 6.5M / year | 39.0M | 78.0M | 156.0M |
| LENA mean plus 1 SD | 18,759 / day | 41.1M | 82.2M | 164.3M |
| Hart and Risley, professional | 11.2M / year | 67.2M | 134.4M | 268.8M |
| Frank upper bound | 10^6 / month | 72.0M | 144.0M | 288.0M |
| Sperry, highest ambient mean (Black Belt) | 3,203 / hour | 99.9M | 199.9M | 399.7M |

Summary range for a six-year-old: about 7M to 100M words, with a central estimate near 27M (LENA mean) to 39M (Hart and Risley working class). Two times that is 14M to 200M (central 54M to 78M). Four times is 29M to 400M (central 108M to 156M).

**Comparison with BabyLM.** The 10M Strict-Small track sits near the low end of what a single six-year-old hears. The 100M Strict track sits at roughly 4x the LENA mean or between 2x and 4x the Hart and Risley working-class figure. A "2x to 4x a six-year-old" experiment therefore lands on the BabyLM 100M track for the central estimates, which gives us published baselines at the same budget. Two caveats apply. BabyLM's corpus is 29% CHILDES and the rest is subtitles, Simple Wikipedia, children's books and dialogue [45], so it is richer in written and adult text than what a child hears. And children hear speech with prosody, in context, with interaction, while text models get transcripts.

## 6. Child-level evaluation

### 6.1 What exists

**BLiMP and BLiMP Supplement.** Minimal pairs of grammatical and ungrammatical sentences across phenomena such as agreement, binding, islands; the model must prefer the grammatical one. BabyLM's text tracks use both [45].

**Zorro (Huebner et al. 2021).** A BLiMP-style suite restricted to the vocabulary of child-directed speech (AO-CHILDES, about 5M words, ages 1 to 6), covering 11 of BLiMP's 12 phenomena. With it, RoBERTa-base trained from scratch on 5M words scored 59.2 against 81.1 for the 30B-word RoBERTa-base; BabyBERTa (8M parameters, no unmasking, single sentences) reached 80.5, within half a point of RoBERTa-base with about 15x fewer parameters and 6,000x fewer words [55]. Zorro is the grammar test that fits a CHILDES-sized vocabulary.

**EWoK (Ivanova et al. 2024; TACL 2025).** 4,374 items over 11 world-knowledge domains (social interactions, spatial relations, physical relations, agents, numbers, and others); the same target sentence must be judged plausible after one context and implausible after another. Even LLMs from 1.3B to 70B fall below humans; spatial relations was hardest for LLMs (mean 0.62, best 0.75, humans 0.96) [56]. BabyLM models sit near chance (Section 7).

**BabyLM 2025 additions** [47, 57]: entity tracking (state of numbered boxes after moves, scored as five-way choice), WUG past tense and WUG adjective nominalization (correlation with human choices for nonce words), COMPS (property inheritance with nonce words), reading-time and eye-tracking fit, age-of-acquisition curves against MacArthur-Bates CDI norms, and a subsampled (Super)GLUE (BoolQ, MultiRC, RTE, WSC, MRPC, QQP, MNLI). The organizers tried zero-shot prompting for GLUE but "concluded that the limited data size does not allow for robust in-context learning to emerge," so GLUE still needs fine-tuning.

**DevBench (Tan et al. 2024, NeurIPS Datasets and Benchmarks).** Seven multimodal tasks (lexical, syntactic, semantic) with item-level child and adult data; the metric compares model response patterns with the human response distribution, and the authors do not rank by accuracy. CVCL, trained on one child's head-camera data, was "mostly dissimilar to humans" on lexical and syntactic tasks but best on visual object categorization [58].

**MEWL (Jiang et al. 2023, ICML).** Nine few-shot word-learning tasks in visual scenes (cross-situational learning, bootstrapping, pragmatic learning), aligned to developmental findings on fast mapping; the authors report "a sharp divergence in human and machine word learning" [59].

**Child-directed questions.** I found no established benchmark of questions a six-year-old answers, scored against children. Related: a 2025 SIGDIAL paper built a 1,000-item set of caregiver questions from CHILDES and scored model replies for coherence with ChatGPT [60]; Poh, Tombolini and Barak (2025) report CHILDES has the highest question rate in the BabyLM corpus (20.5% of utterances) [61]. A curated, child-normed QA set would need to be built; mark its validity as untested.

**Production versus comprehension.** Hu, Tan, Feng and Frank (2025, CogSci) report that "language production is harder than comprehension for children and language models" (title only verified; details UNVERIFIED) [62]. This matters because almost every BabyLM metric is a comprehension-style probability comparison, while "answering a question" is production.

### 6.2 What a six-year-old can do that a child-scale LM cannot (from the evidence above)

- Basic physical and spatial world knowledge: EWoK maximum 58.4% on 100M words, near the 50% chance level [45].
- Tracking entities through a short story: 2025 baselines score 19% to 40% on the five-way entity-tracking task (Section 7).
- Learning a new word from one or a few exposures and using it: MEWL shows a large human-machine gap [59]; child-scale LMs have no measured in-context learning [47].
- Answering novel questions by generation. BabyLM evaluations avoid generation entirely.
- Morphological generalization to nonce words: WUG past tense shows no improvement for the first 10M to 50M words in BabyLM 2025 [47]; children produce "wugs" at preschool age (Berko 1958; UNVERIFIED in this pass, no source fetched).

### 6.3 Proposed evaluation set for our experiment

Every item must use vocabulary present at least twice in the training corpus, following BabyLM's filtering rule [45]. Score both systems the same way: probability comparison for the minimal-pair parts, and generated answers for the question part.

| Block | Source | What it measures | Scoring | Why include |
|---|---|---|---|---|
| A. Grammar (child vocabulary) | Zorro, all phenomena [55] | Syntax and morphology | Minimal-pair accuracy | Fits a CHILDES-scale lexicon; strong published baselines |
| B. Grammar (broad) | BLiMP + Supplement [45] | Syntax, discourse | Minimal-pair accuracy | Comparability with every BabyLM paper |
| C. World knowledge | EWoK-core-1.0, child-vocabulary filtered [56] | Physical, spatial, social, number | Context-target plausibility | The place child-scale LMs fail |
| D. Entity tracking | BabyLM 2025 version [57] | State over a short narrative | Five-way choice | Six-year-olds pass simple versions |
| E. Novel-word generalization | WUG past tense and adjective [57]; COMPS [57] | Morphological and property generalization | Correlation with humans; minimal pairs | Tests rule extraction from few examples |
| F. Fast mapping (text-only) | New: MEWL-style tasks rendered in text, with nonce words defined in one or two sentences, then queried [59] | One-shot word learning | Accuracy | The central data-efficiency claim |
| G. Child questions | New: 300 to 500 questions written to a six-year-old norm (where, what happens if, which is bigger, why did she cry), drawn from CHILDES question types [61]; answers judged by two raters blind to system | Generative question answering | Exact or rater-judged correctness | Andrew's stated target; no existing benchmark |
| H. Human-likeness | Reading-time fit; AoA correlation with CDI [57] | Cognitive plausibility | Correlations | Optional; useful for papers |

Blocks F and G are new and need a pilot with children or at least with published child norms before any claim that a system "answers like a six-year-old." Until then, report them as engineering measures.

## 7. Training baselines on child-scale data

### BabyLM 2024 (second challenge) results [45]

Thirty-one papers, 64 models. Text tracks were scored on BLiMP, BLiMP Supplement, (Super)GLUE after fine-tuning, and EWoK (hidden). Macro averages from the findings paper's Table 3:

| Track | System | BLiMP | BLiMP Supp. | (Super)GLUE | EWoK | Text avg |
|---|---|---|---|---|---|---|
| Strict (100M) | GPT-BERT (winner) | 86.1 | 76.8 | 81.5 | 58.4 | 75.7 |
| Strict | BabbleGPT | 77.9 | 69.5 | 71.7 | 52.0 | 67.8 |
| Strict | Best baseline, LTG-BERT | 69.2 | 66.5 | 68.4 | 51.9 | 64.8 |
| Strict-Small (10M) | GPT-BERT (winner) | 81.2 | 69.4 | 76.5 | 54.6 | 70.4 |
| Strict-Small | DeBaby | 74.2 | 63.7 | 73.7 | 54.3 | 66.5 |
| Strict-Small | Best baseline, BabyLlama | 69.8 | 59.5 | 63.3 | 50.7 | 61.6 |

Findings that matter for us:
- The winner, GPT-BERT, mixes masked and causal language modeling by shifting masked predictions one position so both objectives predict the next token; a 1:7 causal-to-masked ratio worked well. It sits on an LTG-BERT backbone. Combining objectives "clearly improves performance over single objective baselines."
- Training FLOPs predicted average score (mixed-effects regression, beta = 2.7 per log-FLOP unit, p < 0.01). Data limits do not remove the compute effect.
- Backbone architecture had no statistically significant effect. HGRN (a gated RNN) scored roughly level with transformers.
- Curriculum learning was the most popular approach and had a negative, near-significant effect (beta = -3.6, p = 0.055). "Linguistic bias" approaches had a significant negative effect (beta = -7.3).
- Significant positive effects: training-objective changes (+4.5), dataset construction (+4.8), architecture changes (+3.5).
- EWoK, a world-knowledge test, stayed near chance: maximum 58.4%, most near 50%. The organizers state that "current systems do not learn world knowledge within 100M words."
- No multimodal submission beat the multimodal baselines.
- BabyLlama-style distillation from an ensemble of teachers beats training a single causal model.

"Average" results: the findings paper gives a figure of all submissions but no single mean in the text I retrieved. The baselines are a fair stand-in for a typical well-tuned standard model: LTG-BERT at 64.8 text average on 100M words and BabyLlama at 61.6 on 10M words.

### BabyLM 2023 (first challenge) [46]

Baselines were OPT-125M, RoBERTa-base and T5-base trained 20 epochs. Winners in both Strict and Strict-Small were ELC-BERT, an LTG-BERT variant trained for over 450 epochs (Strict) and over 2,000 epochs (Strict-Small). Aggregate scores (weighting BLiMP plus supplement 50%, (Super)GLUE 30%, MSGS 20%): Strict ELC-BERT 0.74 against best baseline OPT-125M 0.60; Strict-Small ELC-BERT 0.66 against OPT-125M 0.50. LTG-BERT models beat the Llama 2 70B and RoBERTa-base skylines on overall score. The curriculum-learning paper CLIMB won an award for a careful negative result: none of its curricula gave broad gains.

### BabyLM 2025 (third challenge) and GPT-2 small numbers [47, 57]

The 2025 challenge capped total training exposure (100M words seen for Strict-Small, 1B for the other tracks, so at most 10 passes), required intermediate checkpoints, and added human-likeness tasks. Aggregates from the findings paper's Table 3 (human-likeness score, NLP score, macro average):

| Track | System | Human-likeness | NLP | Macro |
|---|---|---|---|---|
| Strict | GPT-2 small baseline | 17.1 | 55.4 | 36.2 |
| Strict | Best baseline, GPT-BERT mixed MNTP | 24.2 | 62.7 | 43.5 |
| Strict | Best submission, BLaLM (workshop) | 23.6 | 54.4 | 39.0 |
| Strict | Simple-Diffusion (NLP winner) | 12.6 | 58.4 | 35.5 |
| Strict-Small | GPT-2 small baseline | 19.8 | 49.1 | 34.5 |
| Strict-Small | MoEP (human-likeness winner) | 31.5 | 53.2 | 42.3 |
| Strict-Small | AMLM-Hard | 26.7 | 58.0 | 42.3 |
| Interaction | SimPO baseline (GPT-2 small student, Llama-3.1-8B teacher) | 20.4 | 54.1 | 37.3 |
| Interaction | BLM (winner) | 20.8 | 54.4 | 37.6 |

Per-task GPT-2 small baselines from the 2025 evaluation pipeline repository [57]:

| Budget | BLiMP | BLiMP Supp. | EWoK | Entity tracking | WUGs |
|---|---|---|---|---|---|
| 10M words (Strict-Small) | 67.3 | 59.1 | 49.8 | 18.9 | 39.0 |
| 100M words (Strict) | 75.1 | 65.0 | 51.2 | 30.1 | 42.5 |

Findings: objective and architecture changes won again (diffusion masked LMs, mixture of experts, RL with a teacher); curriculum learning stayed popular and did not win; FLOPs correlated with score only in the Interaction track; the Strict baselines (GPT-BERT variants) beat every Strict submission on the macro average. WUG past tense showed no change for the first 10M to 50M words, then rose.

### Best and typical results, summarized

- Best text-only results at 100M words: GPT-BERT 2024, BLiMP 86.1, text average 75.7, with BLiMP within 2.5 points of the human score [45].
- Best at 10M words: GPT-BERT 2024, BLiMP 81.2, text average 70.4 [45].
- Typical well-tuned standard model: LTG-BERT 64.8 (100M) and BabyLlama 61.6 (10M) text averages in 2024 [45]; GPT-2 small BLiMP 75.1 (100M) and 67.3 (10M) in 2025 [57].
- World knowledge at any of these budgets: near chance [45, 47].
- Caveat: the 2024 and 2025 pipelines differ (the 2025 pipeline was rewritten and its GLUE subset changed), so compare scores within a year, and rerun every baseline on our own pipeline before comparing with a candidate.

### The baseline to match budget for budget

(Opinion.) Run two baselines on exactly our training data:
1. **GPT-2 small (124M parameters), causal LM**, with the BabyLM 2025 training recipe and its 10-pass exposure cap. It is the "standard model" Andrew means, it has published numbers at 10M and 100M words, and it is the student architecture in the interaction-track baseline.
2. **GPT-BERT (mixed causal and masked)**, the strongest published recipe at both budgets. Beating GPT-2 small shows we beat a standard LLM; beating GPT-BERT shows we beat the best known child-scale method.

Optionally add **BabyBERTa** (8M parameters), because it reaches RoBERTa-base Zorro scores on 5M words and is cheap.

### Compute on this machine

Training FLOPs are approximately 6 x parameters x training tokens (the standard dense-transformer approximation). Tokens per word depends on the tokenizer; I assume 1.2 to 1.5 BPE tokens per English word, which is an assumption and should be measured on the corpus.

| Model | Words seen (10 passes) | Training FLOPs (approx.) |
|---|---|---|
| GPT-2 small, 124M | 100M (10M corpus) | 0.9 to 1.1 x 10^17 |
| GPT-2 small, 124M | 1B (100M corpus) | 0.9 to 1.1 x 10^18 |
| BabyBERTa-size, 8M | 100M | 5.8 to 7.2 x 10^15 |
| BabyBERTa-size, 8M | 1B | 5.8 to 7.2 x 10^16 |

The only measured training rate on this machine is in this repository's IMPROVEMENT-PLAN.md (sections B.6 and G.3) [65]: about 40,000 tokens per second on one CPU thread for a 101,644-parameter transformer, which the plan converts to about 2 x 10^10 training FLOPs per second per thread including overhead, with an unmeasured assumed 5x speedup on 8 threads. Extrapolating from that rate only:

- GPT-2 small, 1B words seen: about 5 x 10^7 thread-seconds, roughly 580 days on one thread or about 115 days at the assumed 8-thread rate. Not feasible on CPU.
- GPT-2 small, 100M words seen: about 58 days on one thread, about 12 days at the assumed 8-thread rate.
- BabyBERTa-size, 100M words seen: about 3.5 to 4 days on one thread, under a day at the assumed 8-thread rate.

The plan notes these figures are more likely high than low because larger matrix products run more efficiently than the tiny measured model. The host GPU (MPS) has not been measured; a one-hour GPT-2 small throughput run on MPS is the measurement that decides whether the 100M-word GPT-2 baseline is practical locally. Until then, the feasible local comparison is the 10M-word budget with GPT-2 small on GPU, or BabyBERTa-size models on CPU.

## 8. Developmental and interactive learning

### 8.1 Curricula and child-directed speech: mostly no help

- **CLIMB (Martinez et al. 2023)**, a systematic test of vocabulary, data-difficulty and objective curricula, found none gave broad improvements [46].
- **BabyLM 2024 regression:** curriculum learning had a negative, near-significant coefficient (beta = -3.6, p = 0.055) on average score [45].
- **Feng et al. (2024, EMNLP), "Is child-directed speech effective training data?"** GPT-2 and RoBERTa trained on 29M words of CHILDES did worse on Zorro and word similarity than the same budget of mixed BabyLM data or OpenSubtitles; a GPT-4-generated synthetic dialogue corpus (TinyDialogues) did better than natural CHILDES. Age ordering (global curriculum) had little effect; scrambling utterance order within conversations (local coherence) hurt. The authors conclude that the child's advantage lies in the learner: "the child's learning algorithm is substantially more data-efficient than current language modeling techniques" [63].
- **Exceptions:** Haga et al. (2024) found synthetic "variation sets" (consecutive paraphrases, common in speech to children) can improve BabyLM scores, though "the exact characterization of this relationship remains unclear" [45]. Salhan et al. (2024) used a curriculum over objectives (which parts of speech to mask when) and the organizers singled it out as a more promising curriculum direction than data ordering [45].

For current architectures, the order and register of child input do not explain children's efficiency, so the search should concentrate on the learner.

### 8.2 Interaction and learning from a teacher

The BabyLM 2025 interaction track allowed a teacher model (Llama-3.1-8B-Instruct or smaller) to give feedback or generate data conditioned on the student's outputs, with no access to teacher weights or distributions, and 100M external words plus 100M self-generated words. The baseline: a GPT-2 small student completes text, the teacher revises it, and the student trains on the revision then with SimPO preference optimization, over 20 rounds. The winner (BLM) barely beat that baseline (macro 37.6 against 37.3), and the organizers note that "performance was generally higher in the strict track" despite the extra flexibility [47]. Interaction as currently implemented (a large LM correcting a small one) did not beat plain pretraining at the same word budget. It is also a weak model of a child's interaction, which is grounded, about the here and now, and contingent on the child's own attention.

### 8.3 Grounding at child scale

**Vong, Wang, Orhan and Lake (2024, Science).** CVCL, a contrastive vision-language model, trained on 61 hours of one child's head-camera video (SAYCam-S, ages 6 to 25 months; 600,000 frames, 37,500 utterances, about 225,000 word tokens). The authors estimate this is about 1% of the child's waking hours and 0.8% to 2.2% of the language the child heard. On 22 concepts in a four-way image-choice test, CVCL scored 61.6%, and for 11 of 22 concepts it was within 5 points of CLIP trained on 400M image-text pairs. It generalized zero-shot to new visual referents [64].

**Limits.** DevBench found CVCL "mostly dissimilar to humans" on lexical and syntactic tasks, though most human-like on visual categorization [58]. In BabyLM 2024 no multimodal submission beat the multimodal baselines, and the 2025 multimodal best submission scored below both baselines [45, 47]. Word-referent mapping for concrete nouns is learnable from tiny grounded data; grammar and abstract words have not been shown to benefit from vision at child scale.

### 8.4 Section verdict

(Opinion.) The best developmental lever in the literature is grounding for word meaning, and it works for concrete nouns only so far. Curricula, child-directed register and teacher feedback have each been tested at BabyLM scale and gave small or no gains. If children's efficiency comes from the learner, an experiment should hold data fixed and vary the learner, which is the design below.

## Implications

Everything in this section is opinion, built on the evidence above.

### Answers to Andrew's questions

**Are vectors and linear algebra the wrong substrate?** The evidence does not show that. No non-vector learner in Section 1 has matched a small transformer on any child-scale language benchmark. Where alternatives win (Lake et al.'s one-shot characters, SMC-steered small models beating 8x larger ones, AlphaGeometry), they win on inference, composition or search, usually on top of a learned vector model. The defensible bet is a vector model kept small and used for perception and scoring, with new mathematics in how hypotheses are represented (populations of discrete samples, programs) and in the learning objective.

**Could a sampling or Monte Carlo representation replace vectors?** It can replace point estimates of structure: which parse, which referent, which meaning, which program. A particle set over structured hypotheses is a real alternative representation for those, with a cost linear in particle count and a known accuracy profile (few particles reproduce human garden-path failures). It does not replace the scoring function, and no one has shown sampling-based learning of a lexicon and grammar from child-scale text. That makes it a research gap worth a proposal.

**Are tokens the wrong unit?** At child scale, the evidence says the unit matters little: BPE beat phoneme and character units in BabyLM 2024, and speech-first learners are near chance on child-scale audio. Learned, surprise-driven chunking (entropy patching, Chunk-and-Pass) is the version of this idea worth testing.

**Should we teach mathematics first?** Teach structure first, briefly. Formal languages with hierarchical and crossing dependencies, a few percent of the budget, saved 33% of natural-language tokens at 1B parameters and transferred through attention. School mathematics has no direct evidence; include arithmetic as one procedural source and compare it with Dyck-style data at equal tokens.

**Could a small LLM plus something radically different work?** Yes where an exact checker exists. For six-year-old questions, the checker can be a small world model: a probabilistic program or simulator for objects, containers, quantities, motion and agents' goals. EWoK, entity tracking and child questions are where child-scale LMs fail and where such a world model would answer. The hybrid's weak point will be translation from sentences into the world model; at child scale the translator has to be learned from the same word budget, unlike Wong et al., who used Codex.

### How the research system should generate proposals on these substrates

1. **Require a fixed proposal schema.** Each proposal names: (a) the representation of linguistic hypotheses (vector, sparse code, particle set, program, assembly, graph); (b) the learning rule and what it updates per word; (c) what one observation shares across many contexts (Miller and Chomsky's estimation problem; a proposal with no sharing mechanism is rejected); (d) inference cost per word in operations; (e) the evaluation block (Section 6.3) it predicts it will move, with a numeric prediction; (f) a kill criterion at the 10M-word gate; (g) a runtime estimate from a measured rate on this machine.
2. **Generate by crossing three axes**, then prune: substrate (Section 1 list) x objective (next-unit prediction, masked plus causal, energy or contrastive, program-likelihood, SMC evidence) x data intervention (formal pre-pretraining, synthetic variation sets, grounding stubs). Drop combinations whose only novelty is curriculum or child-directed register, since BabyLM evidence is negative there, unless they bring a new mechanism.
3. **Stage gates.** 1M-word toy (Zorro subset, CPU, hours), then 10M words (full blocks A to F), then 50M and 100M only for proposals that beat GPT-2 small at 10M on blocks A plus C, or match it with at least 10x fewer training FLOPs.
4. **Novelty check against prior art** in this note and the other research reports under reports/, recorded per proposal, so that "new mathematics" means a mechanism not already tested at BabyLM scale.
5. **Pair every non-vector proposal with its best vector analogue** at equal parameters and words, so a win can be attributed to the substrate.

### Child-scale comparison experiment

**Question.** Does a candidate learner trained on two to four times a six-year-old's language input answer child-level questions better than a standard model trained on the same data?

**Data.**
- Budgets: 50M words (about 2x the LENA-mean six-year-old at 26.9M, or 1.3x Hart and Risley working class) and 100M words (about 4x LENA mean, 2.6x Hart and Risley working class). A 10M-word pilot (near one low-input six-year-old) runs first.
- Corpus: the BabyLM 100M text corpus (29% CHILDES, 26M children's books, dialogue, subtitles, Simple Wikipedia) so results are comparable to published numbers; the 50M set is a stratified subsample. A second arm can use a CHILDES-heavier mix, though Feng et al. predict no gain.
- Exposure: at most 10 passes, as in BabyLM 2025. Report words seen, unique words, and training FLOPs for every run.
- Evaluation vocabulary filter: drop test items with any word seen fewer than twice in the 10M subset.

**Systems.**
- Baseline 1: GPT-2 small (124M), causal LM, BabyLM 2025 recipe.
- Baseline 2: GPT-BERT (mixed causal and masked), BabyLM 2025 recipe.
- Optional baseline 3: BabyBERTa-size model (8M).
- Candidate A (hybrid): a small LM translator, a probabilistic world-model program library, and SMC over interpretations, all built or trained within the word budget; any hand-built world-model code counts as prior knowledge and is reported as such.
- Candidate B (structure first): each baseline plus k-Shuffle Dyck pre-pretraining at 1% to 3% of tokens, to separate data effects from architecture effects.
- Candidate C (substrate): the best non-vector proposal that passes the 10M gate.
- Three seeds per system and budget.

**Evaluation.** Blocks A to H from Section 6.3, at checkpoints of 10M, 25M, 50M and 100M words seen.

**What result would count.**
- *Data efficiency win:* the candidate at 50M words meets or beats GPT-2 small at 100M words on the mean of blocks A to E (reference points: GPT-2 small 100M BLiMP 75.1, EWoK 51.2, entity tracking 30.1), by more than two seed standard deviations, at equal or lower training FLOPs.
- *Child-question win:* at equal words, the candidate beats both baselines on block G by at least 10 percentage points and on block C by at least 5 points above the best published BabyLM EWoK score (58.4).
- *Compute win:* equal scores on blocks A to E at one tenth of the baseline's training FLOPs.
- *Stop:* below GPT-2 small at equal words on both Zorro and BLiMP at the 10M gate.
- Pre-register these thresholds before training. Report every block, including those where the candidate loses.

**Feasibility on this machine.** GPT-2 small at 100M words and 10 passes is about 10^18 FLOPs, which the measured CPU rate puts at roughly 115 days on 8 threads; a GPU throughput measurement on this machine comes first. The 10M pilot (about 10^17 FLOPs for GPT-2 small, under 10^16 for an 8M model) fits locally.

## References

Numbers are grouped by topic and are not in order of first appearance. "Not fetched" marks entries whose bibliographic details come from memory or from another paper's reference list and were not opened in this pass.

1. Chomsky, N. (1956). Three models for the description of language. IRE Transactions on Information Theory, IT-2(3), 113 to 124. https://chomsky.info/three-models-for-the-description-of-language/ and https://researchr.org/publication/chomsky56
2. Chomsky, N. (1957). Syntactic Structures. Mouton. Not fetched. https://en.wikipedia.org/wiki/Syntactic_Structures
3. Miller, G. A., and Chomsky, N. (1963). Finitary models of language users. In Luce, Bush and Galanter (eds.), Handbook of Mathematical Psychology, vol. II, 419 to 491. https://faculty.wcas.northwestern.edu/robvoigt/courses/2021_fall/ling334/readings/finitary_models.pdf
4. Pullum, G. K., and Gazdar, G. (1982). Natural languages and context-free languages. Linguistics and Philosophy 4, 471 to 504. https://www.jstor.org/stable/25001071 ; https://pullum.ppls.ed.ac.uk/publications.html
5. Bresnan, J., Kaplan, R. M., Peters, S., and Zaenen, A. (1982). Cross-serial dependencies in Dutch. Linguistic Inquiry 13(4), 613 to 635. https://web.stanford.edu/~bresnan/bresnan.kaplan.peters.zaenen.LI.1982.pdf
6. Shieber, S. (1985). Evidence against the context-freeness of natural language. Linguistics and Philosophy 8(3), 333 to 343. Details from the Wikipedia "Cross-serial dependencies" reference list; not fetched. https://doi.org/10.1007/BF00630917
7. Pereira, F. (2000). Formal grammar and information theory: together again? Phil. Trans. R. Soc. A 358, 1239 to 1253. https://royalsocietypublishing.org/doi/10.1098/rsta.2000.0583 ; PDF https://sites.socsci.uci.edu/~rfutrell/teaching/itl-davis/readings/pereira2000formal.pdf
8. Vul, E., Goodman, N., Griffiths, T. L., and Tenenbaum, J. B. (2014). One and done? Optimal decisions from very few samples. Cognitive Science 38(4), 599 to 637. https://onlinelibrary.wiley.com/doi/10.1111/cogs.12101
9. Sanborn, A. N., and Chater, N. (2016). Bayesian brains without probabilities. Trends in Cognitive Sciences 20(12), 883 to 893. https://doi.org/10.1016/j.tics.2016.10.003
10. Levy, R., Reali, F., and Griffiths, T. L. (2009). Modeling the effects of memory on human online sentence processing with particle filters. NeurIPS 21 (2008 proceedings). https://papers.nips.cc/paper_files/paper/2008/hash/a02ffd91ece5e7efeb46db8f10a74059-Abstract.html
11. Lew, A. K., Zhi-Xuan, T., Grand, G., and Mansinghka, V. K. (2023). Sequential Monte Carlo steering of large language models using probabilistic programs. https://arxiv.org/abs/2306.03081
12. Loula, J., LeBrun, B., Du, L., et al. (2025). Syntactic and semantic control of large language models via sequential Monte Carlo. ICLR 2025. https://arxiv.org/abs/2504.13139
13. Cusumano-Towner, M. F., Saad, F. A., Lew, A. K., and Mansinghka, V. K. (2019). Gen: a general-purpose probabilistic programming system with programmable inference. PLDI 2019. https://dl.acm.org/doi/10.1145/3314221.3314642
14. Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction. Science 350, 1332 to 1338. https://www.science.org/doi/10.1126/science.aab3050
15. Berkes, P., Orbán, G., Lengyel, M., and Fiser, J. (2011). Spontaneous cortical activity reveals hallmarks of an optimal internal model of the environment. Science 331, 83 to 87. https://pmc.ncbi.nlm.nih.gov/articles/PMC3065813/
16. Mnih, A., and Hinton, G. (2007). Three new graphical models for statistical language modelling. ICML 2007. https://www.cs.toronto.edu/~hinton/absps/threenew.pdf
17. Deng, Y., Bakhtin, A., Ott, M., Szlam, A., and Ranzato, M. (2020). Residual energy-based models for text generation. ICLR 2020. https://arxiv.org/abs/2004.11714
18. Alaghi, A., and Hayes, J. P. (2013). Survey of stochastic computing. ACM TECS 12(2s), article 92. https://dl.acm.org/doi/10.1145/2465787.2465794
19. Joshi, A., Halseth, J. T., and Kanerva, P. (2016). Language geometry using random indexing. Quantum Interaction 2016. https://redwood.berkeley.edu/wp-content/uploads/2018/12/joshi2016language.pdf ; earlier version https://arxiv.org/abs/1412.7026
20. Coecke, B., Sadrzadeh, M., and Clark, S. (2010). Mathematical foundations for a compositional distributional model of meaning. Linguistic Analysis 36. https://arxiv.org/abs/1003.4394
21. Kartsaklis, D., et al. (2021). lambeq: an efficient high-level Python library for quantum NLP. https://arxiv.org/abs/2110.04236 (not fetched; cited via Coecke's ESSLLI 2022 notes, which also state the text-composition limitation: http://www.cs.ox.ac.uk/people/bob.coecke/ESSLLI-2022.pdf)
22. Papadimitriou, C. H., Vempala, S. S., Mitropolsky, D., Collins, M., and Maass, W. (2020). Brain computation by assemblies of neurons. PNAS 117(25), 14464 to 14472. https://www.pnas.org/doi/abs/10.1073/pnas.2001893117
23. Mitropolsky, D., Collins, M. J., and Papadimitriou, C. H. (2021). A biologically plausible parser. TACL 9, 1374 to 1388. https://aclanthology.org/2021.tacl-1.81/
24. Zhang, S., Patel, A., Rizvi, S., et al. (2025). Intelligence at the edge of chaos. ICLR 2025. https://arxiv.org/abs/2410.02536
25. Goodman, N. D., Mansinghka, V. K., Roy, D., Bonawitz, K., and Tenenbaum, J. B. (2008). Church: a language for generative models. UAI 2008. Not fetched. https://arxiv.org/abs/1206.3255
26. Bingham, E., et al. (2019). Pyro: deep universal probabilistic programming. JMLR 20. Not fetched. https://jmlr.org/papers/v20/18-403.html
27. Xue, L., et al. (2022). ByT5: towards a token-free future with pre-trained byte-to-byte models. TACL. https://aclanthology.org/2022.tacl-1.17/
28. Yu, L., Simig, D., Flaherty, C., Aghajanyan, A., Zettlemoyer, L., and Lewis, M. (2023). MegaByte: predicting million-byte sequences with multiscale transformers. NeurIPS 2023. https://arxiv.org/abs/2305.07185
29. Pagnoni, A., et al. (2024; ACL 2025). Byte Latent Transformer: patches scale better than tokens. https://arxiv.org/abs/2412.09871 ; https://aclanthology.org/2025.acl-long.453/
30. Goriely, Z., Diehl Martinez, R., Caines, A., Beinborn, L., and Buttery, P. (2024). From babble to words: pre-training language models on continuous streams of phonemes. BabyLM 2024. https://aclanthology.org/2024.conll-babylm.4.pdf
31. Lakhotia, K., et al. (2021). On generative spoken language modeling from raw audio. TACL. https://aclanthology.org/2021.tacl-1.79/
32. Hassid, M., et al. (2023). Textually pretrained speech language models (TWIST). https://arxiv.org/abs/2305.13009
33. Lavechin, M., et al. (2023). BabySLM: language-acquisition-friendly benchmark of self-supervised spoken language models. Interspeech 2023. https://arxiv.org/abs/2306.01506
34. LCM team, Meta (2024). Large Concept Models: language modeling in a sentence representation space. https://arxiv.org/abs/2412.08821 ; https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/
35. Christiansen, M. H., and Chater, N. (2016). The Now-or-Never bottleneck: a fundamental constraint on language. Behavioral and Brain Sciences 39, e62. https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/nowornever-bottleneck-a-fundamental-constraint-on-language/938D54E80A2A90A1C5990F4915B5E8D8
36. Papadimitriou, I., and Jurafsky, D. (2020). Learning music helps you read: using transfer to study linguistic structure in language models. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.554/
37. Papadimitriou, I., and Jurafsky, D. (2023). Injecting structural hints: using language models to study inductive biases in language learning. Findings of EMNLP 2023. https://aclanthology.org/2023.findings-emnlp.563/
38. Hu, M. Y., Petty, J., Shi, C., Merrill, W., and Linzen, T. (2025). Between circuits and Chomsky: pre-pretraining on formal languages imparts linguistic biases. ACL 2025. https://aclanthology.org/2025.acl-long.478/
39. Aryabumi, V., et al. (2024). To code, or not to code? Exploring impact of code in pre-training. https://arxiv.org/abs/2408.10914
40. Hart, B., and Risley, T. R. (2003). The early catastrophe: the 30 million word gap by age 3. American Educator (summarizing their 1995 book Meaningful Differences, Brookes). https://www.aft.org/ae/spring2003/hart_risley
41. Gilkerson, J., Richards, J. A., Warren, S. F., et al. (2017). Mapping the early language environment using all-day recordings and automated analysis. American Journal of Speech-Language Pathology 26(2), 248 to 265. https://doi.org/10.1044/2016_AJSLP-15-0169
42. LENA Foundation. The LENA Natural Language Study (technical report LTR-02-2). https://www.lena.org/wp-content/uploads/2016/07/LTR-02-2_Natural_Language_Study.pdf
43. Sperry, D. E., Sperry, L. L., and Miller, P. J. (2019). Reexamining the verbal environments of children from different socioeconomic backgrounds. Child Development 90(4), 1303 to 1318. https://doi.org/10.1111/cdev.13072
44. Frank, M. C. (2023). Bridging the data gap between children and large language models. Trends in Cognitive Sciences. https://pubmed.ncbi.nlm.nih.gov/37659919/ ; PDF https://sites.socsci.uci.edu/~lpearl/colareadinggroup/readings/Frank2023_ChildrenLLMs.pdf
45. Hu, M. Y., Mueller, A., Ross, C., et al. (2024). Findings of the second BabyLM Challenge. CoNLL-BabyLM 2024. https://arxiv.org/abs/2412.05149 ; https://aclanthology.org/2024.conll-babylm.1/
46. Warstadt, A., Mueller, A., Choshen, L., et al. (2023). Findings of the BabyLM Challenge. CoNLL-BabyLM 2023. https://aclanthology.org/2023.conll-babylm.1.pdf
47. Charpentier, L., Choshen, L., Cotterell, R., Gul, M. O., Hu, M. Y., et al. (2025). Findings of the third BabyLM Challenge. BabyLM Workshop 2025. https://aclanthology.org/2025.babylm-main.28/
48. Shinnick, Z., Jiang, L., Saratchandran, H., van den Hengel, A., and Teney, D. (2025). Transformers pretrained on procedural data contain modular structures for algorithmic reasoning. https://arxiv.org/abs/2505.22308
49. Krishna, K., Bigham, J., and Lipton, Z. C. (2021). Does pretraining for summarization require knowledge transfer? Findings of EMNLP 2021. https://aclanthology.org/2021.findings-emnlp.273.pdf
50. Wong, L., Grand, G., Lew, A. K., Goodman, N. D., Mansinghka, V. K., Andreas, J., and Tenenbaum, J. B. (2023). From word models to world models: translating from natural language to the probabilistic language of thought. https://arxiv.org/abs/2306.12672
51. Trinh, T. H., Wu, Y., Le, Q. V., He, H., and Luong, T. (2024). Solving olympiad geometry without human demonstrations. Nature. https://www.nature.com/articles/s41586-023-06747-5
52. Romera-Paredes, B., et al. (2024). Mathematical discoveries from program search with large language models. Nature. https://www.nature.com/articles/s41586-023-06924-6
53. Gao, L., Madaan, A., Zhou, S., et al. (2023). PAL: program-aided language models. ICML 2023. https://proceedings.mlr.press/v202/gao23f.html
54. Pan, L., Albalak, A., Wang, X., and Wang, W. (2023). Logic-LM: empowering large language models with symbolic solvers for faithful logical reasoning. Findings of EMNLP 2023. https://aclanthology.org/2023.findings-emnlp.248/
55. Huebner, P. A., Sulem, E., Fisher, C., and Roth, D. (2021). BabyBERTa: learning more grammar with small-scale child-directed language. CoNLL 2021. https://aclanthology.org/2021.conll-1.49/
56. Ivanova, A. A., Sathe, A., Lipkin, B., et al. (2025). Elements of World Knowledge (EWoK). TACL 13, 1245 to 1270. https://aclanthology.org/2025.tacl-1.57/ ; https://arxiv.org/abs/2405.09605
57. BabyLM (2025). Evaluation pipeline 2025, task list and baseline results. https://github.com/babylm/evaluation-pipeline-2025
58. Tan, A. W. M., et al. (2024). DevBench: a multimodal developmental benchmark for language learning. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.10215
59. Jiang, G., Xu, M., Xin, S., Liang, W., Peng, Y., Zhang, C., and Zhu, Y. (2023). MEWL: few-shot multimodal word learning with referential uncertainty. ICML 2023. https://proceedings.mlr.press/v202/jiang23i.html
60. Levandovsky, E., Manaseryan, A., and Kennington, C. (2025). Learning to speak like a child: reinforcing and evaluating a child-level generative language model. SIGDIAL 2025. https://aclanthology.org/anthology-files/pdf/sigdial/2025.sigdial-1.30.pdf
61. Poh, W., Tombolini, M., and Barak, L. (2025). What did you say? Generating child-directed speech questions to train LLMs. Venue UNVERIFIED (likely BabyLM 2025). https://exa.ai/library/publication/d3k53wzb0nn
62. Hu, J., Tan, A. W. M., Feng, S. Y., and Frank, M. C. (2025). Language production is harder than comprehension for children and language models. CogSci 2025. Listed at https://langcog.stanford.edu/publications ; paper not fetched.
63. Feng, S. Y., et al. (2024). Is child-directed speech effective training data for language models? EMNLP 2024. https://aclanthology.org/2024.emnlp-main.1231.pdf ; https://arxiv.org/abs/2408.03617
64. Vong, W. K., Wang, W., Orhan, A. E., and Lake, B. M. (2024). Grounded language acquisition through the eyes and ears of a single child. Science. https://www.science.org/doi/10.1126/science.adi1374
65. This repository: IMPROVEMENT-PLAN.md, sections B.6 and G.3 (measured single-thread training rate and extrapolation), /Volumes/XS1000/acoliver/projects/bigbrane/IMPROVEMENT-PLAN.md
