# Test the value and limits of a small predictive state

Research record: 2026-09-30. No experiments have been run. The workspace was empty when inspected. This document records a bounded literature investigation; [EXPERIMENT.md](EXPERIMENT.md) specifies the first study.

## What the first study can establish

A hidden Markov model supplies a mathematically explicit baseline for learning a compact predictive state. Exact filtering, learning hidden transition matrices, and replacing latent-state coordinates with observable predictions all have established precedents. A useful first result would measure when this structural prior helps relative to small learned recurrent and attention controls, then measure its cost under a controlled mismatch.

The question unresolved in this study is:

> Across independently sampled generators, does a learned four-state HMM reach a fixed next-token prediction error with fewer observed training tokens than a tiny ordinary RNN and a one-layer, one-head causal attention model? If it does, how much of that advantage survives when the generator depends on two previous latent states but the fitted HMM retains four states?

This is an empirical question about the specified generators, learners, and training budgets. The literature accessed here does not settle that exact comparison. We have not established that this comparison is novel, and have not searched exhaustively for related benchmarks. An HMM victory on HMM data would be evidence about a disclosed prior, rather than evidence of a new learning mechanism or of general language competence.

The working hypothesis is that a correctly specified small latent-state family reduces estimation burden at low sample sizes. Its advantage may shrink or reverse when its state representation cannot retain the required history. EM optimization failures and weak neural training can also determine the outcome. The experiment separates these possibilities as far as its small scope permits.

## 1. Bayesian filtering and learning a generator are separate problems

For hidden state z_t in {1,...,K}, symbol x_t in {0,...,V-1}, row-stochastic transition matrix A, emission matrix B, and initial distribution pi:

\[
P(z_{1:L},x_{1:L})=\pi_{z_1}B_{z_1,x_1}
\prod_{t=2}^{L} A_{z_{t-1},z_t}B_{z_t,x_t}.
\]

Let q_t(i)=P(z_t=i | x_{1:t-1}) be the prior state distribution before observing token t. Then

\[
p_t(v)=\sum_i q_t(i)B_{iv},\qquad
b_t(i)=\frac{q_t(i)B_{i,x_t}}{p_t(x_t)},\qquad
q_{t+1}(j)=\sum_i b_t(i)A_{ij},\quad q_1=\pi.
\]

The posterior b_t is a sufficient statistic for predicting the future under the assumed generator. Dense filtering costs O(K^2+KV) per token if all V prediction probabilities are requested, with O(K) changing state memory. For fixed K and V, sequence processing is linear in L. The learned matrices require O(K^2+KV) storage. This result says nothing about how much data is needed to estimate those matrices, or whether K can remain small as a task becomes more complex.

Baum, Petrie, Soules, and Weiss [S1] give the likelihood reestimation foundation. Rabiner's author-hosted tutorial [S2] supplies the forward-backward procedure, practical scaling, initialization, and multiple-sequence treatment. These establish how to implement a baseline without inventing an architecture. They also identify local optimization as a limitation.

In Baum-Welch, forward-backward smoothing estimates expected state occupancies and transition counts using an entire training sequence. The M-step updates A, B, and pi from those counts. Looking ahead within a training example is legitimate for fitting a generative model. At validation and test time, each next-token probability must use only its prefix. Smoothing a test sequence would leak future observations.

The true-matrix oracle in the study evaluates the irreducible prediction uncertainty and implementation accuracy. It has privileged information. Only the learned HMM, RNN, and attention model are competitors.

## 2. System identification and predictive-state representations already connect these fields

Littman, Sutton, and Singh [S3] represent state by probabilities of selected future action-observation tests. Their linear predictive-state representation has dimension no greater than the number of states in the corresponding minimal finite POMDP representation. The representation result does not establish that any such state is easy to discover from a small dataset. The paper distinguishes proposed learning benefits from demonstrated representation properties.

Without actions, a useful schematic predictive state is

\[
s_t(r)=P(X_{t:t+\ell_r-1}=u_r\mid X_{1:t-1}),
\]

for chosen future symbol strings u_r. A sufficient collection of these predictions can determine other future probabilities. This formulation grounds the state in observable events rather than requiring interpretation of each hidden-state coordinate.

Hsu, Kakade, and Zhang [S4] connect observable operators to subspace identification. They estimate observation singleton, pair, and triple statistics, use an SVD to find a low-dimensional subspace, and learn operators for sequence probabilities. Their guarantees require rank and conditioning assumptions on the HMM and further conditions for conditional-probability guarantees. Their algorithm need not recover the original transition and emission matrices.

Three restrictions matter for this project:

1. Polynomial sample complexity under stated assumptions is not a universal promise of fewer samples than a neural learner. Small singular values can make learning difficult.
2. Their simple analysis samples independent sequence beginnings. Counting many correlated positions as independent examples would misrepresent that result. Our study counts tokens and independent sequences separately.
3. Finite-sample observable-operator estimates need not behave as a directly parameterized positive stochastic HMM without additional treatment. Such treatment must be specified and assessed if spectral learning is added later.

A spectral or predictive-state competitor could later help distinguish representational benefit from EM's optimization behavior. It is omitted from the first harness to keep the attribution understandable. An EM failure alone cannot reject compact predictive-state representations.

## 3. Information bottleneck specifies relevance, but does not supply free sample efficiency

Tishby, Pereira, and Bialek [S5] formulate a compressed representation S of an input H that preserves information about a relevance variable Y:

\[
\min_{P(S\mid H)} I(S;H)-\beta I(S;Y).
\]

For sequence learning, H could be a prefix and Y a future block. This makes the compression objective depend on a chosen prediction task. Their formulation assumes access to the relevant joint distribution; estimating it from limited data remains a separate problem. Their optimization equations do not constitute a demonstrated small-sample sequence learner for our setting.

A project hypothesis, not an established result from this investigation, is that compressing prefixes according to future predictive distributions could remove nuisance variation while retaining transferable state. A next-token-only relevance variable can miss information needed for a distant retrieval request or a multi-step composition problem. Future-block choice and horizon would therefore be experimental design choices, not harmless implementation details.

Information bottleneck is also different from parameter counting. A state with few real-valued coordinates does not automatically have a short finite-precision code, low mutual information, or low training cost. The first harness reports parameters, changing state storage, prediction error, and measured resources separately. It does not estimate mutual information in neural hidden states.

## 4. MDL is a route to choosing complexity, with unresolved coding choices

Rissanen [S6] proposes choosing model structure and parameters by minimizing the description length of model and data. A schematic two-part criterion is

\[
\mathcal{C}(M,D)=L(M)+L(D\mid M),\qquad
L(D\mid M)=-\sum_n \log_2 P_M(x^{(n)}).
\]

A future study might choose state count using an explicit code for K, quantized parameters, and data, or use a specified prequential code. The code must account for the training or selection procedure and precision assumptions. Simply multiplying the number of stored floats by a bit width is a storage report, not a justified MDL selection rule.

The first experiment fixes K=4 for the learned HMM and discloses that the matched generators really have four latent states. It does not learn K. This prior information can account for much of its benefit. Adaptive state complexity is a later hypothesis: can a training-data-only complexity rule preserve the matched advantage while expanding under mismatch? No adaptive state-selection method is proposed as a new result here.

## 5. Physics motivates hierarchical summaries, but its assumptions need testing

Koch-Janusz and Ringel [S7] maximize information between a coarse variable extracted from a spatial block and its separated environment, using a buffer to suppress short-range correlations. They iterate coarse-graining and demonstrate results on classical statistical-physics systems, including Ising and dimer models. These are concrete results in their domains.

A sequence analogue would compress a temporal block to preserve information about a separated future interval, then repeat at larger scales. The project conjecture is that summaries at distinct scales might isolate slow predictive variables and discard local noise. This investigation supplies no evidence that doing so reduces the sample requirement or compute cost of general sequence learning.

Spatial equilibrium examples do not establish temporal causality, compositional closure, or efficient exact retrieval. A hierarchy can erase details needed later. If a task requires retaining m independent uniformly random values from an alphabet of size V for exact later retrieval, its memory requirement grows with m log_2 V under those assumptions. A fixed-dimensional representation with unrestricted real precision is not an escape from a finite-precision storage accounting.

Temporal coarse-graining would also need to test whether a block summary is closed under the next update: omitted history can make the coarse process non-Markovian. That failure motivates the second-order mismatch stress test in the first experiment, although the first experiment contains no hierarchy.

## Controls and boundaries

Elman's recurrent model [S8] is precedent for training an ordinary recurrent hidden state from temporal data. Vaswani and colleagues [S9] specify scaled dot-product attention, causal masking, and positional encoding, and analyze its sequence-length cost. Our controls are deliberately small variants with fully specified equations. They do not reproduce the original papers' architectures or results.

The initial research program also included arbitrary retrieval and composition. Those remain outside this first experiment. Passing latent-state prediction would not imply passing either one. A later retrieval study should count stored independent values and finite-precision memory; a later composition study should hold out specified combinations rather than merely new random sequences. Neither should reuse a successful HMM result as evidence of transfer.

## Decision path

[EXPERIMENT.md](EXPERIMENT.md) fixes the generators, data sizes, oracle, candidates, model searches, prefix alignment, metrics, and decision thresholds before implementation.

If the learned HMM meets the matched advantage criterion, run the already specified second-order stress test and report its result regardless of direction. Loss of the advantage would bound the usefulness of the structural prior and motivate a separate adaptive-complexity study. Survival would justify examining further mismatches, with new preregistration, rather than claiming generality.

If the HMM does not meet the criterion, report the negative or inconclusive result. An oracle test can detect a broken evaluator; restart and training traces can suggest optimization limitations. Those diagnostics do not authorize changing the generator, widening models, or increasing budgets until a preferred result appears. Any changed protocol belongs to a separately labeled experiment.

## Sources accessed and access limits

All entries below were accessed during this investigation on 2026-09-30. A publisher page or abstract is distinguished from readable full text. Search-index extracts came from the linked primary papers, rather than from a secondary description. Some direct PDF fetches returned PDF bytes without readable extraction; those fetches are not counted as full-text reading.

| ID | Primary source and accessed URL | Access and basis used here |
| --- | --- | --- |
| S1 | Baum, Petrie, Soules, Weiss (1970), *A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains*. [Project Euclid](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-41/issue-1/A-Maximization-Technique-Occurring-in-the-Statistical-Analysis-of-Probabilistic/10.1214/aoms/1177697196.full), DOI 10.1214/aoms/1177697196. | Direct publisher page verified authors, date, and citation; full text was subscription-restricted. The reestimation mechanism is supported by S2 and S4, not a claimed full reading of S1. Publisher identifies volume 41, issue 1, pages 164-171; one discovery index had an inconsistent issue field. |
| S2 | Rabiner (1989), *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. [Author-hosted paper](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf); also [indexed paper copy](https://www.cs.umb.edu/~rvetro/vetroBioComp/HMM/Rabiner1989%20A%20Tutorial%20on%20Hidden%20Markov%20Models%20and%20Selected%20Applications%20in%20Speech%20Recognition.pdf). | Live search returned primary-paper extracts including forward initialization, backward probabilities, reestimation, local maxima, and implementation topics. This is a methodological tutorial, not a claim of original discovery of filtering. |
| S3 | Littman, Sutton, Singh (NIPS 2001), *Predictive Representations of State*. [Proceedings page](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html), [paper](https://papers.nips.cc/paper_files/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf). | Direct proceedings page provided abstract and introduction; live indexed PDF extracts supplied the representation result, sufficient-test discussion, and conclusion's distinction between results and prospective advantages. Direct PDF extraction was unreadable. The PDF metadata and indexed text include Singh, omitted from the page's displayed author list. |
| S4 | Hsu, Kakade, Zhang (COLT 2009; journal version 2012), *A Spectral Algorithm for Learning Hidden Markov Models*. [Readable author manuscript](https://arxiv.org/html/0811.4413), [COLT paper](https://www.learningtheory.org/colt2009/papers/011.pdf). | Direct HTML gave readable sections on model assumptions, independent-sequence analysis, rank conditions, observable operators, and learning limitations. Live COLT PDF extracts also supplied the results summary. Direct PDF fetch alone was unreadable. The fetched HTML is version 6, dated 2012, so details attributed here use that version. |
| S5 | Tishby, Pereira, Bialek (1999 manuscript; arXiv 2000), *The Information Bottleneck Method*. [Readable manuscript](https://arxiv.org/html/physics/0004057). | Direct readable text, including relevance assumptions, objective in section 3.1, and the explicit caveat about obtaining the joint distribution from samples. |
| S6 | Rissanen (1978), *Modeling by Shortest Data Description*. [IBM Research publication record](https://research.ibm.com/publications/modeling-by-shortest-data-description), [publisher record](https://www.sciencedirect.com/science/article/abs/pii/0005109878900055), DOI 10.1016/0005-1098(78)90005-5. | Direct author-affiliation record gave abstract; live search gave publisher citation and abstract. Direct publisher fetch returned 403. We use the stated model-and-data description principle, not an unaccessed detailed coding theorem. |
| S7 | Koch-Janusz, Ringel (2018), *Mutual Information, Neural Networks and the Renormalization Group*. [Readable manuscript, version 2](https://arxiv.org/html/1704.06279v2), [abstract and version record](https://arxiv.org/abs/1704.06279). | Direct readable method and validation sections establish spatial-block/environment MI, iterative coarse-graining, buffer rationale, and Ising/dimer examples. No temporal-learning guarantee is inferred. |
| S8 | Elman (1990), *Finding Structure in Time*. [Primary-paper copy](https://papers.baulab.info/papers/Elman-1990.pdf), [publisher](https://onlinelibrary.wiley.com/doi/pdf/10.1207/s15516709cog1402_1). | Live indexed primary-paper extracts described recurrent context units and learning temporal representations. Our tanh control is an Elman-type model, not an exact reproduction. |
| S9 | Vaswani et al. (2017), *Attention Is All You Need*. [Readable manuscript](https://arxiv.org/html/1706.03762). | Direct readable sections 3.2, 3.5, and 4 supplied attention, mask, positional encoding, and complexity. Fetched version 7 is dated 2023; the paper originated in 2017. |

## What remains unknown

There are no measured accuracy, sample-efficiency, memory, or runtime results in this workspace. Dependencies and CPU library behavior have not been inspected because this task is documentation only. Known-state-count selection, a limited hyperparameter search, finite generator replication, and the expressive limits of the tiny controls constrain any eventual result. The source selection leaves later related work unassessed. These restrictions belong with the result, including if it is positive.
