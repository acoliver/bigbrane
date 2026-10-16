# Models converge on familiar ideas; verified search has produced new results

Research note, 12 October 2026. Sources checked through live search or fetch of primary papers and publisher or author pages.

**Summary.** Artificial Hivemind, idea-generation saturation, and controlled post-training studies support concern about convergent proposals [1,5,28,39]. Temperature, random seeds, and distribution prompts can broaden outputs, with validity and task-transfer limits [6,14,15,19,27]. FunSearch and AlphaEvolve provide stronger evidence for new results through executable evaluation and population search [31,32]. For Bigbrane, my recommendation is to prioritize that loop and measure distinct valid mechanism families per budget. The exact OpenAI paper Andrew recalls remains UNVERIFIED; *Why Language Models Hallucinate* is the most plausible OpenAI-authored candidate checked, while Nasr et al.'s outside-authored extraction study is the closest literal match to training-data repetition [4,8,9].

## Question and scope

Andrew reports that 122 of Bigbrane's 196 proposals rediscovered known methods, including delta rules, linear attention, HMMs, and caches. The remaining proposals were mostly small modifications. These figures are project-provided observations, not an independently audited experiment.

This note distinguishes new wording, diversity among proposals, expert-rated novelty, and an executed result that improves on prior methods. None alone establishes the others. Existing evidence supports Andrew's concern about convergent generation, but does not establish that models can only repeat training data.

## 1. Homogenization and regurgitation

### Artificial Hivemind, Jiang et al., 2025

**Claim and evidence.** The authors find both repeated ideas within a model and convergence across models. Infinity-Chat contains 26K real-world open-ended queries; their detailed diversity analysis uses 100 representative queries across more than 70 models, with 25 models detailed in the main paper. Average pairwise cross-model similarity ranges from 0.71 to 0.82. These values are similarity scores, not the fraction of responses that are duplicates. They also collect 31,250 human annotations, with 25 independent annotations per assessed example [1].

**Method.** Mine open-ended queries from WildChat, repeatedly sample model answers, compare within-model and cross-model response similarity, and assess pluralistic human preferences.

**Limitations.** Similarity depends on the representation and prompt set. Closed training pipelines prevent attributing convergence to one cause. Similar ideas across models do not demonstrate verbatim copying from training data. The 26K dataset size must not be confused with the 100-query detailed generation experiment.

**Bigbrane implication, opinion.** Buying diversity by switching model brands may have limited return. Measure mechanism-level overlap across generators rather than assuming model independence.

### LLM-assisted writing, Doshi and Hauser, 2024

**Claim and evidence.** AI assistance improves individual ratings while reducing diversity across writers. A randomized short-story experiment analyzes 293 writers and 3,519 evaluations from 600 evaluators. Access to one AI idea raises rated novelty by 5.4%; access to five raises it by 8.1%. AI-assisted stories are more similar to one another than human-only stories [2].

**Method.** Randomly assign writers to no assistance, one GPT-4 idea, or up to five ideas, then independently rate their eight-sentence stories and compare semantic similarity.

**Limitations.** This is a writing task with nonexpert writers, not invention of mathematical mechanisms. Individual quality and population diversity are different outcomes.

**Bigbrane implication, opinion.** More fluent and individually appealing proposals can still leave the archive covering fewer distinct mechanisms. Track archive diversity separately from proposal quality.

### Plot-level convergence, Xu et al., 2025

**Claim and evidence.** In *Echoes in AI*, GPT-4 and Llama-3 story continuations repeat plot elements within and across models. The study examines 100 stories with about 3,700 segments; human-written plots have higher Sui Generis uniqueness scores [3].

**Method.** Repeated continuations under shared prefixes, with semantic entailment checks for echoed plot elements; human surprise judgments provide partial validation.

**Limitations.** The metric uses GPT-4 entailment judgments. Some source stories may have appeared in training. Plot uniqueness does not measure mathematical novelty.

**Bigbrane implication, opinion.** Compare update equations, state representations, and predictions rather than titles or wording. The corresponding failure is a familiar mechanism described with new terminology.

### Extractable memorization, Nasr et al., 2023 preprint / ICLR 2025

**Claim and evidence.** *Scalable Extraction of Training Data from (Production) Language Models* demonstrates extraction of memorized material from open and closed models. The 2023 version reports over 10,000 verbatim examples from ChatGPT for $200 in queries, and a divergence attack yielding training text 150 times more frequently than ordinary behavior [4]. The ICLR 2025 paper also studies fine-tuning attacks; version-specific attack rates should not be mixed.

**Method.** Induce non-chatbot continuations and match 50-token strings against a nine-terabyte auxiliary web corpus. The later paper adds fine-tuning-based extraction.

**Limitations.** Adversarial extraction measures a privacy failure, not the percentage of ordinary scientific proposals copied from training. The web proxy gives incomplete access to proprietary training data. This is research about OpenAI models by outside authors, not an OpenAI finding that every output is memorized.

**Bigbrane implication, opinion.** Recognizing a known algorithm is sufficient to classify a rediscovery. Avoid claiming literal training-data copying unless there is matching evidence.

## 2. Alignment, typicality, and diversity loss

### RLHF diversity loss, Kirk et al., ICLR 2024

**Claim and evidence.** Controlled comparisons find that RLHF improves absolute out-of-distribution performance but substantially reduces per-input output diversity relative to supervised fine-tuning. Some cross-input metrics also fall. Increasing the KL penalty did not restore diversity in their experiment [5].

**Method.** Compare fine-tuning stages and best-of-N across two base models on summarization and instruction following, using multiple diversity and generalization metrics.

**Limitations.** These tasks and training recipes do not establish a universal creativity penalty for every form of post-training. RLHF versus SFT is the clearest comparison here; it should not be restated as proof that removing safety training improves scientific invention.

**Bigbrane implication, opinion.** Separate generation and instruction-following utility in evaluation. A matched base/SFT/RLHF experiment is more informative than comparing unrelated products marketed as uncensored.

### Verbalized Sampling and typicality bias, Zhang et al., 2025 preprint / ICML 2026

**Claim and evidence.** The authors identify preference for familiar responses as one driver of output concentration. On 6,874 correctness-matched HelpSteer response pairs, estimated typicality coefficients are 0.57 ± 0.07 and 0.65 ± 0.07 under two base-model probability estimates, both with p < 10^-14. Verbalized Sampling increases creative-writing diversity by 1.6 to 2.1 times relative to direct prompting in the reported experiments, without losses on their tested factuality and safety measures [6].

**Method.** Model typicality bias theoretically, test preference datasets, then ask models for multiple responses and their probabilities, rather than one response. Compare distribution prompting with direct and list prompts across creative writing, dialogue simulation, synthetic data, and open-ended questions.

**Limitations.** Correctness matching cannot hold every dimension of utility constant. Model-stated probabilities are generated text, not calibrated access to the decoder distribution. Recovering variety associated with pretraining is different from inventing a new learning rule. Safety results apply to evaluated tasks.

**Bigbrane implication, opinion.** Test distribution prompts as a low-cost intervention: request several distinct mechanism families with estimated prevalence, sample candidates externally, and independently verify novelty and validity. Preserve ordinary safety controls.

### Counterevidence to a blanket post-training penalty

**Claim and evidence.** Padmakumar et al.'s 2025 *Measuring LLM Novelty As The Frontier Of Original And High-Quality Output* finds that post-training improves their combined novelty metric for OLMo and OLMo-2, primarily through quality improvements with similar or slightly improved originality [7].

**Method.** Use open training corpora to measure unseen n-gram fractions, combine them with task-specific LLM-rated quality using a harmonic mean, and evaluate stories, poetry, and creative tool use. Later versions include three model families and SFT/DPO/RLVR-trained models.

**Limitations.** N-gram originality measures text form, not mechanism novelty; quality uses a model judge. Diversity among samples and novelty relative to training are different measurements.

**Bigbrane implication, opinion.** Removing post-training may recover variety while making usable proposals rarer. Compare the number of valid, distinct, non-rediscovered mechanisms per fixed generation budget.

## 3. Which OpenAI work might Andrew mean?

The exact paper remains **UNVERIFIED**. The phrase alone cannot identify it, and the sources checked so far do not support a universal result that language models only repeat training data.

1. **Likely if Andrew remembers a widely discussed OpenAI theoretical paper:** *Why Language Models Hallucinate* (September 2025), with a 2026 Nature version, *Evaluating large language models for accuracy incentivizes hallucinations*. It links next-token errors to classification and explains why accuracy-only grading rewards guessing. Arbitrary, rarely supported facts cannot be inferred from recurring patterns. It does not prove that all useful generation requires memorizing the answer [8,9]. **Bigbrane implication, opinion:** reward an explicit unknown novelty status instead of confident claims of invention.
2. **Closest literal match to training-data repetition:** Nasr et al.'s extraction study [4]. It proves that some training text can be recovered from ChatGPT. Its original preprint is from 2023, with an ICLR 2025 version, and its authors are outside OpenAI. Attribution or year may be the remembered mismatch.
3. **A 2026 OpenAI-hosted candidate about familiar ideas:** *Training novices to think, or giving them LLMs? Evidence from an RCT* (August 2026). In 1,053 first-year undergraduates, ChatGPT Edu increases idea volume and expert-rated performance. Causal-reasoning training produces the clearest gains in within- and between-solution diversity, while GPT alone has no distinguishable effect on between-solution diversity. The task is a merchandising business problem [10]. This supports concern about standard solutions, without demonstrating memorization or lack of mathematical invention. OpenAI hosting alone should not be treated as proof that every author is an OpenAI employee.
4. **Relevant non-OpenAI candidate:** Nagarajan et al.'s *Roll the dice & look before you leap* (ICML 2025) directly studies algorithmic creativity and finds next-token training excessively memorizing on controlled tasks relative to multi-token alternatives [11]. OpenAI models are cited, but the work is not an OpenAI publication.

OpenAI's 2026 *First Proof* report also supplies a boundary on the claim: it reports plausible research-level proof attempts, acknowledges an incorrect attempt for problem 2, and discloses human selection and feedback. It is evidence of a model-plus-human research workflow, with incomplete external verification, rather than a controlled test of autonomous invention [12].

## 4. Sampling interventions: more randomness has a validity cost

### Temperature, Peeperkorn et al., 2024

**Claim and evidence.** In a fixed narrative-generation setting, higher temperature has a weak positive relationship with novelty and a moderate relationship with incoherence; neither cohesion nor typicality changes consistently. Human inter-rater agreement averages alpha = 0.639 [14]. The study provides evidence against treating temperature as a general-purpose creativity control. It also finds some novelty gain, so describing its result as temperature producing only noise would overstate it.

**Method.** Hold model, context, and prompt fixed, generate stories at different temperatures, measure distance from an exemplar, and obtain human ratings across four creativity-related criteria.

**Limitations.** One narrative setting and imperfect rater agreement. It cannot determine the optimum temperature for new mathematical mechanisms or current reasoning models.

**Bigbrane implication, opinion.** Sweep available temperature settings against valid mechanism yield. Keep malformed equations, instability, constraint violations, and rediscovery as separate failure categories. A larger embedding distance alone should not count as progress.

### Min-p, Nguyen et al., 2024 preprint / ICLR 2025

**Claim and evidence.** *Turning Up the Heat* reports improved quality and diversity relative to competing samplers, particularly at higher temperatures, across Mistral and Llama models from 1B to 123B. Benchmarks include GPQA, GSM8K, and AlpacaEval creative writing; human evaluations favor min-p on tested quality and creativity measures [15]. The latest checked version is v8, revised November 2025.

**Method.** Keep tokens whose probability is at least a chosen fraction of the most probable token's probability, renormalize, and sample. The threshold adapts to each conditional distribution. The authors recommend testing a base threshold around 0.05 to 0.1, with higher thresholds sometimes protecting accuracy at a diversity cost. Sampler ordering matters; their described implementation applies temperature before truncation.

**Limitations.** Creative writing and correctness benchmarks do not establish prior-art novelty. Hyperparameters, truncation order, and inference implementation affect outcomes. Very large temperatures do not have a universal validity guarantee. Closed APIs may provide no min-p or logits access.

**Bigbrane implication, opinion.** Test min-p on a controllable local generator, alongside temperature plus nucleus sampling. Compare matched token and execution budgets. Do not infer that an unsupported API parameter changed the sampler.

### Diverse beam search, contrastive decoding, and base-guided decoding

**Claim and evidence.** A 2024 decoding-method comparison finds ordinary beam search and diverse beam search repetitive on open-ended text; stochastic methods perform better on its MAUVE evaluations. Contrastive search and contrastive decoding reduce degeneration, but still incur extra compute [16]. Diverse beam search rewards differences between beam groups; contrastive decoding contrasts a stronger model with a weaker model under a plausibility restriction. Their objectives address output form and degeneration.

A more directly relevant 2025 intervention is *Mind the Gap*: instruction-tuning reduces narrative diversity, with DPO contributing the largest loss in staged OLMo comparisons. Conformative decoding mixes an instruction model with its base counterpart after truncation. The paper reports diversity gains with maintained quality for most tested models, with Llama-3.1-8B an exception on some within-prompt metrics [17].

**Method.** Compare lexical and semantic diversity, distributional quality, and decoder costs. Conformative decoding uses nucleus p = 0.95 and mixing gamma = 0.5 in its main comparison.

**Limitations.** MAUVE, Vendi scores, and lexical spread do not measure new algorithms. Base-guided decoding requires compatible model vocabularies and access to both distributions. It adds inference work. Results are task-dependent.

**Bigbrane implication, opinion.** Prioritize archive-level structural diversity before a decoder that mainly penalizes repeated words. Base-guided decoding is a research arm if local paired checkpoints are available.

### High-temperature brainstorming followed by low-temperature refinement

**Evidence status.** Multi-stage divergent generation and refinement have supporting product-ideation evidence [18], while min-p supports some high-temperature generation [15]. A controlled demonstration that the exact high-temperature-then-low-temperature schedule raises verified mathematical novelty was **UNVERIFIED** in this search.

**Bigbrane implication, opinion.** Separate a broad candidate stage from a constrained formalization stage. Save the original mechanism, its equations, and a mechanism identifier before refinement. Reject a refinement that silently replaces it with a familiar delta rule or attention variant. Test the schedule as an experiment rather than adopting its plausibility as evidence.

## 5. Prompting, random seeds, analogies, and constraints

### Multi-stage prompting and personas, Meincke, Mollick, and Terwiesch, 2024

**Claim and evidence.** In GPT-4 product brainstorming, 35 prompting strategies produce different diversity levels. Human idea pools have cosine similarity around 0.243 versus 0.255 to 0.432 for GPT-4 pools. A multi-stage strategy generates short candidate ideas, checks differences, revises, and expands them; it comes closest to human pool diversity. The authors estimate an opportunity space of about 4,700 ideas versus 3,700 with their base prompt. These are estimates from repetition patterns, not counted scientific inventions. The multi-stage advantage diminishes after roughly 750 to 800 ideas [18].

**Method.** Generate products for college students priced under $50. Evaluate embedding similarity, unique ideas, and exhaustion as generation scales. Personas are among the compared prompts: a Steve Jobs prompt changes similarity from baseline 0.377 to 0.368, while a creativity-tool prompt reaches 0.387.

**Limitations.** Working paper, consumer products, and semantic similarity. Persona effects are mixed and smaller than the headline multi-stage effect. Different role names can still generate the same underlying mechanism.

**Bigbrane implication, opinion.** Assign distinct mathematical obligations to explorers, such as conserved quantities, sparse events, or an explicit memory budget. Measure differences in proposals. Role labels alone provide weak evidence of diversity.

### Random concepts and Andrew's proposed "weird salad", Agrawal and Goyal, 2026

**Claim and evidence.** *Addressing LLM Diversity by Infusing Random Concepts* directly tests unrelated random-word and random-sentence prefixes. On 100 list-question prompts, four models show increases in unique responses and distribution entropy at the 5% significance level. Models include Gemma-3-4B, Mistral Large, Amazon Nova Pro, and Claude-3.5-Sonnet, with temperature fixed at 0.9. The authors explicitly describe relative gains as small in some conditions [19].

**Method.** Externally generate words or sentences, prepend them to repeated list requests such as naming actors, and compare counts and entropy with unmodified prompts. A random-string appendix also finds changes.

**Limitations.** This broadens retrieval of familiar list items. It does not test mathematical invention, correctness guarantees, or large improvements in creative tasks. The authors leave transfer to those tasks as future work.

**Bigbrane implication, opinion.** Random-prefix perturbation has direct list-diversity evidence; transfer to mathematical invention remains untested. Generate seeds externally and record them. Compare meaningless strings, random concepts, and structured cross-domain seeds, rather than mixing them into one condition.

An ICCC 2026 study, *How Context Injection Shapes Creativity in LLM-Generated Alternative Uses*, provides a closer creative-task test. It generates 2,880 responses across four models, six objects, six conditions, and 20 responses per condition. Random context has the highest model-judged novelty, while unrelated context increases categorical flexibility; related or object-focused context improves judged value [20]. Its judges and small object set limit generalization. **Bigbrane implication, opinion:** choose seed types according to the desired structural change, and require independent validity checks.

### Analogical reasoning and cross-domain inspirations

**Claim and evidence.** Yasunaga et al.'s analogical prompting improves reasoning accuracy by prompting self-generated relevant examples and general knowledge before solving a problem. Its evaluated tasks include GSM8K, MATH, Codeforces, and BIG-Bench [21]. It provides evidence for solving new instances using prior structures, rather than a direct test of inventing new methods.

The 2025 *Relevant or Random* study finds random examples can match or exceed relevant examples, including a reported 4% GSM8K boost using random biology examples [22]. Its findings weaken a simple inference that any gain from analogical prompting demonstrates useful analogical transfer.

**Method.** Compare zero-shot and few-shot reasoning with model-generated examples; the follow-up perturbs relevance and checks example correctness.

**Limitations.** Benchmark accuracy and mathematical mechanism novelty differ. Incorrect examples can harm reasoning; domain distance alone does not establish a transferable relationship.

**Bigbrane implication, opinion.** Ask for an explicit mapping of source states, dynamics, and invariants into a learning rule. Require a prediction or update equation that depends on the mapping. A biological metaphor with no mathematical consequence should not count as a distinct mechanism.

### Conceptual blending, Kombine, 2025 preprint / October 2026 revision

**Claim and evidence.** Schapiro et al.'s latest checked version evaluates 35 LLMs and 75 humans on associations, analogies, blends, and invented concepts from unrelated entities. Models are 10 to 16 percentage points better at association and analogy than blending. About one in five invented concepts for a given item reuses properties generated by another model. Reuse is 14 times more likely for blends than analogies and 2.3 times more likely within a provider than across providers. Humans score below nearly every model on the tested tasks, while occupying different semantic regions [23].

**Method.** Structured entity-pair tasks require a shared abstraction, a coherent blend, and a derived invention. The paper measures task success and cross-output property reuse. These numbers are from v6, revised 7 October 2026; they should not be retroactively attributed to v1.

**Limitations.** Under review. Conceptual coherence and task scoring cannot guarantee absence of prior human invention. High per-response scores coexist with pool convergence. Successful blends require finding an abstraction, which is itself a bottleneck.

**Bigbrane implication, opinion.** Test explicit abstraction-first blending: extract a common mathematical relation, instantiate it in a learner, and list what new property follows. Use human-provided or externally sampled conceptual pairs to widen the source pool.

### Literature-conditioned novelty, SciMON and MOOSE-Chem

**Claim and evidence.** SciMON improves rated scientific idea novelty using retrieved inspirations and iterative comparison with prior work, while finding GPT-4 proposals shallow and often incremental [24]. MOOSE-Chem uses inspiration retrieval, multi-step composition, mutation/recombination, and ranking to recover hypotheses from 51 chemistry papers first public in 2024, using a model with a pre-2024 cutoff [25]. Its ablations find more varied mutations improve the best matching hypothesis but can lower average matching quality.

**Method.** SciMON retrieves neighboring ideas, uses them as negative examples when overlap exceeds a similarity threshold, and asks for a changed proposal. MOOSE-Chem compares proposals with held-out published hypotheses and expert assessments.

**Limitations.** SciMON's novelty is relative to an incomplete reference corpus, and embeddings can miss equivalent methods. MOOSE-Chem is explicitly a rediscovery benchmark, even when the answer was unavailable to the model. A declared cutoff provides evidence against direct access, without auditing all proprietary training content. Wet-lab success of new proposals is outside that benchmark.

**Bigbrane implication, opinion.** Run prior-art retrieval after initial generation and return concrete equivalence evidence to the explorer. Separate retrieval for inspiration from retrieval for rejection. Select distant but mathematically relevant sources for the inspiration branch, rather than always feeding the most familiar attention papers.

### Constraints and specification-level injection

**Claim and evidence.** NeoCoder's Denial Prompting progressively prohibits techniques used in earlier Codeforces solutions. On 199 problems with 30 reference human solutions per problem, alternatives become more divergent as restrictions accumulate, but correctness and constraint adherence fall. GPT-4's joint creativity measure reaches zero at the fifth restriction level; MCTS, self-correction, planning, and sampling do not consistently improve the joint measure [26].

A June 2026 preprint, *Where You Inject Diversity Matters*, finds that generating different intermediate specifications and then conditioning responses on them improves diversity with comparable quality across five open-ended tasks and four backbone models [27]. The distinction between selecting a high-level configuration and adding token noise is relevant to generation architecture.

**Method.** NeoCoder requires both correct execution and alternative techniques. Specification-level injection varies configurations before generating final outputs, and measures how much source variation survives.

**Limitations.** Restricting every known mechanism can eliminate valid solutions. Different specifications can collapse during realization. Neither study demonstrates new data-efficient language learners.

**Bigbrane implication, opinion.** Vary a few substantive design commitments per explorer, and check that the final equations honor them. Use structural exclusion for diagnosed duplicates, with a separate unrestricted branch. Avoid a growing list of prohibitions with no surviving feasible search space.

## 6. Research ideas: perceived novelty frequently fails execution

### Si, Yang, and Hashimoto, 2024 / ICLR 2025

**Claim and evidence.** The original study recruits 104 unique NLP researchers: 49 idea writers and 79 reviewers, with 24 participating in both roles. It collects 298 reviews. LLM proposals receive higher novelty ratings than expert proposals, consistently at p < 0.05 under the paper's statistical comparisons, while feasibility is slightly weaker. At scale, 4,000 seed proposals yield only about 200 non-duplicates under the paper's similarity test [28].

**Method.** Retrieve literature, generate candidate ideas with Claude-3.5-Sonnet, deduplicate, rank, expand, and compare anonymized proposals with human ideas. The duplicate criterion is embedding cosine similarity above 0.8 to an earlier same-topic idea. Prior idea titles are already supplied to discourage repetition.

**Limitations.** Expert-rated novelty before execution can reward persuasive framing. The uniqueness count depends on an embedding threshold; it is not an exact count of mathematically equivalent ideas. Selected LLM proposals reflect a larger search budget than one human's single proposal.

**Bigbrane implication, opinion.** Repeatedly requesting "new ideas" and showing old titles may still exhaust the model's repertoire. Plot new valid mechanism families per batch, and alter search conditions when marginal yield falls.

### The 2025 execution follow-up, Si, Hashimoto, and Yang

**Claim and evidence.** Forty-three expert researchers execute randomly assigned human or LLM ideas, and experts review the resulting four-page papers blind. LLM ideas lose significantly more score from ideation to execution on novelty, excitement, effectiveness, and overall rating, p < 0.05 [29]. At the idea-level, effectiveness changes from about 6.00 to 4.13 for AI proposals and 4.83 to 4.78 for human proposals; overall AI rating changes from about 5.38 to 3.41. Post-execution mean rankings favor human proposals, but those final differences are not statistically significant when treating the 43 ideas as independent units.

**Method.** A blinded randomized execution study using proposals from the original comparison. Executors implement experiments and submit code and papers; reviewers judge actual evidence, design, and feasibility.

**Limitations.** Forty-three projects, one research domain, and a bounded implementation period. Review-level and idea-level significance differ; the former must not be used to claim decisive superiority across independent projects. The stronger result is the differential drop after execution.

**Bigbrane implication, opinion.** Add small executable falsification experiments before ranking proposals as inventions. Include faithful known-method baselines, ablations, and a record of whether the intended mechanism survived implementation.

### AI Scientist-v2, 2025

**Claim and evidence.** The authors report one of three submitted papers meeting an ICLR workshop acceptance threshold, with reviews 6, 6, and 7, mean 6.33. The workshop specializes in negative and unexpected results. Humans choose three initial directions from roughly 40 generated ideas, run multiple random seeds, and select the best complete manuscript per direction. No human edits the selected manuscript's code, results, figures, or prose within the autonomous run [30].

**Method.** Agentic tree search explores and debugs experiments, runs the selected research idea, writes a manuscript, and uses feedback/reflection to revise it.

**Limitations.** Workshop evaluation has a different standard from main-track acceptance. Best-of-many human selection prevents estimating an unconditional success rate as one third. The authors acknowledge inconsistent quality and audit problems. Producing a paper and obtaining favorable reviews do not by themselves establish reproducible mechanism novelty.

**Bigbrane implication, opinion.** Borrow execution and branching, while reporting every attempted idea and failed run. Keep experimental evidence and evaluator code outside the generator's control. Do not use workshop acceptance as a substitute for reproduced results.

## 7. Evolutionary and open-ended search with verifiers

### FunSearch, Romera-Paredes et al., December 2023

**Claim and evidence.** FunSearch produces verifiable new cap-set constructions and online bin-packing heuristics. It finds an eight-dimensional cap set of size 512 and improves the reported asymptotic cap-set capacity lower bound to 2.2202 [31]. These are results relative to prior mathematical knowledge, rather than ratings of proposal wording.

**Method.** A frozen pretrained LLM proposes code inside a human-provided program skeleton. An executable evaluator rejects invalid programs and scores valid ones. A program database maintains multiple evolving islands, uses high-performing examples in prompts, and periodically reseeds weaker islands. The search evolves priority functions rather than asking for an entire mathematical theory in one completion.

**Limitations.** Humans define the problem representation and evaluator. Code search requires many proposals and evaluations, and success on constrained combinatorics does not establish general scientific creativity. Numerical and executable checks need to match the mathematical claim.

**Bigbrane implication, opinion.** Evolve executable learning updates inside a fixed interface, preserve multiple lineages, and use measured learning behavior for selection. This has stronger discovery evidence than repeated open-ended brainstorming.

### AlphaEvolve, 2025

**Claim and evidence.** AlphaEvolve discovers a rank-48 algorithm for multiplying 4 by 4 complex-valued matrices, versus rank 49 from recursive Strassen, and improves 14 reported matrix-multiplication targets [32]. The complex-field qualification matters. AlphaTensor's rank-47 result over the two-element field is a different setting, and scalar multiplication count alone does not guarantee a practical speedup for real matrices.

**Method.** Gemini models propose edits to programs or codebases. A population/database, evolutionary selection, and one or more automated evaluators iteratively search for better algorithms. Candidate algorithms and supporting constructions are checked computationally; practical infrastructure changes also undergo deployment-specific validation.

**Limitations.** A white paper from the system's developers, with compute and deployment conditions that differ from Bigbrane. Humans design evaluators and select problem formulations. The discovery claims concern particular algorithms and constructions, with no claim that unconstrained chat routinely invents them.

**Bigbrane implication, opinion.** Pair search with a short feedback loop and multiple objectives: prediction quality, data efficiency, memory, and measured scaling. Keep correctness and complexity requirements independent of the fitness score. Archive code and evaluator output as the result, rather than only a prose proposal.

### Novelty search, Lehman and Stanley, 2011

**Claim and evidence.** Behavioral novelty search outperforms objective-driven evolution on the paper's deceptive maze-navigation and biped-walking tasks [33]. The study shows that a direct target score can remove stepping stones needed to reach the target.

**Method.** Reward distance from neighboring behaviors in the current population and an archive. Behavioral descriptors, rather than genotype or text differences, define novelty.

**Limitations.** Descriptor choice determines what the algorithm explores. An irrelevant descriptor can encourage useless novelty. The result does not support ignoring objectives in every domain or treating random sampling as novelty search.

**Bigbrane implication, opinion.** Preserve learners with different response profiles even when their current aggregate language loss is worse. A learner that handles interference or sparse observations differently may be useful for a later combination.

### Quality-diversity and MAP-Elites, Mouret and Clune, 2015

**Claim and evidence.** MAP-Elites returns multiple high-performing solutions across predefined feature niches, with favorable comparisons in neural-network and soft-robot search domains [34]. The original authors describe the empirical evidence as preliminary.

**Method.** Discretize a low-dimensional behavior/feature space. Each cell stores its best-scoring solution. Mutate or recombine archive members and replace the elite only within the corresponding niche. Improvement in one niche does not automatically erase all other niches.

**Limitations.** Fixed descriptors constrain the repertoire; fixed MAP-Elites alone cannot create indefinitely new niches. Sparse bins and evaluation cost matter. A diverse archive does not establish historical originality of its entries.

**Bigbrane implication, opinion.** Define cells by measurable learner behavior and cost, such as interference sensitivity, memory growth, online adaptability, and performance on long dependencies. Avoid cells defined only by model-invented labels. Compare a quality-diversity archive against a single global leaderboard.

### OMNI and OMNI-EPIC

**Claim and evidence.** OMNI combines learning progress with a foundation-model notion of interestingness, outperforming uniform sampling and learning-progress-only curricula in Crafter, BabyAI, and AI2-THOR settings [35]. OMNI-EPIC extends the approach to generated task/environment/reward code and an archive of successful and failed tasks [36].

**Method.** OMNI reduces selection weight for tasks judged uninteresting. OMNI-EPIC retrieves similar archived tasks, proposes a different task at an appropriate difficulty, generates code, uses a post-generation interestingness check, and records the RL learner's result.

**Limitations.** Interestingness is learned from human data and can favor familiar patterns. OMNI-EPIC explicitly cannot rule out similarity to training environments. Its current simulator limits the task space; unlimited innovation and cross-task agent generalization are not demonstrated.

**Bigbrane implication, opinion.** Condition exploration on an archive of both results and failures, and use a model's interestingness rating as one search-allocation signal. Require an independent mechanism check and executed evidence before admitting a discovery.

### Clune's AI-generating algorithms and Promptbreeder

**Claim and evidence.** Clune's 2019 AI-GA paper proposes three coupled research directions: learning architectures, learning learning algorithms, and generating environments [37]. This is a research agenda, without an experiment proving open-ended general intelligence.

Promptbreeder supplies an executed example of evolving the search procedure: it improves task prompts and the mutation prompts that generate new prompts, and reports benchmark gains over hand-written strategies on arithmetic, commonsense reasoning, and hate-speech classification [38].

**Method.** Promptbreeder uses an LLM mutation operator, population selection through fitness tournaments, and repeated evaluation, with frozen model weights.

**Limitations.** Benchmark prompt optimization differs from discovery of a learning mechanism. Fitness evaluated repeatedly on the same task distribution can overfit. Evolving mutation prompts does not remove the need for a trustworthy evaluator.

**Bigbrane implication, opinion.** Evolve explorer instructions using held-out valid-novel-mechanism yield. AI-GA suggests co-designing task distributions and update rules, while keeping final evaluation tasks hidden from both generators.

### What these systems have in common

**Synthesis, opinion based on [31-38].** The evidenced mechanisms combine proposal generation with external feedback, repeated search, persistent populations or archives, and an explicit way to keep exploration alive. FunSearch and AlphaEvolve provide the clearest examples of verified new mathematical results. Novelty search and quality-diversity explain why a single winner-take-all score may lose useful branches. OMNI provides a route for selecting the next problem, with weaker evidence for historical novelty. None isolates a universal causal contribution from every component; the full discovery systems and component studies support different parts of this recommendation.

## 8. Base models and less post-trained generators

### Matched base/aligned comparisons, West and Potts, 2025

**Claim and evidence.** *Base Models Beat Aligned Models at Randomness and Creativity* compares Llama-3.1 base models with SFT, DPO, Tulu, and instruction variants on random-number generation, mixed-strategy games, and poetry. Aligned variants exhibit more concentrated patterns. Human poetry assessments favor base outputs on originality and aligned outputs on pleasantness [39]. These findings add task-specific evidence to Kirk et al. [5] and *Mind the Gap* [17].

A human co-writing experiment gives a related contrast: Padmakumar and He compare solo writing, GPT-3 assistance, and InstructGPT assistance. Key-point Rouge-L homogenization is 0.1536, 0.1578, and 0.1660 respectively. InstructGPT increases similarity at p < 0.05; GPT-3 does not significantly differ from solo writing. The authors attribute the reduction chiefly to model-contributed text [13].

**Method.** West and Potts use matched model families and several unpredictability tasks, with human judgments for poetry. Padmakumar and He run controlled argumentative-essay co-writing and separate user-contributed from model-contributed key points.

**Limitations.** Unpredictability, poetry originality, and mathematical invention differ. Game evaluation has model probability-estimation assumptions; human creative judgments are noisy. Base models also inherit pretraining biases. Marketing a model as less aligned supplies no controlled evidence about its instruction quality or scientific novelty.

**Bigbrane implication, opinion.** Use a matched base checkpoint as one generator arm, with demonstrations of the desired technical output. Keep a separate instruction-tuned formalizer and a fixed validator. Compare useful novel yield, not only entropy. Ordinary execution isolation, privacy handling, and content safety remain required. The cited results provide no reason to disable those controls.

### A 2026 measurement warning

**Claim and evidence.** *Death of the Novel(ty)* uses 8,618 expert-writer annotations. Approximately 91% of expressions in the highest n-gram-novelty quartile are not judged creative. Greater n-gram novelty in open models is associated with lower pragmaticality [40]. Together with the originality-quality frontier results [7], this prevents a blanket conclusion that a more varied base model is the best usable generator.

**Method.** Expert close-reading annotations assess novelty, sensicality, and pragmatic suitability, then compare with text-overlap metrics and model judgments.

**Limitations.** Writing rather than mathematics. Associations do not determine which generator wins under Bigbrane's constraints.

**Bigbrane implication, opinion.** Random text, unfamiliar notation, and unusual metaphors must earn no novelty reward without a defined, valid mechanism.

## 9. Work in 2025 to 2026 on exploration beyond familiar outputs

### Algorithmic creativity and input seed-conditioning, Nagarajan et al., ICML 2025

**Claim and evidence.** On controlled graph-discovery and pattern-construction tasks, multi-token/teacherless training and diffusion can produce a larger fraction of coherent, original, unique outputs than next-token training. The paper also finds input seed-conditioning comparable to temperature sampling, and sometimes better [11].

**Method.** Train and test with random prefix strings paired with otherwise identical task definitions. At inference, new seeds induce different coherent outputs even with greedy decoding. The tasks permit direct tests of whether outputs were in the training set and whether their structure is valid.

**Limitations.** Models are trained with the seed-conditioning scheme; adding arbitrary strings to a production API is a different intervention. Toy tasks isolate a mechanism but do not establish an architectural impossibility theorem for next-token learners. Teacherless optimization can be difficult, with mixed small-model outcomes.

**Bigbrane implication, opinion.** Distinguish a no-training random-prefix experiment [19] from a trained seed-conditioned generator. The latter is a candidate for local generator training if the inference-only interventions fail.

### Diversity-aware RL: DARLING and uniqueness-aware rewards

**Claim and evidence.** DARLING jointly rewards quality and semantic diversity. Its math comparison reports average pass@1 gains over GRPO of 3.51 and 1.90 percentage points for its 4B and 14B settings, and pass@128 gains of 7.62 and 10.16 points across four benchmarks [41]. It also reports improvements on instruction-following and writing measures.

*Rewarding the Rare*, ACL Findings 2026, reports improvements in pass@k and area under the pass@k curve without sacrificing pass@1 across math, physics, and medical reasoning [42]. It rewards correct rollouts with rare high-level solution strategies. No exact effect-size table from that paper was extracted for this note.

**Method.** DARLING partitions same-prompt rollouts with a learned semantic classifier and multiplies normalized diversity with quality reward. Uniqueness-aware RL clusters complete solution strategies with an LLM judge and reweights advantages inversely with cluster size.

**Limitations.** Learned cluster labels can confuse wording variation with new strategy. More ways to solve known questions do not establish new scientific mechanisms. Benchmark pass@k measures the chance of a correct solution among samples; it does not directly count distinct correct strategies or prove access beyond pretraining.

**Bigbrane implication, opinion.** Reward verified useful members of underrepresented mechanism families, with a separate prior-art signal. Evaluate the classifier using renamed, reparameterized, and algebraically equivalent known rules. A correctness floor should accompany diversity pressure.

### Representation-based exploration, 2025 preprint / ICLR 2026

**Claim and evidence.** Tuyls et al. add exploration bonuses based on pretrained hidden-state representations at inference and during RL. The July 2026 version reports that its Qwen-2.5-7B-Instruct reaches the same AIME 2024 pass rate with 80 samples that GRPO needs 256 samples to reach, about threefold sample efficiency. For Qwen-2.5-14B-Instruct, the authors report more than 50% verifier-efficiency improvements on almost all tested tasks [43].

**Method.** Use a representation-based novelty/exploration bonus to choose diverse rollouts, retaining task verification for correctness. Compare test-time allocation and post-training against ordinary sampling and RL.

**Limitations.** Hidden-state access is unavailable on many hosted APIs. Representation novelty is relative to explored behavior, not the scientific literature. Efficiency is within tested reasoning settings and evaluator assumptions.

**Bigbrane implication, opinion.** Prototype an archive-based exploration bonus at proposal level first. If training locally, representation-based rewards offer a more supported alternative to increasing token entropy indiscriminately.

### Self-play with execution: Absolute Zero and Darwin Gödel Machine

**Claim and evidence.** Absolute Zero Reasoner generates and solves code reasoning tasks with executor feedback. Its 7B coder setting reports an average benchmark improvement of 1.8 percentage points over its compared zero-setting baselines [44]. "Zero data" concerns the added reasoning-training stage; it starts with pretrained models.

Darwin Gödel Machine evolves coding-agent implementations with frozen foundation models. The reported SWE-bench score rises from 20.0% to 50.0%, and Polyglot from 14.2% to 30.7%, with an archive contributing relative to the paper's no-open-ended-exploration baseline [45].

**Method.** Absolute Zero rewards learning progress in task proposal and execution-verified correctness in solving. DGM selects archive members, modifies their code, evaluates them, and keeps functional variants as potential stepping stones.

**Limitations.** Neither experiment shows a foundation model trained from nothing. DGM improves agent code and workflow rather than model weights. Benchmark gains and task self-generation are evidence of learning/search, without proving new-to-humanity learning rules. Self-play guided solely by majority opinion would lack the executor's independent grounding.

**Bigbrane implication, opinion.** Explore generator–solver interaction using synthetic tasks with hidden executable answers. Prevent the proposer from changing the verifier or seeing final held-out tests. Keep the archive and include underexplored parents rather than repeatedly mutating only the last winner.

### Test-time scientific search: Co-Scientist and InfRL

**Claim and evidence.** The 2025 Co-Scientist preprint uses generation, debate, ranking tournaments, and evolution. Its 2026 Nature version reports experimental validation of proposed acute-myeloid-leukemia drug candidates and combinations [46]. A separate 2025 Cell report says the system's highest-ranked bacterial gene-transfer hypothesis matches a mechanism already experimentally established by the human investigators but unpublished when the model was tested [47].

**Method.** Tool-assisted literature search and asynchronous, multi-agent tournament refinement produce ranked hypotheses. Human investigators select and experimentally test candidates; the bacterial test uses a withheld experimental result as an answer.

**Limitations.** Internal Elo scores are relative model judgments. In vitro activity does not establish clinical efficacy. The bacterial case supports inference of a withheld mechanism, with human prior discovery, rather than first discovery by the system. Limited cases cannot supply a general breakthrough success rate.

**Bigbrane implication, opinion.** Use debate to expose missing assumptions and experiment designs. Final selection should rely on external measurements and prior-art examination. Agreement among agents should not count as verification.

InfRL, KDD 2026, offers a closer idea-optimization study across 500 papers and five domains. It uses three frozen agents to propose changes, adapt feedback, and assign comparative rewards across ten iterations. For GPT-4o, reported normalized novelty rankings rise from 0.737 for single-pass and 0.941 for self-reflection to 0.978; feasibility rankings rise from 0.116 and 0.381 to 0.648 [48]. These are relative ranking scores, not probabilities of real-world discovery. Human ratings support the trends, but the ideas are not validated through full experimental execution. **Bigbrane implication, opinion:** test feedback-conditioned search as an alternative to identical repeated prompts, and replace its final judged reward with execution whenever possible.

## Unresolved claims and scope limits

The exact OpenAI paper Andrew recalls remains **UNVERIFIED**. The hallucination paper is the most plausible OpenAI-authored candidate checked; the extraction paper best matches literal repetition, with different authorship and initial year. No checked OpenAI publication proves that all model-generated ideas reproduce training data. A universal claim of that form remains **UNVERIFIED** and conflicts with the verified discoveries reported by FunSearch and AlphaEvolve [31,32].

The following transfers remain **UNVERIFIED**: random concept injection producing historically new mathematical mechanisms; the exact high-temperature/low-temperature schedule improving valid novel yield; removing safety alignment increasing that yield; and a model's self-reported probability accurately measuring scientific novelty. Newer papers reporting diversity, benchmark scores, or judged novelty have been treated according to those outcomes, without promoting them into proof of independent scientific discovery.

This is a targeted literature note, not a systematic review or independent replication. Sources were checked through live searches or fetches of primary abstracts, articles, papers, and author/publisher pages. Some full texts were accessible only through search extracts; effect sizes are omitted where their interpretation could not be checked. No Bigbrane code, model, or experiment was changed.

## Implications for the Bigbrane generator

**The ranking and mechanisms below are opinion, grounded in the cited results.** Evidence strength refers to the source task. Every transfer to small, data-efficient, sub-quadratic language learners still needs evaluation.

### Ranked mechanisms

1. **Executable population search with independent evaluation. Strongest discovery evidence: FunSearch and AlphaEvolve [31,32].** Have explorers generate exact state, prediction, and update functions inside a fixed interface. Use cheap validity and scaling checks before training. Evaluate a population repeatedly, feed back observed failures and scores, and preserve multiple lineages. Hold data, parameter count, training-token count, and inference budget fixed against known-method baselines.
2. **A quality-diversity archive and mechanism-aware novelty test. Strong behavioral-search evidence, indirect evidence for Bigbrane [33-36,45].** Maintain elites in different measured niches and keep selected stepping stones. Normalize variable names and compare algebraic forms, update dependencies, and behavior against the 122 known rediscoveries. Store both archive novelty and literature novelty. Embeddings can retrieve candidates for comparison but should not decide equivalence alone.
3. **Prior-art feedback coupled to falsification experiments. Direct idea-generation and execution evidence [24,28,29].** Return the closest known update rule with an explicit equivalence argument. Require a distinguishing consequence, and test it. Use labels such as known method, incremental variant, distinct but unvalidated, and literature novelty unresolved. Retain useful incremental variants without counting them as new mechanism families.
4. **Distribution prompting and diverse intermediate specifications. Direct diversity evidence, no verified learner discovery [6,27].** Compare ordinary one-answer prompting with requests for multiple mechanisms and their estimated probabilities, and with separately generated structural specifications. Choose candidates with an external sampler. Make specifications differ in update locality, memory organization, invariants, or adaptation timescale, and check realization against the final equations.
5. **Quality-gated novelty rewards or archive-aware search allocation. Reasoning and training evidence [41-44].** Favor valid proposals in underrepresented structural or behavioral clusters. Sample parents partly by uncertainty and underexploration. Do not reward nonsense or duplicate equations with new names. For local training, compare semantic-cluster rewards and representation bonuses with entropy-only exploration.
6. **Externally supplied distant concepts, abstraction-first analogy, and random-prefix controls. Direct diversity evidence with limited transfer [19-25].** Treat meaningless strings, random words, unrelated domain concepts, and structured analogies as separate arms. Ask for a mathematical mapping and a concrete prediction. Save seeds and source material. Measure how often the intervention changes the mechanism rather than the explanation.
7. **Temperature and min-p sweeps, with a preserved-mechanism formalization stage. Sampler evidence, weaker mechanism-novelty evidence [14,15,18].** Compare supported temperatures and truncation settings at a fixed budget. Record whether losses come from invalidity or rediscovery. Formalization must preserve the candidate update, and should report inability to make it work rather than substituting a known method.
8. **Matched base-model exploration and base-guided decoding. Direct unpredictability/writing evidence, mixed usable-novelty evidence [5,7,13,17,39,40].** Use paired checkpoints from one family and equal example support. Keep base generation separate from formatting and validation. A model sold as uncensored and a familiar chat model are a confounded comparison. Safety removal is unsupported as a scientific-creativity intervention.

### A test that separates the interventions

**Experimental design, opinion.** Start with the existing prompt and its actual generator settings as the baseline. Compare individual arms for distribution prompting, intermediate specifications, unrelated seed concepts, random-prefix noise, and sampler changes. Include a matched base arm if it is available. After measuring single interventions, combine those that improve valid distinct yield. Use repeated randomized runs, a fixed generation-token budget, and a separately reported evaluator/training budget. Multi-stage methods consume extra tokens and must be charged for them.

For every candidate, record the original mechanism, formalized equations, code, source seed, model/checkpoint, accessible sampling controls, closest prior methods, complexity assumptions, and execution outcomes. Test sub-quadratic behavior against sequence length while holding other dimensions explicit; a fixed-size hidden state can move cost into its dimension. Evaluate sample efficiency at equal training data and model capacity. These checks are proposed requirements for Bigbrane's stated goal, rather than findings that any cited creativity intervention already satisfies them.

The primary outcome should be **distinct, valid, non-rediscovered mechanism families per fixed budget**. Report a second outcome for families that also improve a held-out learner metric. Track raw diversity, validity, duplicate/rediscovery rate, unresolved prior-art status, and within-refinement mechanism changes separately. Blind human adjudication to the generation arm where practical, and calibrate automated equivalence checks on known duplicates and known distinct methods.

Use one fixed held-out evaluation suite that generators cannot edit, plus a separate development suite for search. Preserve failed attempts and the count of all seeds and submissions. FunSearch-style feedback, the execution gap, and AI Scientist's best-of-many selection make these accounting details necessary to interpret a claimed improvement [29-32].

## Numbered references

1. Jiang et al. *Artificial Hivemind* (2025). https://arxiv.org/abs/2510.22954 ; https://proceedings.neurips.cc/paper_files/paper/2025/file/754d5a526a5ee5a47220664a0eb92751-Paper-Datasets_and_Benchmarks_Track.pdf
2. Doshi and Hauser. *Generative AI enhances individual creativity but reduces the collective diversity of novel content* (2024). https://www.science.org/doi/10.1126/sciadv.adn5290 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11244532/
3. Xu et al. *Echoes in AI: Quantifying lack of plot diversity in LLM outputs* (2025). https://www.pnas.org/doi/10.1073/pnas.2504966122
4. Nasr et al. *Scalable Extraction of Training Data from (Production) Language Models* (2023 / 2025). https://arxiv.org/abs/2311.17035 ; https://proceedings.iclr.cc/paper_files/paper/2025/file/cce0e917b050208170151f77b497fc71-Paper-Conference.pdf
5. Kirk et al. *Understanding the Effects of RLHF on LLM Generalisation and Diversity* (2024). https://proceedings.iclr.cc/paper_files/paper/2024/file/5a68d05006d5b05dd9463dd9c0219db0-Paper-Conference.pdf
6. Zhang et al. *Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity* (2025 / 2026). https://arxiv.org/abs/2510.01171 ; https://proceedings.mlr.press/v306/zhang26hy.html
7. Padmakumar et al. *Measuring LLM Novelty As The Frontier Of Original And High-Quality Output* (2025; initially titled *Beyond Memorization*). https://arxiv.org/html/2504.09389
8. Kalai et al. *Why Language Models Hallucinate* (2025). https://openai.com/index/why-language-models-hallucinate/ ; https://arxiv.org/abs/2509.04664
9. Kalai et al. *Evaluating large language models for accuracy incentivizes hallucinations* (2026). https://www.nature.com/articles/s41586-026-10549-w
10. *Training novices to think, or giving them LLMs? Evidence from an RCT* (August 2026). https://cdn.openai.com/pdf/novices-and-llm-august-2026.pdf
11. Nagarajan et al. *Roll the dice & look before you leap: Going beyond the creative limits of next-token prediction* (2025). https://proceedings.mlr.press/v267/nagarajan25a.html ; https://arxiv.org/abs/2504.15266
12. OpenAI. *Our First Proof submissions* (2026). https://openai.com/index/first-proof-submissions/
13. Padmakumar and He. *Does Writing with Language Models Reduce Content Diversity?* ICLR 2024. https://openreview.net/pdf?id=Feiz5HtCD0 ; https://proceedings.iclr.cc/paper_files/paper/2024/hash/02dec8877fb7c6aa9a79f81661baca7c-Abstract-Conference.html
14. Peeperkorn et al. *Is Temperature the Creativity Parameter of Large Language Models?* ICCC 2024. https://arxiv.org/abs/2405.00492 ; https://computationalcreativity.net/iccc24/papers/ICCC24_paper_70.pdf
15. Nguyen et al. *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*. 2024 preprint, ICLR 2025; checked v8. https://arxiv.org/abs/2407.01082 ; https://arxiv.org/html/2407.01082v8
16. *A Thorough Examination of Decoding Methods in the Era of LLMs*. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.489.pdf
17. Peeperkorn et al. *Mind the Gap: Conformative Decoding to Improve Output Diversity of Instruction-Tuned Large Language Models*. 2025 preprint. https://arxiv.org/abs/2507.20956 ; https://arxiv.org/pdf/2507.20956
18. Meincke, Mollick, and Terwiesch. *Prompting Diverse Ideas: Increasing AI Idea Variance*. 2024 working paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708466 ; https://arxiv.org/pdf/2402.01727
19. Agrawal and Goyal. *Addressing LLM Diversity by Infusing Random Concepts*. January 2026 preprint. https://arxiv.org/abs/2601.18053 ; https://arxiv.org/html/2601.18053v1
20. *How Context Injection Shapes Creativity in LLM-Generated Alternative Uses*. ICCC 2026. https://computationalcreativity.net/iccc26/papers/ICCC26_paper_171.pdf
21. Yasunaga et al. *Large Language Models as Analogical Reasoners*. ICLR 2024. https://arxiv.org/abs/2310.01714 ; https://proceedings.iclr.cc/paper_files/paper/2024/file/4990dad2c1696224de42573d0222554a-Paper-Conference.pdf
22. *Relevant or Random: Can LLMs Truly Perform Analogical Reasoning?* ACL Findings 2025. https://aclanthology.org/2025.findings-acl.1230/
23. Schapiro et al. *Measuring Combinatorial Creativity in Large Language Models and Humans*. 2025 preprint; findings here use v6, October 2026, under review. https://arxiv.org/abs/2509.21043v6 ; https://arxiv.org/html/2509.21043v6
24. Wang, Downey, Ji, and Hope. *SciMON: Scientific Inspiration Machines Optimized for Novelty*. ACL 2024. https://aclanthology.org/2024.acl-long.18/ ; https://arxiv.org/abs/2305.14259
25. *MOOSE-Chem: Large Language Models for Rediscovering Unseen Chemistry Scientific Hypotheses*. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/51fd9a7d1706023cb9f8210cc6ac357c-Paper-Conference.pdf ; https://arxiv.org/html/2410.07076v5
26. *Benchmarking Language Model Creativity: A Case Study on Code Generation*. NAACL 2025, NeoCoder and Denial Prompting. https://arxiv.org/html/2407.09007v2
27. Zhang, Xin, and Zhong. *Where You Inject Diversity Matters: A Unified Framework for Diverse Generation*. June 2026 preprint. https://arxiv.org/abs/2606.10302 ; https://arxiv.org/html/2606.10302v1
28. Si, Yang, and Hashimoto. *Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers*. 2024 preprint / ICLR 2025. https://arxiv.org/abs/2409.04109
29. Si, Hashimoto, and Yang. *The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas*. June 2025 preprint / ICLR 2026. https://arxiv.org/abs/2506.20803 ; https://arxiv.org/html/2506.20803v1 ; https://iclr.cc/virtual/2026/poster/10010564
30. Yamada et al. *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search*. 2025. https://arxiv.org/pdf/2504.08066 ; https://ar5iv.labs.arxiv.org/html/2504.08066
31. Romera-Paredes et al. *Mathematical discoveries from program search with large language models*. Nature, online December 2023. https://www.nature.com/articles/s41586-023-06924-6 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/
32. *AlphaEvolve: A coding agent for scientific and algorithmic discovery*. 2025 white paper. https://arxiv.org/abs/2506.13131 ; https://arxiv.org/html/2506.13131v1 ; https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
33. Lehman and Stanley. *Abandoning Objectives: Evolution through the Search for Novelty Alone*. Evolutionary Computation 19(2), 2011. https://doi.org/10.1162/EVCO_a_00025 ; https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf
34. Mouret and Clune. *Illuminating search spaces by mapping elites*. 2015 preprint; original-version reference. https://arxiv.org/abs/1504.04909v1 ; https://arxiv.org/pdf/1504.04909v1.pdf
35. Zhang, Lehman, Stanley, and Clune. *OMNI: Open-endedness via Models of human Notions of Interestingness*. 2023 preprint. https://arxiv.org/abs/2306.01711 ; https://ar5iv.labs.arxiv.org/html/2306.01711
36. Faldor, Zhang, Cully, and Clune. *OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code*. 2024 preprint / ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/d40d7cbe7210f8a13ea0149eeae9c6de-Paper-Conference.pdf ; https://arxiv.org/abs/2405.15568
37. Clune. *AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence*. 2019. https://arxiv.org/abs/1905.10985
38. Fernando et al. *Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution*. 2023 preprint / ICML 2024. https://arxiv.org/abs/2309.16797 ; https://proceedings.mlr.press/v235/fernando24a.html
39. West and Potts. *Base Models Beat Aligned Models at Randomness and Creativity*. 2025, CoLM. https://arxiv.org/abs/2505.00047 ; https://arxiv.org/pdf/2505.00047v2.pdf ; https://web.stanford.edu/~cgpotts/papers.html
40. Saakyan, Kim, Muresan, and Chakrabarty. *Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity*. 2025 preprint / ICLR 2026; checked v2. https://arxiv.org/abs/2509.22641 ; https://arxiv.org/html/2509.22641v2
41. Li et al. *Jointly Reinforcing Diversity and Quality in Language Model Generations*. DARLING, 2025. https://arxiv.org/html/2509.02534 ; https://ai.meta.com/research/publications/jointly-reinforcing-diversity-and-quality-in-language-model-generations/
42. Hu et al. *Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs*. ACL Findings 2026. https://aclanthology.org/2026.findings-acl.1982/ ; https://aclanthology.org/2026.findings-acl.1982.pdf
43. Tuyls, Foster, Krishnamurthy, and Ash. *Representation-Based Exploration for Language Models: From Test-Time to Post-Training*. 2025 preprint / ICLR 2026; checked July 2026 v2. https://arxiv.org/abs/2510.11686 ; https://arxiv.org/html/2510.11686v2
44. *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. NeurIPS 2025. https://arxiv.org/abs/2505.03335v3 ; https://papers.nips.cc/paper_files/paper/2025/file/9837dc00ff67d176373268ed48042d49-Paper-Conference.pdf
45. *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents*. 2025. https://arxiv.org/abs/2505.22954v3
46. Gottweis et al. *Towards an AI co-scientist* (2025 v1), revised as *Accelerating scientific discovery with Co-Scientist* (Nature 2026). https://arxiv.org/abs/2502.18864 ; https://doi.org/10.1038/s41586-026-10644-y
47. *AI mirrors experimental science to uncover a mechanism of gene transfer crucial to bacterial evolution*. Cell 2025. https://www.cell.com/cell/fulltext/S0092-8674(25)00973-0
48. Guo et al. *InfRL: Inference-time Reinforcement Learning for Research Idea Optimization*. KDD 2026. https://doi.org/10.1145/3770855.3819029 ; https://dl.acm.org/doi/pdf/10.1145/3770855.3819029
