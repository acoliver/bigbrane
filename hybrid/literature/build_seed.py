"""Build seed_reading_list.toml from logged keyless searches.

Run from the repository root:

    .venv-hybrid/bin/python -m hybrid.literature.build_seed              # full build
    .venv-hybrid/bin/python -m hybrid.literature.build_seed --searches-only

Every lens query runs through the real clients (cached under hybrid/literature/cache/).
Every raw hit is written to cache/seed_build_log.jsonl. A pick is accepted only if the
(source, source_id) pair appears in the raw hits of one of its lens's logged queries, so
each seed entry traces back to a recorded search. Metadata comes from the logged hit. The shipped seed_provenance.json projection is
validated against original cached HTTP responses and emitted with the seed list.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from hybrid.literature.client import DEFAULT_SOURCES, Literature, default_literature
from hybrid.literature.events import utc_now
from hybrid.literature.paper import Paper
from hybrid.literature.seed import (
    BUILD_LOG_PATH,
    KNOWN_ALTERNATIVES_LENS,
    LENSES,
    SEED_PATH,
    SEED_PROVENANCE_PATH,
    compile_seed_provenance,
    read_seed_provenance,
    write_seed_provenance,
    SeedEntry,
    search_row,
    write_build_log,
    write_seed,
)

SEARCH_LIMIT = 10


@dataclass(frozen=True)
class Pick:
    source: str
    source_id: str
    relevance: str


LENS_QUERIES: dict[str, tuple[str, ...]] = {
    "reservoir_computing": ("echo state network", "reservoir computing", "liquid state machine"),
    "vector_symbolic_hyperdimensional": (
        "hyperdimensional computing",
        "vector symbolic architectures",
        "hyperdimensional binding sequence",
    ),
    "holographic_reduced_representations": (
        "holographic reduced representations",
        "circular convolution associative memory",
        "holographic embeddings sequence",
    ),
    "modern_hopfield_networks": (
        "modern Hopfield networks",
        "dense associative memory",
        "Hopfield networks transformers attention",
    ),
    "predictive_coding": ("predictive coding networks", "predictive coding learning", "predictive processing hierarchical"),
    "active_inference": ("active inference", "free energy principle", "active inference sequence learning"),
    "dendritic_spiking_computation": (
        "dendritic computation neurons",
        "spiking neural networks sequence",
        "dendrites learning",
    ),
    "neural_odes_koopman": ("neural ODE", "Koopman operator learning", "Koopman sequence model"),
    "tensor_networks_language": (
        "tensor network language model",
        "matrix product states language modeling",
        "tensor train sequence",
    ),
    "renormalization_multiscale": (
        "renormalization group deep learning",
        "multiscale renormalization neural network",
        "coarse graining sequence",
    ),
    "kernel_gaussian_processes": ("kernel methods sequence", "Gaussian process sequence model", "neural tangent kernel"),
    "bayesian_program_induction": ("Bayesian program learning", "program induction few-shot", "rule induction Bayesian"),
    "cellular_automata": ("neural cellular automata", "cellular automata learning", "cellular automaton sequence"),
    "optimal_transport": ("optimal transport sequence", "Sinkhorn optimal transport learning", "Wasserstein sequence model"),
    "sheaves_category_models": ("sheaf neural network", "category theory deep learning", "sheaf theory learning"),
    "compression_mdl": ("minimum description length learning", "compression language modeling", "minimum description length neural"),
    "algorithmic_information": ("algorithmic information theory", "Kolmogorov complexity learning", "algorithmic probability"),
    "sparse_distributed_memory": ("sparse distributed memory", "Kanerva sparse distributed memory", "sparse distributed memory neural"),
    "complementary_learning_systems": (
        "complementary learning systems",
        "hippocampus neocortex memory consolidation model",
        "fast and slow learning memory",
    ),
    "hyperbolic_geometry": ("hyperbolic neural networks", "hyperbolic embeddings hierarchy", "Poincare embeddings"),
    "quantum_inspired": ("quantum-inspired language model", "quantum neural network sequence", "quantum-inspired machine learning"),
    "chemical_reaction_networks": (
        "chemical reaction network computation",
        "chemical reaction networks neural",
        "chemical reaction networks learning",
    ),
    "morphogenesis": ("neural morphogenesis", "reaction diffusion pattern learning", "morphogenetic neural network"),
    "known_llm_alternatives_to_avoid_reinventing": (
        "linear attention transformers are RNNs",
        "Mamba selective state spaces",
        "RWKV linear time RNN",
        "retentive network retention",
        "Hyena long convolution",
        "xLSTM extended LSTM",
        "BitNet 1-bit transformers",
        "large concept models",
        "test-time training layers hidden state",
    ),
}

# Picks were chosen by reading the logged hits. Every pick must appear in its lens's hits.
# Relevance notes are judged against: a non-quadratic, sample-efficient sequence learner
# built on a different mathematical basis.
PICKS: dict[str, tuple[Pick, ...]] = {
    "reservoir_computing": (
        Pick("openalex", "W2054217036", "Fixed random recurrent dynamics hold the memory and only a linear readout is trained, so state updates are not learned."),
        Pick("openalex", "W2036451492", "Unifies reservoir variants and shows which fixed dynamics carry memory, a non-attention basis for sequence state."),
        Pick("openalex", "W2029967456", "Spectral radius and leak rate rules size a fixed-dimension state, the design question a fixed-state learner must answer."),
        Pick("openalex", "W2807199165", "Proves universality of echo state networks, a formal bound on what a fixed-size state map can represent."),
        Pick("openalex", "W3170977978", "Next generation reservoir computing replaces random state with explicit nonlinear features of delayed inputs, a closed-form alternative."),
    ),
    "vector_symbolic_hyperdimensional": (
        Pick("openalex", "W2070862086", "Kanerva's account of computing with high-dimensional random vectors, binding and bundling into a fixed-width state."),
        Pick("openalex", "W1615576556", "Early vector symbolic architecture argument, relevant to encoding order by binding instead of pairwise attention."),
        Pick("openalex", "W3003981068", "Compares vector symbolic binding operators, which fixes the algebra a sequence learner would use."),
        Pick("openalex", "W3204836041", "Theory of hyperdimensional capacity and noise, the kind of bound a feasibility judge would demand."),
        Pick("openalex", "W4318238964", "Capacity analysis of vector symbolic architectures, a direct bound on how many items a fixed-width vector can hold."),
    ),
    "holographic_reduced_representations": (
        Pick("openalex", "W2157306293", "Plate's holographic reduced representations: circular convolution binds into a fixed-width vector, a classical fixed-size state."),
        Pick("openalex", "W2010359042", "Encodes sequence structure with holographic reduced representations, direct evidence that this state can carry order."),
        Pick("openalex", "W2552210802", "Recent theory and utility work on holographic reduced representations, with capacity and retrieval results."),
        Pick("openalex", "W4397028179", "Generalized holographic reduced representations change the binding algebra, the kind of different basis the card would need to justify."),
    ),
    "modern_hopfield_networks": (
        Pick("openalex", "W2412287113", "Dense associative memory with higher-order energies: memory as an energy minimum rather than a sequence recurrence."),
        Pick("openalex", "W3127151792", "Modern Hopfield networks recover transformer attention, so any energy-based card must be checked against this reduction."),
        Pick("openalex", "W2584321260", "Shows dense associative memory is robust to adversarial inputs, a robustness property a memory-based learner would need."),
        Pick("openalex", "W4391776157", "Exponential capacity of dense associative memories, a recent bound against which a fixed-size memory claim can be tested."),
        Pick("arxiv", "2603.06875", "Stochastic attention via Langevin dynamics on the Hopfield energy, a 2026 variant that treats retrieval as sampling."),
    ),
    "predictive_coding": (
        Pick("openalex", "W2604086949", "Predictive coding approximates backpropagation with local error signals, a learning rule that avoids a global gradient."),
        Pick("openalex", "W2401640538", "Deep predictive coding networks for video prediction, a temporal predictor trained by local errors."),
        Pick("openalex", "W4297808394", "Contrastive predictive coding learns sequence representations by predicting future latents without a generative decoder."),
        Pick("openalex", "W4393392404", "Predictive coding networks for temporal prediction, a 2024 sequence version of the learning rule."),
        Pick("arxiv", "2603.06142", "Predictive coding graphs as a superset of feedforward networks, a 2026 result on what this learning rule can represent."),
    ),
    "active_inference": (
        Pick("openalex", "W2148764920", "Friston's free-energy principle as a unified brain theory, the framework that active inference builds on."),
        Pick("openalex", "W2552810632", "Active inference as a process theory, where inference over a sequence is an action and perception loop."),
        Pick("openalex", "W4226400653", "Active inference textbook treatment of generative models over sequences."),
        Pick("arxiv", "2409.01066", "Learning in hybrid active inference models, a 2024 variant that asks whether learning rather than inference drives sample efficiency."),
    ),
    "dendritic_spiking_computation": (
        Pick("openalex", "W2529004582", "Deep learning with segregated dendrites, a learning rule over dendritic compartments with a computational basis unlike attention."),
        Pick("openalex", "W2093002383", "Matching recall and storage in sequence learning with spiking networks, a sequence-memory mechanism built from spikes."),
        Pick("openalex", "W2998412226", "Dendritic action potentials in human cortex, evidence that single neurons compute more than point units."),
        Pick("openalex", "W4402713424", "Learning long sequences in spiking neural networks, a 2024 spiking route to long-context sequence learning."),
    ),
    "neural_odes_koopman": (
        Pick("openalex", "W2926003515", "Augmented neural ODEs give continuous-depth state with a fixed dimension, a basis different from discrete attention."),
        Pick("openalex", "W3208285837", "Neural flows as an efficient alternative to neural ODEs, a closed-form time map that avoids a solver."),
        Pick("openalex", "W3132732981", "A Koopman approach to understanding sequence neural models, reading RNN dynamics as a linear operator."),
        Pick("openalex", "W4391160315", "Koopman operator learning with invertible networks, a 2024 learned linear lift for dynamics."),
        Pick("arxiv", "2606.23957", "Learning the Koopman operator with attention-free transformers, a 2026 route that drops attention entirely."),
    ),
    "tensor_networks_language": (
        Pick("openalex", "W2096391265", "Matrix product states, PEPS and variational renormalization, where bond dimension sets a fixed state budget."),
        Pick("openalex", "W2753545915", "Unsupervised generative modelling with matrix product states, a tensor-network model trained on sequence data."),
        Pick("openalex", "W2619959423", "Compressing recurrent networks with tensor train decomposition, a route to small state from a tensor factorization."),
        Pick("openalex", "W2917217327", "Probabilistic modelling with matrix product states, linking bond dimension to sequence memory."),
        Pick("openalex", "W4200400528", "Review of matrix product states and PEPS concepts and symmetries, the current reference on this basis."),
    ),
    "renormalization_multiscale": (
        Pick("openalex", "W1890000358", "Exact mapping between variational renormalization and deep learning, a coarse-graining view of depth."),
        Pick("openalex", "W1555084454", "Deep learning and the renormalization group, the physics framing of hierarchical features."),
        Pick("openalex", "W3034639988", "Tests whether deep learning is a renormalization group flow, a check on whether depth is coarse-graining."),
        Pick("openalex", "W4404397493", "Lattice physics approaches for neural networks, a 2024 review of lattice-based network models."),
        Pick("openalex", "W4408848246", "Network renormalization, a 2025 method that coarse-grains a network into a multiscale state."),
    ),
    "kernel_gaussian_processes": (
        Pick("openalex", "W2809090039", "Neural tangent kernel, the infinite-width limit that gives a kernel and a reference point for any non-kernel learner."),
        Pick("openalex", "W2970239942", "Inductive bias of neural tangent kernels, what kernel learners can and cannot learn from structure."),
        Pick("openalex", "W2403880556", "Clinical time series modelled with Gaussian process sequences, a data-efficient kernel route over sequences."),
        Pick("openalex", "W4409775841", "Positivity of the neural tangent kernel, a 2025 result on the kernel's sign structure."),
    ),
    "bayesian_program_induction": (
        Pick("openalex", "W2550492213", "Sampling for Bayesian program learning, a program-induction method with sample-efficient posterior search."),
        Pick("openalex", "W3034789704", "DreamCoder, wake-sleep program learning that builds a library from few examples."),
        Pick("openalex", "W2021490264", "Bayesian learning and the psychology of rule induction, a few-shot rule-learning model."),
        Pick("arxiv", "2509.01016", "Error sources in LLM-based hypothesis search for few-shot rule induction, a 2025 analysis of the task the bench will use."),
    ),
    "cellular_automata": (
        Pick("openalex", "W3006201994", "Growing neural cellular automata, a learned local update rule grown into a pattern."),
        Pick("arxiv", "2505.13058", "A path to universal neural cellular automata, a 2025 universality result for learned local rules."),
        Pick("arxiv", "2501.03573", "Neural cellular automata and deep equilibrium models, relating local updates to implicit fixed-point models."),
        Pick("openalex", "W7214964112", "Reasoning with neural cellular automata, a 2026 use of local dynamics for reasoning."),
    ),
    "optimal_transport": (
        Pick("openalex", "W2620318602", "Near-linear time Sinkhorn approximations to optimal transport, a computational route that could make transport-based memory cheap."),
        Pick("openalex", "W2890208366", "Order-preserving optimal transport for distances between sequences, with order built into the transport."),
        Pick("openalex", "W2910092637", "Sequence-to-sequence learning via optimal transport, using transport as a training objective for sequences."),
        Pick("arxiv", "2502.01588", "Differentiable alignment for sequence-to-sequence modelling via optimal transport, a 2025 route."),
    ),
    "sheaves_category_models": (
        Pick("arxiv", "2605.19021", "Deep neural sheaf diffusion, local-to-global message passing with a different algebra from attention."),
        Pick("openalex", "W4409156929", "Learning sheaf Laplacians by optimizing restriction maps, a 2024 learned-restriction variant."),
        Pick("arxiv", "2608.02558", "Benchmarking sheaf neural networks on inductive tasks, evidence on whether sheaf models have been tested."),
        Pick("arxiv", "2610.01322", "Clifford sheaf neural networks, a geometric algebra extension of sheaf models."),
    ),
    "compression_mdl": (
        Pick("openalex", "W32508012", "Grünwald's minimum description length principle, the formal definition of compression as learning."),
        Pick("openalex", "W4386908184", "Language modeling is compression, linking a predictor's code length to sequence learning."),
        Pick("arxiv", "2504.00395", "A theory of machine understanding via the minimum description length principle, a 2025 account of learning by compression."),
        Pick("arxiv", "2510.12077", "Compressibility measures complexity: minimum description length meets singular learning theory, a 2025 link from compression to generalization."),
    ),
    "algorithmic_information": (
        Pick("openalex", "W2057728133", "The discovery of algorithmic probability, Solomonoff's universal prior over sequences."),
        Pick("openalex", "W2165347833", "Transfer learning with Kolmogorov complexity, an empirical use of algorithmic information for few-shot transfer."),
        Pick("openalex", "W2167267841", "Algorithmic probability in theory and application, the practical side of the universal prior."),
        Pick("arxiv", "2509.22445", "Bridging Kolmogorov complexity and deep learning, a 2025 description-length result for networks."),
    ),
    "sparse_distributed_memory": (
        Pick("openalex", "W1573706465", "Kanerva's sparse distributed memory, a fixed-size address-based memory whose capacity bounds any fixed-state claim."),
        Pick("openalex", "W1525999779", "Sparse distributed memory and related models, the wider family of sparse memories."),
        Pick("openalex", "W1988238855", "Sparse distributed memory with N-of-M codes, where sparse coding changes the capacity trade-off."),
        Pick("openalex", "W2141088063", "Sparse distributed memory with rank-order neural codes, a neural encoding route to the same memory."),
        Pick("arxiv", "2607.02967", "Rank-order N-of-M codes for sparse distributed memory, a 2026 update on the same sparse-code trade-off."),
    ),
    "complementary_learning_systems": (
        Pick("openalex", "W2047057213", "Why there are complementary learning systems in the hippocampus and neocortex, a fast and slow split a sample-efficient learner could use."),
        Pick("openalex", "W2100377190", "Hippocampal and neocortical contributions to recognition memory, a complementary-learning computational model."),
        Pick("openalex", "W2424347275", "What learning systems do intelligent agents need, a 2016 argument for complementary fast and slow memories."),
        Pick("openalex", "W2549355627", "Complementary learning systems within the hippocampus as a neural network model of pattern separation."),
    ),
    "hyperbolic_geometry": (
        Pick("openalex", "W2803294777", "Hyperbolic neural networks, where hierarchies embed with exponential volume growth in a non-Euclidean basis."),
        Pick("openalex", "W2796314760", "Hyperbolic entailment cones, order relations as geometric containment."),
        Pick("openalex", "W3171364227", "Fully hyperbolic neural networks, with every layer in hyperbolic space, relevant to whether such state scales."),
        Pick("arxiv", "2410.16813", "Klein model for hyperbolic neural networks, a 2024 choice of model for efficient hyperbolic layers."),
    ),
    "quantum_inspired": (
        Pick("openalex", "W2889425359", "Quantum many-body wave-function inspired language modelling, a tensor-like state for sequences."),
        Pick("openalex", "W4394589674", "Quantum-inspired language models based on unitary transformation, with unitary evolution as the sequence step."),
        Pick("openalex", "W4401042518", "Quantum-inspired language model with a Lindblad master equation, open-system dynamics for a sequence state."),
        Pick("openalex", "W4412814002", "Quantum-inspired machine learning based on the Heisenberg uncertainty principle, a 2025 uncertainty-based model."),
    ),
    "chemical_reaction_networks": (
        Pick("openalex", "W2008591003", "Computation with finite stochastic chemical reaction networks, the computational basis of reaction networks."),
        Pick("openalex", "W1991095994", "Deterministic function computation with chemical reaction networks, a deterministic route to computing functions."),
        Pick("openalex", "W4400038677", "Chemical reservoir computation in a self-organizing reaction network, a 2024 reservoir built from chemistry."),
        Pick("openalex", "W4316507319", "Chemical reaction networks and opportunities for machine learning, a 2023 survey of learning on reaction networks."),
    ),
    "morphogenesis": (
        Pick("openalex", "W2149396346", "Prior learning and Gibbs reaction-diffusion, with learned priors expressed as local pattern dynamics."),
        Pick("openalex", "W2016799738", "Morphogenic neural networks encode abstract rules by data, development treated as a learning rule."),
        Pick("openalex", "W2963017542", "Formal methods approach to pattern synthesis in reaction-diffusion systems."),
        Pick("openalex", "W1529291180", "Spatio-temporal dynamics of reaction-diffusion patterns."),
    ),
    KNOWN_ALTERNATIVES_LENS: (
        Pick("openalex", "W3034573343", "Linear attention as recurrence (Transformers are RNNs), the closest known reduction for any kernel or linear-state card."),
        Pick("arxiv", "2312.00752", "Mamba: selective state spaces, a known linear-time sequence model that any diagonal or gated state card must be compared with."),
        Pick("openalex", "W4389524555", "RWKV, a linear-time RNN with a transformer-style training mode, a known reduction for recurrent state cards."),
        Pick("arxiv", "2307.08621", "Retentive networks, a known parallel-or-recurrent sequence model with exponential decay."),
        Pick("arxiv", "2302.10866", "Hyena, long convolutions as a subquadratic operator, a known alternative to attention."),
        Pick("arxiv", "2405.04517", "xLSTM, an extended LSTM with matrix memory, a known gated recurrent model with expanded state."),
        Pick("arxiv", "2310.11453", "BitNet, 1-bit transformers, a known quantization route that a low-precision claim would have to separate from."),
        Pick("openalex", "W4405354515", "Large concept models, sentence-level concept representations, the closest known relative of concept-level state claims."),
        Pick("arxiv", "2407.04620", "Learning to learn at test time, RNN hidden states that are themselves trained at test time, the closest relative of learning-rule claims."),
    ),
}


def run_searches(literature: Literature) -> dict[str, dict[tuple[str, str], Paper]]:
    hits: dict[str, dict[tuple[str, str], Paper]] = {lens: {} for lens in LENSES}
    wanted = {lens: {(p.source, p.source_id) for p in PICKS.get(lens, ())} for lens in LENSES}

    def rows():
        for lens in LENSES:
            for query in LENS_QUERIES[lens]:
                receipt = literature.search_with_receipt(query, DEFAULT_SOURCES, SEARCH_LIMIT)
                for source, found in receipt.per_source.items():
                    for paper in found:
                        if (source, paper.source_id) in wanted[lens]:
                            hits[lens][(source, paper.source_id)] = paper
                print(f"{lens}: {query!r} -> {len(receipt.papers)} unique, errors={sorted(receipt.errors)}", flush=True)
                yield search_row(lens, receipt)

    write_build_log(rows(), BUILD_LOG_PATH)
    print(f"wrote search rows to {BUILD_LOG_PATH}", flush=True)
    return hits


def picks_to_entries(hits: dict[str, dict[tuple[str, str], Paper]]) -> list[SeedEntry]:
    entries: list[SeedEntry] = []
    for lens in LENSES:
        for pick in PICKS.get(lens, ()):
            found = hits[lens].get((pick.source, pick.source_id))
            if found is None:
                raise ValueError(
                    f"pick {pick.source}:{pick.source_id} for lens {lens} is not in its logged search hits"
                )
            entries.append(SeedEntry(lens, found, pick.relevance))
    return entries


def write_seed_bundle(
    entries: list[SeedEntry], generated_at: str, *, seed_path: Path = SEED_PATH,
    provenance_path: Path = SEED_PROVENANCE_PATH, log_path: Path = BUILD_LOG_PATH,
    cache_root: Path = BUILD_LOG_PATH.parent,
) -> None:
    document = compile_seed_provenance(entries, log_path, cache_root)
    pending_seed = seed_path.with_suffix(seed_path.suffix + ".new")
    pending_provenance = provenance_path.with_suffix(provenance_path.suffix + ".new")
    write_seed(entries, pending_seed, generated_at)
    write_seed_provenance(document, pending_provenance)
    read_seed_provenance(pending_provenance, pending_seed)
    os.replace(pending_seed, seed_path)
    os.replace(pending_provenance, provenance_path)


def main(argv: list[str]) -> int:
    if argv not in ([], ["--searches-only"]):
        print("usage: python -m hybrid.literature.build_seed [--searches-only]", file=sys.stderr)
        return 2
    generated_at = utc_now()
    hits = run_searches(default_literature())
    if argv == ["--searches-only"]:
        return 0
    entries = picks_to_entries(hits)
    write_seed_bundle(entries, generated_at)
    counts = {lens: sum(1 for entry in entries if entry.lens == lens) for lens in LENSES}
    print(f"wrote {len(entries)} seed entries to {SEED_PATH}")
    for lens, count in counts.items():
        print(f"  {lens}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
