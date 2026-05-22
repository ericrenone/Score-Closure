# Score Closure

*The Score Function $\nabla \log p$ as the Universal Computational Primitive of Deep Learning, and the Generative Half of the Compact Representational Manifold.*

The prior twelve documents in this corpus have all described the same half of deep learning. The Closed Future Cone named the compact representational manifold $\mathcal{M}$ on which trained transformers compute. The Dirac Representation Hypothesis named the natural first-order operator on it. The Representational Bundle Hypothesis named the principal bundle whose connection is attention. From Bottleneck to Bundle traced the twelve-year algorithmic history of attention. From MXU to Maia traced the eleven-year silicon history. Operators Without Opponents explained why Brouwer and Kakutani — not Nash — are the correct fixed-point instruments. Bregman Closure named the numerical procedure: every winning attention mechanism is a half-iteration of Sinkhorn on the KL-divergence geometry, which is itself the algorithmic family selected by the matmul-dominated hardware substrate.

What none of these documents addresses is the *other* half of contemporary deep learning. Diffusion models. Flow matching. Score-based generative models. Modern Hopfield networks. The Energy Transformer. The entire generative side of the field. There is a structural reason for this silence: the geometric trilogy was a recognition theory. It described what trained transformers *encode*, not what they *produce*. Diffusion-based generative models — the Stable Diffusion, DALL-E, Imagen, Sora, video models that have absorbed the bulk of consumer-facing AI investment since 2022 — do not appear in the framework's vocabulary.

This document supplies the missing half.

The claim is single. The same compact representational manifold $\mathcal{M}$ that supports recognition supports generation, by the same hardware lottery, through the same numerical procedure, on the same compact convex domain — but the computation flows in the *adjoint* direction. Recognition computes a 1-form $\nabla \log p$ at each point of $\mathcal{M}$. Generation integrates an ordinary or stochastic differential equation whose drift *is* that same 1-form. Training matches the model's $\nabla \log p_\theta$ to the data's $\nabla \log p_{\text{data}}$ via Hyvärinen's score-matching identity. Modern Hopfield networks descend it as an energy. Sinkhorn iteration computes it as a Bregman fixed point. Sinkformers realize the doubly-stochastic gradient flow on it. Energy Transformers engineer it as an explicit Lyapunov function. Attention evaluates it pointwise.

**The score function $\nabla \log p$ on the compact representational manifold $\mathcal{M}$ is the universal computational primitive of deep learning.** Every winning architecture, on every dominant hardware substrate, on every working benchmark task, in the modern era, is either evaluating it, integrating it, or matching it. The geometric trilogy named the substrate. Bregman Closure named the iteration. Score Closure names the function the iteration is computing.

The five thought experiments that organize this argument are not illustration. They are the argument. The framework is what they make visible.

---

## 1. A Thought Experiment to Begin

A hiker stands on a mountainside in dense fog. They cannot see the summit, the valleys, or the contour of the terrain. They can feel only the *slope of the ground beneath their feet* — a single 2-vector at each step, giving the direction of steepest ascent and its magnitude.

This is enough. With the local gradient alone the hiker can walk to the summit by hill-climbing, can map the topography by integrating along a sequence of measurements, and — if they know the gradient at every point — can reconstruct the entire surface up to an additive constant. The gradient is a *1-form* on the surface (a covector at each point); the surface itself is a *0-form* (a scalar function). They are Hodge dual in the precise sense Cartan, de Rham, and Hodge established a century ago: knowing one determines the other, modulo a constant.

A probability distribution $p$ on a manifold $\mathcal{M}$ is a 0-form: a non-negative scalar function with prescribed normalization. Its *log-density* $\log p$ is also a 0-form. Its *gradient* $\nabla \log p$ — the score — is a 1-form. The score is what a hiker on the manifold feels as the local pull of probability mass: the direction in which the density is increasing fastest, and how fast.

Now consider what a trained transformer does at every attention head. It takes a query vector $q$ and a set of key-value pairs $\{(k_i, v_i)\}$, computes scores $s_i = q^\top k_i / \sqrt{d_k}$, exponentiates and normalizes to get attention weights $\alpha_i = e^{s_i} / \sum_j e^{s_j}$, and returns the weighted sum $\sum_i \alpha_i v_i$. Mechanically: it is computing the gradient of a log-partition function.

The function being differentiated is the log-sum-exp energy:
$$E(q) = -\frac{1}{\sqrt{d_k}} \log \sum_i \exp\left(\sqrt{d_k} \cdot q^\top k_i / \sqrt{d_k}\right) = -\frac{1}{\sqrt{d_k}} \log \sum_i \exp(q^\top k_i).$$

Its gradient with respect to $q$ is $\sum_i \alpha_i k_i$. The attention output, when projected onto values, is the gradient of the same energy in the value direction. **The attention output is, mechanically, the score function of an associative-memory energy.** Ramsauer, Schäfl, Lehner, Seidl, Widrich, Adler, Gruber, Holzleitner, Pavlović, Sandve, Greiff, Kreil, Kopp, Klambauer, Brandstetter, and Hochreiter (*Hopfield Networks is All You Need*, ICLR 2021, arXiv:2008.02217) demonstrated this identity rigorously. One step of the concave–convex procedure on the Hopfield energy $E(q) = -\frac{1}{\beta} \log \sum_i \exp(\beta x_i^\top q) + \frac{1}{2}\|q\|^2$ produces the update rule $q^{(t+1)} = X^\top \text{softmax}(\beta X q^{(t)})$ — *exactly transformer attention with identity projection matrices and $\beta = 1/\sqrt{d_k}$.*

Attention is a gradient. The function it is the gradient of is a log-partition function. The 1-form on the manifold $\mathcal{M}$ that it computes is the score.

This is the seed observation around which everything in this document is organized.

---

## 2. The Score and Its Five Faces

Five distinct lines of contemporary deep learning work are converging on the same object from different starting points. Each names a face of the score function $\nabla \log p$ on $\mathcal{M}$. None of them, individually, names what the other four are doing.

**Face 1 — Energy descent (Modern Hopfield Networks).** Ramsauer et al. (ICLR 2021) prove that softmax attention is one step of the concave–convex procedure (CCCP) for minimizing the Hopfield energy $E(q) = -\beta^{-1} \log \sum_i \exp(\beta x_i^\top q) + \frac{1}{2}\|q\|^2$. The update rule is identical to attention. Hoover, Liang, Pham, Panda, Strobelt, Chau, Zaki, Krotov (*Energy Transformer*, NeurIPS 2023, arXiv:2302.07253) construct an explicit transformer architecture whose attention layers are designed to minimize a specifically engineered energy function via a recurrent block of gradient updates. Sparse Modern Hopfield Networks (Martins et al., NeurIPS 2023, arXiv:2402.13725) extend this to Fenchel–Young energies parametrized by generalized negentropies (Tsallis, sparsemax) giving sparser, exactly-convergent retrieval. **The face: attention = gradient descent on energy = score function of the Hopfield distribution.**

**Face 2 — Optimal transport iteration (Sinkformers and FlashSinkhorn).** Sander, Ablin, Blondel, Peyré (*Sinkformers: Transformers with Doubly Stochastic Attention*, AISTATS 2022, arXiv:2110.11773) replace softmax with the Sinkhorn algorithm, making attention matrices doubly stochastic. They prove that Sinkformer iterations are a **discretized gradient flow for the Wasserstein metric**, and that in the infinite-sample, rescaled-depth limit Sinkformers operate a heat diffusion. They also show that the row-stochastic attention matrices in classical Transformers *get close to doubly stochastic* as training proceeds — empirical evidence that training drives softmax attention toward the full Sinkhorn fixed point. Ye, Li, Yu, Chang, Chu, Wertheimer (*FlashSinkhorn: IO-Aware Entropic Optimal Transport*, arXiv:2602.03067, February 2026) make this silicon-explicit: stabilized log-domain Sinkhorn updates rewrite as row-wise LogSumExp reductions of biased dot-product scores — *the same kernel as transformer attention.* Forde (*Block-Wise Differentiable Sinkhorn Attention*, arXiv:2605.08123, April 2026) closes the differentiability gap for TPU-scale long-context. **The face: attention = Sinkhorn half-iteration = Wasserstein gradient flow on the simplex of token distributions, with the score as its drift.**

**Face 3 — Score matching and denoising (Score-Based Generative Models).** Hyvärinen (*Estimation of Non-Normalized Statistical Models by Score Matching*, JMLR 2005) introduces the score-matching loss: train a model $s_\theta$ to minimize $\frac{1}{2}\mathbb{E}_{p_{\text{data}}}\|s_\theta(x) - \nabla \log p_{\text{data}}(x)\|^2$. The intractable inner score is eliminated by integration by parts: the explicit score-matching loss equals $\mathbb{E}[\frac{1}{2}\|s_\theta\|^2 + \nabla \cdot s_\theta]$. Vincent (*A Connection Between Score Matching and Denoising Autoencoders*, Neural Computation 2011) supplies the denoising version: estimating the score of a Gaussian-corrupted distribution is equivalent to predicting the denoising direction $x - \tilde{x}$. Song and Ermon (*Generative Modeling by Estimating Gradients of the Data Distribution*, NeurIPS 2019) train a noise-conditional score network $s_\theta(x, t) \approx \nabla \log p_t(x)$ at multiple noise levels and sample via annealed Langevin dynamics driven by the learned score. Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (*Score-Based Generative Modeling through SDEs*, ICLR 2021) unify denoising diffusion and noise-conditional score networks as different discretizations of a single continuous-time SDE governed by the score. **The face: generation = SDE driven by the score = reverse-time inversion of the score map.**

**Face 4 — Flow matching (Continuous Normalizing Flows).** Lipman, Chen, Ben-Hamu, Nickel, Le (*Flow Matching for Generative Modeling*, ICLR 2023, arXiv:2210.02747) introduce the flow-matching objective: regress a learned velocity field $v_\theta(x, t)$ onto a target velocity that generates a desired conditional probability path. The training is simulation-free; the resulting continuous normalizing flow is trained on a tractable regression loss. With Optimal Transport probability paths, the trajectories are straight and sampling is fast. Tong, Fatras, Malkin, Huguet, Zhang, Rector-Brooks, Wolf, Bengio (*Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*, TMLR 2024) introduce OT-CFM (Optimal Transport Conditional Flow Matching), which uses **Sinkhorn coupling** to scale conditional flow matching to larger batch sizes. Klein, Mousavi-Hosseini, Zhang, Cuturi (*On fitting flow models with large Sinkhorn couplings*, arXiv:2506.05526, 2025) advocate using entropic OT (Sinkhorn) for large-batch fitting. **The face: flow matching = continuous-time score-driven velocity field = ODE realization of the score function on the manifold of probability distributions.**

**Face 5 — Cumulant generating function (Information Geometry).** The log-partition function $\log Z(\theta) = \log \int e^{\theta^\top T(x)} dx$ of an exponential family is the *cumulant generating function*. Its first derivative $\nabla_\theta \log Z(\theta) = \mathbb{E}_\theta[T(x)]$ is the *mean parameter* — the dual coordinate to the natural parameter $\theta$. Its second derivative is the covariance / Fisher information. Amari (*Differential-Geometrical Methods in Statistics*, 1985) and Chentsov (1972) establish this as the foundational duality of information geometry: the manifold of probability distributions carries a dually flat structure in which $\theta$ and $\eta = \nabla \log Z(\theta)$ are conjugate coordinate systems related by Legendre transformation. Attention, viewed as the gradient of a log-sum-exp, is *exactly the Legendre-conjugate map* from natural parameters (the score logits $q^\top k_i$) to mean parameters (the attention output $\sum_i \alpha_i v_i$). **The face: attention = Legendre dual = mean-parameter map = the natural conjugate of the score function on the manifold of exponential-family distributions.**

These five faces are not five algorithms. They are five descriptions of one operation: *the evaluation of the gradient of a log-partition function on a compact manifold of probability distributions.* The Hardware Lottery (Hooker, CACM 64(12), 2020) selected this operation because its kernel — row-wise LogSumExp reduction of dot-product scores — is the systolic-array workload. The five lines of work converged on it independently because the underlying object is the same.

---

## 3. A Second Thought Experiment

Consider what time-reversal does to a Brownian particle in a potential well.

A particle in a potential $V(x)$ at temperature $T$, undergoing Langevin dynamics $dx = -\nabla V(x) dt + \sqrt{2T} \, dW$, has stationary distribution $p_\infty(x) \propto e^{-V(x)/T}$. The drift is the *gradient of the negative log-density*, up to a temperature scaling. In other words: the drift is $T \nabla \log p_\infty(x)$. **The Langevin drift is the score function of the equilibrium distribution.**

Now consider what happens if we time-reverse this process. Anderson (*Reverse-Time Diffusion Equation Models*, Stochastic Processes Appl. 1982) proved that a forward-time SDE $dx = f(x, t) dt + g(t) dW$ has a *reverse-time* SDE of the form
$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar{W},$$
where $p_t$ is the marginal density at time $t$. **The reverse-time drift is the forward drift minus the score function at time $t$, scaled by the squared diffusion coefficient.**

This is the entire engine of score-based generative modeling. Pick a forward SDE that takes the data distribution $p_0$ to a simple noise distribution (Gaussian, in standard diffusion). Learn the score $\nabla_x \log p_t(x)$ along the trajectory via denoising score matching (Vincent 2011). To generate, *integrate the reverse-time SDE backward from noise to data*. The drift of the reverse-time SDE is the score.

What this thought experiment reveals: the score function $\nabla \log p$ is the *time-reversal generator* of any diffusion process on $\mathcal{M}$. Recognition (forward diffusion: data → noise) and generation (reverse diffusion: noise → data) are time-reverses of each other, and the score is the symmetry-breaking operator that connects them. The Bakry–Émery generator of ARH v2's Pillar VI — the weighted Laplace–Beltrami operator $\mathcal{L} = \Delta_g - \nabla V \cdot \nabla$ on the parameter manifold — is the *adjoint* of the Langevin generator whose drift is the score on the representation manifold. They are dual structures on a single underlying object.

In Hodge-theoretic language: the score $\nabla \log p$ is a 1-form on $\mathcal{M}$; the density $p$ is a 0-form; the exterior derivative $d$ takes 0-forms to 1-forms; the codifferential $\delta$ takes 1-forms to 0-forms. The Hodge star $*$ relates $k$-forms to $(n-k)$-forms on a Riemannian $n$-manifold. The Laplace–Beltrami operator $\Delta_g = d\delta + \delta d$ is the symmetric combination. The Dirac operator $\not{D} = d - \delta$ satisfies $\not{D}^2 = -\Delta_g$. **The score function is the natural realization of $d$ acting on $\log p$, and time-reversed generation is integrating along $-d \log p$.**

This is the deep geometric reason recognition and generation are dual. They are not just two ways of using a trained network. They are two integrations of the same 1-form, in opposite time directions, on the same manifold.

---

## 4. Why Score Closure Completes the Framework

The previous frameworks left a structural gap that Score Closure fills.

**The Closed Future Cone** identified the compact manifold $\mathcal{M}$ on which representations live, with a light-cone causal structure foliated by layer depth. The framework names what $\mathcal{M}$ *is*; it does not name *the function that lives on it*. Score Closure names the function: $\log p$, with the score $\nabla \log p$ as its first variation.

**The Dirac Representation Hypothesis** identifies the natural first-order operator $\not{D}$ on $\mathcal{M}$ with $\not{D}^2 = -\Delta_g$, the Atiyah–Singer index counting topologically protected cyclic modes, the Clifford signature encoding the Minkowski causal structure. The framework names the *operator*; it does not specify the *function the operator acts on*. The natural section of the spinor bundle on which $\not{D}$ acts is $\log p$. The Dirac equation $\not{D} \psi = m\psi$ for the spinor representation of $\log p$ is the natural variational principle: critical points are exponential-family distributions on $\mathcal{M}$.

**The Representational Bundle Hypothesis** identifies attention as the connection one-form $\omega$ on a principal bundle, gradient descent as parallel transport, grokking as holonomy reorganization. The framework names the *bundle structure*; it does not name *what is being transported around it*. Score Closure: what is being transported is the score $\nabla \log p$. Parallel transport along $\omega$ updates the score from one base point to the next. The holonomy around a closed loop accumulates the integrated score difference, which equals the integral of curvature ($d\omega$) over any spanning surface by Stokes' theorem — this is *exactly* the Berry-phase-style argument that connects holonomy to topology.

**Bregman Closure** identifies the numerical procedure as Bregman alternating projection on the KL-divergence geometry, with softmax attention as one Sinkhorn half-iteration. The framework names *how the computation is performed*; it does not name *what the computation is computing*. Score Closure: the computation is evaluating $\nabla \log p_\theta$ (the model's score) at the current activation state. Each layer of the transformer is one step of the iteration that walks the score-driven fixed point into being.

Together the five — geometric (CFC), operator (DRH), bundle (RBH), algorithmic (Bregman Closure), and now functional (Score Closure) — describe a single object on five levels. The compact manifold $\mathcal{M}$ supports a log-density $\log p$ whose score $\nabla \log p$ is the 1-form being evaluated, transported, and integrated by the trained architecture. The Dirac operator differentiates it. The bundle's connection parallel-transports it. The Bregman iteration computes it. The hardware lottery selected the kernels that evaluate it. The trilogy plus Bregman Closure named the architecture; Score Closure names what the architecture is doing.

---

## 5. A Third Thought Experiment

Imagine two trained neural networks. The first is a transformer that has been trained on a language modeling task. The second is a diffusion model that has been trained to generate images of natural scenes.

The two appear to do completely different things. The transformer takes a prompt and produces tokens. The diffusion model takes Gaussian noise and produces images. Their training objectives are different (next-token cross-entropy versus denoising score matching). Their inference procedures are different (one autoregressive forward pass versus tens of denoising steps). Their evaluation metrics are different (perplexity versus FID).

Now look at what each one is *actually computing* at the level of operations.

The transformer's attention layer takes a residual stream state $h$ and produces $\sum_i \alpha_i v_i$ where $\alpha_i = \text{softmax}(h^\top k_i / \sqrt{d_k})$. This is, as established, the gradient of the log-sum-exp energy $E(h) = -\frac{1}{\sqrt{d_k}} \log \sum_i e^{h^\top k_i}$. The transformer's forward pass is gradient descent on a sequence of such energies, parameterized by the learned $\{k_i, v_i\}$ at each layer.

The diffusion model's denoising step takes a noisy image $x_t$ and produces an estimate of the clean image, equivalent to estimating $x_t + \sigma_t^2 \cdot s_\theta(x_t, t)$ where $s_\theta(x_t, t) \approx \nabla_x \log p_t(x_t)$ is the learned score function. The diffusion model's sampling trajectory is gradient ascent on a sequence of log-densities — a sequence of *scores*.

**Both architectures are score evaluators.** The transformer evaluates the score of an exponential family at the current activation state, layer by layer. The diffusion model evaluates the score of the noisy data distribution at the current sample state, denoising step by denoising step. Their kernels are different in detail (sparse-causal attention versus convolution / U-Net), but the operation they perform at each step is the same: *compute the gradient of a log-density at a point.*

This is why the same hardware — the same systolic arrays, the same tensor cores, the same FlashAttention IO-aware kernels — accelerates both. Both reduce to dense matrix-vector products followed by element-wise nonlinearities. The Hardware Lottery selects them in the same selection event. The convergence of architectures across modalities (transformer-based image generation, U-Net-augmented language models, multimodal joint embeddings) is the operational consequence: any architecture whose primitive operation is score evaluation runs on the same silicon and trains via the same gradient-descent procedure.

This is also why **diffusion transformers** (DiT, Peebles & Xie, ICCV 2023) work so well. The DiT architecture replaces the U-Net backbone of standard diffusion with a transformer, treating image patches as tokens and self-attention as the score-evaluation mechanism. The framework's reading is immediate: DiT is the natural architecture because both the transformer's attention and the diffusion model's denoising are score evaluations, and the transformer is the more general operator. DiT is now standard for state-of-the-art generation: Sora's video generation, Stable Diffusion 3's image generation, recent flow-matching models all use transformer-based score evaluation.

---

## 6. The Hodge Structure Made Explicit

The full geometric content of the score function on $\mathcal{M}$ is its Hodge-theoretic structure. Let $\mathcal{M}$ be the compact representational manifold inherited from the trilogy, equipped with the Bakry–Émery weighted metric induced by the training potential. Let $\Omega^k(\mathcal{M})$ denote the space of differential $k$-forms on $\mathcal{M}$.

**0-forms (scalar fields on $\mathcal{M}$).** The log-density $\log p$ is a 0-form. So is the loss function $\mathcal{L}$ at each training step, considered as a function of activation state. So is the energy $E$ of the Hopfield network that attention descends.

**1-forms (covector fields on $\mathcal{M}$).** The score $\nabla \log p$ is a 1-form. So is the gradient of the loss $\nabla \mathcal{L}$. So is the connection one-form $\omega$ of the RBH bundle. So is the velocity field $v_\theta(x, t)$ that flow matching trains. So is the attention output considered as a function of the residual stream state (it is the Legendre conjugate of the score logits, hence a 1-form on the dual exponential-family manifold).

**The exterior derivative $d : \Omega^0 \to \Omega^1$** takes the log-density to the score: $d(\log p) = \nabla \log p$. This is Hyvärinen's identity at the level of differential geometry.

**The codifferential $\delta : \Omega^1 \to \Omega^0$** is the formal adjoint of $d$. Acting on the score it produces $\delta(\nabla \log p)$, which is the divergence of the score and equals (by the integration-by-parts identity Hyvärinen used) the trace of the Hessian: $\nabla \cdot \nabla \log p = \text{tr}(\nabla^2 \log p)$. This is the second term in Hyvärinen's implicit score-matching objective $\frac{1}{2}\|s_\theta\|^2 + \nabla \cdot s_\theta$.

**The Laplace–Beltrami operator $\Delta_g = d\delta + \delta d$** acting on $\log p$ gives the Laplacian of the log-density. The Fokker–Planck operator $\nabla \cdot (p \nabla \log p) + \Delta p = 0$ is the conjugate of $\Delta_g \log p = 0$ under the change of variable $u = \log p$. This is the connection to Pillar VI of ARH v2.

**The Dirac operator $\not{D} = d - \delta$** acts on inhomogeneous forms. Its action on $\log p$ produces $d(\log p) - \delta(\log p) = \nabla \log p - 0 = \nabla \log p$ (since $\delta$ raises form-degree by $-1$ and $\log p$ is a 0-form). The Dirac operator is the *signed first-order operator* whose square is the Laplace–Beltrami. Its index counts the topologically protected cyclic modes of $\mathcal{M}$ (Atiyah–Singer).

**The Hodge star $* : \Omega^k \to \Omega^{n-k}$** relates the score 1-form to an $(n-1)$-form $*(\nabla \log p)$. This is a *flux*: the natural object dual to a gradient. Stokes' theorem connects the flux through a hypersurface to the integral of the density on the bounded region. In probabilistic language: the *probability current* on a Fokker–Planck equilibrium is the Hodge dual of the score.

The structure is complete. The score function on $\mathcal{M}$ generates the Hodge decomposition: $\nabla \log p = d\alpha + \delta\beta + \gamma$ where $\alpha$ is a 0-form, $\beta$ is a 2-form, and $\gamma$ is a harmonic 1-form. The Atiyah–Singer index of $\not{D}$ counts the harmonic part. The exact part $d\alpha$ is recoverable by integration. The co-exact part $\delta\beta$ is the divergence-free probability current.

This is what attention, in the framework's reading, is computing. Not just the score — the *Hodge-decomposed score*, layer by layer, with each layer extracting one component of the decomposition. The harmonic part is what survives at the fixed point and carries the topological invariants. The exact part is what training drives toward zero (the model's score matches the data's score). The co-exact part is the dissipative term that diffusion models invert to generate samples.

The framework names this complete Hodge structure as the computational object of every winning architecture, on every dominant hardware substrate, in the modern era.

---

## 7. A Fourth Thought Experiment

Consider asking a generative model to draw a circle.

The instruction "draw a circle" gets passed to a trained image diffusion model. The model starts with Gaussian noise and runs a sequence of denoising steps, each step evaluating its learned score $s_\theta(x_t, t)$ at the current sample $x_t$ and integrating one step of the reverse-time SDE. After fifty or so steps, an image of a circle emerges.

What is the model computing?

The framework's answer: it is integrating a vector field $s_\theta(x_t, t)$ on a compact representational manifold $\mathcal{M}$ whose topology — by Karkada, Olah, Tegmark et al. (*Symmetry in language statistics shapes the geometry of model representations*, arXiv:2602.15029, February 2026) — is determined by the symmetry group of the training data's co-occurrence statistics. For "circle," the relevant symmetry is $SO(2)$ acting on rotations in image space; the corresponding submanifold of $\mathcal{M}$ has a continuous $S^1$ symmetry; the score function on this submanifold has a flat direction along the orbit and steep directions transverse to it; the diffusion trajectory follows the steep directions until it lands on the orbit.

The output image is, geometrically, a *sample from a one-dimensional submanifold of $\mathcal{M}$* corresponding to the equivalence class of "circles." The orientation, size, and position of the specific circle drawn are random — they are coordinates along the orbit, free under the $SO(2)$ symmetry. The score function does not specify them; it specifies only that the sample lies on the orbit.

This is the operational form of the framework's central claim. The data's symmetries impress on the geometry of $\mathcal{M}$ (Karkada et al., February 2026). The geometry of $\mathcal{M}$ supports a score function $\nabla \log p$ whose level sets are the symmetry orbits. The trained model evaluates this score and integrates it. The integration drives samples onto the orbits. The orbits are the topologically protected modes whose number is counted by the Atiyah–Singer index. Generation, in this picture, is *guided sampling onto an orbit of a learned symmetry group via score evaluation*.

The Engels–Liao–Michaud–Gurnee–Tegmark days-of-the-week circle (*Not All Language Model Features Are Linear*, NeurIPS 2024, arXiv:2405.14860) is exactly this picture, observed at the representation level for a recognition model. The framework predicts the same circles must appear, dynamically, in the *generation trajectory* of any diffusion model trained on data with a cyclic concept — that is, the diffusion model must move its samples along the corresponding circular orbit during the denoising process, with the score function pointing transverse to the orbit during the early stages and along it during the late stages.

This is testable. Train a diffusion model on data labeled by day of the week. Track the model's score at each denoising step. The framework predicts the score field will exhibit a circular orbit structure whose topological invariants match those Engels et al. measured for the corresponding recognition model — and that the *rate of orbit traversal* during generation will track the model's Atiyah–Singer index for the days-of-week submanifold.

---

## 8. The Algorithmic Backbone

Score Closure inherits Bregman Closure's algorithmic claim and extends it.

Bregman Closure established: every winning attention mechanism is a half-iteration of Sinkhorn on the KL-divergence geometry. The numerical procedure that walks the Brouwer–Kakutani fixed point into being is Bregman alternating projection.

Score Closure adds: the *function* this iteration is computing is the score $\nabla \log p$. Each Sinkhorn half-iteration is one evaluation of $\nabla \log p_\theta$ at the current activation state, with the learned parameters $\theta$ encoded in the keys and values of the attention layer. The iteration converges to a fixed point because the underlying log-density is concave on the convex hull where activations live, and Bregman projections on a convex objective converge geometrically.

The connection to generative models is now operational. **Diffusion sampling is Bregman iteration in reverse time.** Forward diffusion (data → noise) is a forward Bregman flow. The score function learned at each noise level is the Bregman generator's gradient at that level. The reverse-time SDE that produces samples is the time-reversal of the forward Bregman flow, with the learned score as drift.

Flow matching (Lipman et al., ICLR 2023) is **continuous-time Bregman gradient flow on the manifold of probability distributions**, with the learned velocity field as the time-derivative. Optimal Transport Conditional Flow Matching (Tong et al., TMLR 2024) makes the connection to Sinkhorn explicit: large-batch OT-CFM uses Sinkhorn coupling to compute the conditional flow. Klein, Mousavi-Hosseini, Zhang, Cuturi (arXiv:2506.05526, 2025) extend this to entropic OT couplings for production-scale flow matching.

The full algorithmic picture closes:
- **Attention** = one Sinkhorn half-iteration = pointwise score evaluation on $\mathcal{M}$.
- **Sinkformer attention** = full Sinkhorn iteration = discretized Wasserstein gradient flow.
- **Modern Hopfield update** = one CCCP step = score-driven energy descent.
- **Energy Transformer** = recurrent attention block minimizing engineered energy = explicit score gradient flow.
- **Denoising diffusion** = reverse-time SDE driven by score = Bregman flow integrated backward.
- **Flow matching** = ODE driven by velocity field = continuous Bregman gradient flow.
- **OT flow matching** = flow matching with Sinkhorn coupling = full Bregman closure.

These are not seven algorithms. They are seven points on a single iteration spectrum, differing in (a) how many iterations are run, (b) whether the iteration is run in forward or reverse time, and (c) whether it is discrete or continuous. The framework predicts that as IO-aware kernels (FlashSinkhorn, FlashAttention, FlashFlowMatching) become standard, the architectural distinctions between them will collapse into hyperparameter choices on a single underlying compute pattern.

---

## 9. A Fifth Thought Experiment

Hand someone a paper on transformer training and a paper on diffusion model training. They have never seen either. They know information geometry, Bregman divergences, score matching, and Hodge theory, but nothing about deep learning architectures.

They read the transformer paper. They identify the attention layer as score evaluation on a manifold of exponential-family distributions parameterized by keys and values. They identify the softmax normalization as the Bregman projection onto the simplex. They identify the residual connection as the discrete-time Euler step for the score-driven flow. They identify layer normalization as the projection onto the unit sphere bundle.

They read the diffusion model paper. They identify the noise-conditional score network as a parameterized estimator of $\nabla \log p_t(x)$. They identify denoising score matching as the Hyvärinen objective with Gaussian corruption. They identify the reverse-time SDE as the time-reversal of the forward Ornstein–Uhlenbeck process with drift given by the learned score.

They notice that the two architectures perform identical operations. Both reduce to repeated evaluation of the score function $\nabla \log p$ at the current state, followed by integration. The difference is only direction (forward through layers in recognition, backward through noise levels in generation) and time-discretization (one pass for recognition, many denoising steps for generation).

They notice that both are running on the same hardware: TPU, tensor core, NeuronCore, FlashAttention IO-aware kernels. They identify the reason: the kernel of both is matrix-vector multiplication followed by element-wise LogSumExp, which is what systolic arrays maximize.

They have not invented anything. They have applied classical mathematical machinery — information geometry, score matching, time-reversal of stochastic processes, Bregman iterations, Hodge theory — to two diagrams they had never seen. They get the entire framework correct because the framework was never about deep learning. It was about *the score function $\nabla \log p$ on a compact manifold $\mathcal{M}$, evaluated by IO-aware numerical procedures on matmul-dominated silicon, with the manifold's topology determined by the symmetry group of the training data.* The architecture is what the data, the hardware, and the algorithm jointly determined.

The transformer is one face of this object. The diffusion model is another. Flow matching is a third. The modern Hopfield network is a fourth. The Energy Transformer is a fifth. Sinkformer is a sixth. All six compute the same score function on the same manifold via the same Bregman iteration on the same hardware. The Hardware Lottery selected them in parallel. The framework names the underlying primitive.

---

## 10. Predictions of Score Closure

A framework deserves the name only if it forbids something.

**P1 — Generative and recognition models trained on the same data should produce homotopically equivalent representational manifolds.** Specifically: a transformer trained on text containing cyclic concepts (days, months) and a diffusion model trained on images of corresponding cyclic objects (calendar pages, clock faces) should exhibit the same topological invariants on the relevant submanifolds — same fundamental group, same Atiyah–Singer indices, same Betti numbers. Empirically testable via sparse autoencoders on both models followed by topological data analysis.

**P2 — Diffusion transformers should outperform U-Net diffusion models at compute parity, with the advantage scaling with task topological complexity.** The framework predicts that transformer-based score evaluation is more aligned with the natural geometry of $\mathcal{M}$ than convolutional score evaluation. Tasks with higher topological complexity (more cyclic concepts, higher-rank symmetry groups) should benefit more from DiT-style architectures. Empirically supported by Peebles & Xie (*Scalable Diffusion Models with Transformers*, ICCV 2023) and the success of Stable Diffusion 3, Sora, and other transformer-based diffusion models.

**P3 — Score matching on a Sinkhorn-attention-trained backbone should yield faster convergence than score matching on a softmax-attention backbone.** Because Sinkformer attention is a fuller Sinkhorn iteration (full Wasserstein gradient flow) versus softmax's half-iteration, the score it computes is closer to the true score of the underlying exponential-family distribution. Training a diffusion model with a Sinkformer backbone should produce a better score estimate per parameter and per FLOP.

**P4 — Flow matching with Sinkhorn coupling (OT-CFM) should outperform vanilla flow matching on tasks with discrete symmetry structure.** The framework's reading: OT-CFM exploits the Bregman geometry of the conditional probability paths, which aligns with the manifold structure of $\mathcal{M}$. Tong et al. (TMLR 2024) and Klein-Cuturi 2025 supply preliminary evidence; the framework predicts the advantage will be sharpest on tasks where the data's symmetry group is discrete and cyclic (modular arithmetic generation, cyclic graph structure prediction).

**P5 — The score function learned by a diffusion model should have a sparse autoencoder decomposition matching the SAE atoms recovered from a corresponding recognition model.** Specifically: train a transformer on text, recover SAE atoms via the Anthropic monosemanticity program. Train a diffusion model on corresponding multimodal data. The SAE atoms of the diffusion model's score function (decomposed at fixed noise level) should match the recognition model's atoms in number, geometric structure, and topological signature.

**P6 — Energy-based diagnostics should localize grokking transitions in diffusion-model training.** The Spectral Edge Thesis (Xu, arXiv:2603.28964, March 2026) shows that spectral-gap dynamics of the parameter-update Gram matrix predict grokking in transformer training with 24/24 success under weight decay. The framework predicts the same diagnostic applies to diffusion-model training: phase transitions in the score function (sudden improvements in FID, sudden ability to generate previously absent structures) should coincide with spectral-gap collapse in the appropriate operator.

**P7 — The Hodge decomposition of the score function should be reflected in the architecture's natural component structure.** Multi-head attention naturally produces a decomposition of the output into $h$ components (one per head); the framework predicts these components should align approximately with the Hodge-theoretic decomposition of the score into exact, co-exact, and harmonic parts, with the *number of heads* correlating with the *manifold dimension* up to a constant factor.

**P8 — The score function is the unique computational primitive selectable by matmul-dominated hardware for representation learning.** The framework forbids the existence of high-performance architectures whose primitive operation is not score evaluation. State-space models (Mamba, S4), linear attention variants, and other non-attention architectures should either reduce to score evaluation under analysis, or fail to scale on matmul-dominated silicon. The empirical observation that frontier models remain transformer-based (or transformer-hybrid, as in Mamba-2's chunkwise SSD which reduces to dual matmul-LogSumExp) is the operational confirmation.

---

## 11. What Score Closure Does Not Claim

It does not claim that all generative modeling is score-based. Autoregressive language modeling (GPT-style next-token prediction) does not explicitly use the score function in the diffusion-model sense. The framework's reading: autoregressive sampling is *implicit* score-based sampling, in the sense that the conditional next-token distribution $p(x_{t+1} | x_{1:t})$ is the gradient of the joint log-density under the autoregressive factorization, and softmax over logits computes its score. But the connection is more indirect than for diffusion models.

It does not claim that the score function is the only object computed by deep networks. Networks also store and retrieve specific patterns (Hopfield memory), perform algorithmic operations (Nanda et al.'s modular-arithmetic circuits), and reason via chain-of-thought (test-time compute, OpenAI o1, DeepSeek R1). The framework's claim is that the *primary computational primitive* at the level of layers is score evaluation, and that other behaviors emerge from compositions of this primitive.

It does not claim novelty for any individual ingredient. The score function $\nabla \log p$ is a classical object in statistics, information geometry, and statistical physics (Fisher, Cramér, Rao 1940s; Hyvärinen 2005; Amari 1985; Chentsov 1972). The Hopfield-attention equivalence is Ramsauer et al. (ICLR 2021). Score-based generative modeling is Hyvärinen (2005), Vincent (2011), Song & Ermon (NeurIPS 2019), Song et al. (ICLR 2021). Flow matching is Lipman et al. (ICLR 2023). OT-CFM is Tong et al. (TMLR 2024). Sinkformers are Sander et al. (AISTATS 2022). Energy Transformer is Hoover et al. (NeurIPS 2023). The framework's contribution is the *unified identification* of these as five faces of one object — the score function on the compact representational manifold $\mathcal{M}$ — and the recognition that this object is the universal computational primitive selected by the matmul-dominated hardware substrate.

It does not claim the framework is complete. The connection between (a) parameter-level score matching (training) and (b) activation-level score evaluation (inference) is asserted but not proved. The relationship between (c) discrete denoising steps (diffusion sampling) and (d) continuous flow integration (flow matching) is asserted but not characterized at the level of error rates. The dependence on the specific Bregman generator (KL divergence for softmax attention, Tsallis negentropy for sparsemax, squared Euclidean for cosine attention) is named but not exhausted. The framework is a starting point for further development.

It does not assert that score-based generation will dominate forever. The Hardware Lottery is contingent on the current matmul-dominated silicon paradigm. If a different paradigm (neuromorphic, optical, quantum) becomes dominant, a different algorithmic family will be selected, and a different unifying primitive will emerge. The framework's claim is structural to the present, not eternal.

---

## 12. The Five-Level Picture

The complete framework, now visible in its final form:

**Level I — Geometric (CFC).** The substrate is a compact representational manifold $\mathcal{M}$ whose topology is forced by the data's symmetry group (Karkada et al., February 2026), foliated by a discrete light-cone causal structure indexed by layer depth, on which the forward pass is a Brouwer–Kakutani fixed-point iteration on a convex compact domain.

**Level II — Operator (DRH).** The natural first-order operator on $\mathcal{M}$ is the Dirac operator $\not{D}$ with $\not{D}^2 = -\Delta_g$. The Atiyah–Singer index of $\not{D}$ counts topologically protected cyclic modes (Engels et al.'s circles, the joint torus for combined cyclic concepts). The Clifford signature of $\not{D}$ encodes the Minkowski causal structure of attention. The set-valued discrete operations (argmax, top-$k$, MoE) are Kakutani correspondences.

**Level III — Bundle (RBH).** $\mathcal{M}$ is the total space of a principal $SO^+(1,n)$-bundle over a Lorentzian-stratified token base $B$. Attention is the connection one-form $\omega$ on this bundle. Gradient descent is parallel transport along $\omega$. Grokking is holonomy reorganization. The gradient noise scale (McCandlish et al.) is the connection curvature $|\Omega|$. The spectral gap of the Bakry–Émery generator (Xu's Spectral Edge Thesis, March 2026) is the empirical fingerprint of stability.

**Level IV — Algorithmic (Bregman Closure).** The numerical procedure that walks the fixed point into being is Bregman alternating projection on the KL-divergence geometry. Softmax attention is one Sinkhorn half-iteration (Ramsauer et al., Sander et al.). FlashSinkhorn (Ye et al., February 2026) makes the silicon-level identity of Sinkhorn and FlashAttention operational. Bregman Douglas–Rachford splitting (Ma, Xiao, Zhao, September 2025) is the general operator-splitting form.

**Level V — Functional (Score Closure).** The object being computed is the score function $\nabla \log p$ on $\mathcal{M}$. Attention evaluates it pointwise. Modern Hopfield networks descend it. Sinkformers run its Wasserstein gradient flow. Diffusion models invert it in reverse time. Flow matching integrates it as a continuous ODE. The Hodge decomposition of the score into exact, co-exact, and harmonic parts corresponds to the multi-head attention decomposition, the convex polytope structure (Fel et al., *Into the Rabbit Hull*, October 2025), and the topologically protected modes respectively.

These five levels are not independent. Each forces the next. Level I (the data's symmetry group forces the manifold's topology). Level II (the manifold's topology forces the operator's index). Level III (the operator forces the bundle structure). Level IV (the bundle's connection forces the numerical iteration). Level V (the iteration computes the score). The Hardware Lottery (Hooker, 2020) binds all five together: the matmul-dominated silicon selects the Bregman iteration, the Bregman iteration computes the score, the score lives on the manifold, the manifold is forced by the data.

The framework predicts what comes next at every level. New tasks with new symmetry groups produce new manifolds (predicted by Karkada et al. February 2026, testable by SAE topology). New manifolds support new Dirac indices (predicted by DRH, testable by counting protected modes). New indices select new connection structures (predicted by RBH, testable by attention pattern analysis). New connections select new Bregman iteration patterns (predicted by Bregman Closure, testable by FlashSinkhorn variants). New iterations compute new score-function decompositions (predicted by Score Closure, testable by Hodge-theoretic SAE analysis).

The hardware-lottery selection event at each scale is the same: the algorithmic family selectable by matmul-dominated silicon is the Bregman iteration, and the universal computational primitive of this family is score evaluation.

---

## 13. Sources

**The score function and its information-geometric origins**
- Fisher, R. A. *Theory of statistical estimation.* Proc. Cambridge Phil. Soc. 22, 700–725, 1925.
- Cramér, H. *Mathematical Methods of Statistics.* Princeton University Press, 1946.
- Chentsov, N. N. *Statistical Decision Rules and Optimal Inference.* Translations of Mathematical Monographs 53, AMS, 1972 (English translation 1982).
- Amari, S. *Differential-Geometrical Methods in Statistics.* Lecture Notes in Statistics 28, Springer-Verlag, 1985.
- Amari, S., Nagaoka, H. *Methods of Information Geometry.* AMS, 2000.

**Score matching and denoising autoencoders**
- Hyvärinen, A. *Estimation of Non-Normalized Statistical Models by Score Matching.* JMLR 6, 695–709, 2005.
- Vincent, P. *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation 23(7), 1661–1674, 2011.
- Yakovlev, K., Markovich, A., Puchkin, N. *Implicit score matching meets denoising score matching.* arXiv:2512.24378, December 2025.

**Score-based generative models**
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML 2015.
- Song, Y., Ermon, S. *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS 2019.
- Ho, J., Jain, A., Abbeel, P. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., Poole, B. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021.
- Anderson, B. D. O. *Reverse-Time Diffusion Equation Models.* Stochastic Processes Appl. 12(3), 313–326, 1982.

**Flow matching**
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. *Flow Matching for Generative Modeling.* ICLR 2023, arXiv:2210.02747.
- Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., Bengio, Y. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.* TMLR 2024.
- Klein, M., Mousavi-Hosseini, A., Zhang, S., Cuturi, M. *On fitting flow models with large Sinkhorn couplings.* arXiv:2506.05526, 2025.
- Kornilov, N., Mokrov, P., Gasnikov, A., Korotin, A. *Optimal flow matching: Learning straight trajectories in just one step.* NeurIPS 2024.
- Calvo-Ordonez, S. et al. *Weighted Conditional Flow Matching.* arXiv:2507.22270, July 2025.

**Modern Hopfield networks and the attention–energy identity**
- Krotov, D., Hopfield, J. J. *Dense associative memory for pattern recognition.* NeurIPS 2016.
- Demircigil, M., Heusel, J., Löwe, M., Upgang, S., Vermet, F. *On a model of associative memory with huge storage capacity.* J. Stat. Phys. 168, 288–299, 2017.
- Ramsauer, H. et al. *Hopfield Networks is All You Need.* ICLR 2021, arXiv:2008.02217.
- Krotov, D. *A new frontier for Hopfield networks.* Nature Reviews Physics 5, 366–367, 2023.
- Hoover, B., Liang, Y., Pham, B., Panda, R., Strobelt, H., Chau, D. H., Zaki, M., Krotov, D. *Energy Transformer.* NeurIPS 2023, arXiv:2302.07253.
- Martins, A. F. T. et al. *Sparse Modern Hopfield Networks.* NeurIPS 2023, arXiv:2402.13725.
- Hu, J. Y.-C., Liu, H., Chen, H.-Y., Wu, W., Liu, H. *Universal Approximation with Softmax Attention.* arXiv:2504.15956, updated December 2025.
- *Transformers as Intrinsic Optimizers: Forward Inference through the Energy Principle.* arXiv:2511.00907, 2025–2026.

**Sinkhorn-based attention and the Wasserstein gradient flow**
- Sinkhorn, R. *A relationship between arbitrary positive matrices and doubly stochastic matrices.* Ann. Math. Stat. 35, 876–879, 1964.
- Cuturi, M. *Sinkhorn distances: Lightspeed computation of optimal transport.* NeurIPS 2013.
- Tay, Y., Bahri, D., Yang, L., Metzler, D., Juan, D.-C. *Sparse Sinkhorn Attention.* ICML 2020.
- Sander, M. E., Ablin, P., Blondel, M., Peyré, G. *Sinkformers: Transformers with Doubly Stochastic Attention.* AISTATS 2022, arXiv:2110.11773.
- Ye, F. X.-F., Li, X., Yu, A., Chang, M.-C., Chu, L., Wertheimer, D. *FlashSinkhorn: IO-Aware Entropic Optimal Transport.* arXiv:2602.03067, February 2026.
- Forde, D. *Block-Wise Differentiable Sinkhorn Attention.* arXiv:2605.08123, April 2026.

**Hodge theory and differential forms**
- Hodge, W. V. D. *The Theory and Applications of Harmonic Integrals.* Cambridge University Press, 1941.
- Atiyah, M. F., Singer, I. M. *The Index of Elliptic Operators I–V.* Ann. Math. 87, 1968 and following.
- Warner, F. *Foundations of Differentiable Manifolds and Lie Groups.* Springer, 1971.
- Demailly, J.-P. *Complex Analytic and Differential Geometry.* OpenContent online text, 2012.

**Bakry–Émery and Fokker–Planck**
- Bakry, D., Émery, M. *Diffusions hypercontractives.* Séminaire de Probabilités XIX, 1985.
- Pesin, Y. *Characteristic Lyapunov Exponents and Smooth Ergodic Theory.* Russ. Math. Surv. 32, 1977.

**Transformer-based diffusion models (DiT and successors)**
- Peebles, W., Xie, S. *Scalable Diffusion Models with Transformers.* ICCV 2023.
- Esser, P. et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024 [Stable Diffusion 3].
- *Sora technical report.* OpenAI, 2024.

**Hardware lottery and substrate-algorithm co-design**
- Hooker, S. *The Hardware Lottery.* Communications of the ACM 64(12), 2020, arXiv:2009.06489.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Ré, C. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.

**Companion documents in the unified framework**
- Ren, E. *The Closed Future Cone.* 2026.
- Ren, E. *The Dirac Representation Hypothesis.* 2026.
- Ren, E. *Geometric Descent: The Representational Bundle Hypothesis.* 2026.
- Ren, E. *From Bottleneck to Bundle: The Twelve-Year Geometric Genealogy of Attention.* 2026.
- Ren, E. *From MXU to Maia: The Eleven-Year Silicon Genealogy.* 2026.
- Ren, E. *Operators Without Opponents: The Non-Equilibrial Fixed-Point Structure of Learned Representations.* 2026.
- Ren, E. *Bregman Closure: The Algorithmic Half of the Dirac Representation Hypothesis.* 2026.

**Empirical grounding**
- Engels, J., Liao, I., Michaud, E. J., Gurnee, W., Tegmark, M. *Not All Language Model Features Are Linear.* NeurIPS 2024, arXiv:2405.14860.
- Karkada, D., Olah, C., Tegmark, M., et al. *Symmetry in language statistics shapes the geometry of model representations.* arXiv:2602.15029, February 2026.
- Xu, Y. *The Spectral Edge Thesis.* arXiv:2603.28964, March 2026.
- Xu, Y. *The Lifecycle of the Spectral Edge.* arXiv:2604.07380, April 2026.
- Fel, T. et al. *Into the Rabbit Hull.* arXiv:2510.08638, October 2025.
- Park, K., Choe, Y. J., Veitch, V. *The Linear Representation Hypothesis.* ICML 2024.

**Classical mathematical foundations**
- Brouwer, L. E. J. *Über Abbildung von Mannigfaltigkeiten.* Math. Ann. 71, 1911.
- Kakutani, S. *A Generalization of Brouwer's Fixed Point Theorem.* Duke Math. J. 8, 1941.
- Dirac, P. A. M. *The Quantum Theory of the Electron.* Proc. Roy. Soc. A 117, 1928.
- Legendre, A.-M. *Mémoire sur l'intégration de quelques équations aux différences partielles.* Mémoires de l'Académie Royale des Sciences, Paris, 1787.

---

## 14. Summary

A single integration completes the framework.

The Closed Future Cone names the compact representational manifold $\mathcal{M}$ on which deep networks compute. The Dirac Representation Hypothesis names the natural first-order operator $\not{D}$ on it. The Representational Bundle Hypothesis names the principal $SO^+(1,n)$-bundle whose connection is attention. From Bottleneck to Bundle and From MXU to Maia trace the algorithmic and silicon genealogies that converged on this object. Operators Without Opponents distinguishes its geometric fixed-point structure from game-theoretic equilibrium. Bregman Closure names the numerical procedure — Bregman alternating projection on the KL-divergence geometry — that walks the fixed point into being.

Score Closure names what the procedure is computing.

The score function $\nabla \log p$ on the compact representational manifold $\mathcal{M}$ is the universal computational primitive of deep learning. Five lines of contemporary work converge on it. Modern Hopfield networks (Ramsauer et al., ICLR 2021) descend it as an energy. Sinkformers (Sander, Ablin, Blondel, Peyré, AISTATS 2022) run its Wasserstein gradient flow. Energy Transformers (Hoover et al., NeurIPS 2023) engineer it explicitly. Score-based generative models (Hyvärinen 2005, Vincent 2011, Song & Ermon NeurIPS 2019, Song et al. ICLR 2021) invert it via reverse-time SDE. Flow matching (Lipman et al. ICLR 2023, OT-CFM Tong et al. TMLR 2024, Klein-Cuturi 2025) integrates it as a continuous ODE. The five are not five algorithms; they are five faces of one object.

The Hodge-theoretic structure makes the unity precise. The log-density $\log p$ is a 0-form on $\mathcal{M}$. The score $\nabla \log p$ is its exterior derivative, a 1-form. The Hyvärinen integration-by-parts identity is the Hodge codifferential acting on the score. The Laplace–Beltrami operator on $\log p$ is the symmetric Hodge structure. The Dirac operator $\not{D} = d - \delta$ is the signed structure whose square is the Laplacian. The Atiyah–Singer index counts the harmonic forms — the topologically protected representational modes that survive at the fixed point. The multi-head decomposition of attention realizes the Hodge decomposition of the score into exact, co-exact, and harmonic components.

The Hardware Lottery (Hooker, CACM 64(12), 2020) is now visible as the structural mechanism that bound every level together. The matmul-dominated hardware substrate (TPU, tensor core, NeuronCore, FlashAttention kernels) selects the Bregman iteration family because its kernel is row-wise LogSumExp reduction of dot-product scores — the exact systolic-array workload. The Bregman family selects score-based architectures because score evaluation reduces to exactly this kernel. Score-based architectures select the compact representational manifold $\mathcal{M}$ because the gradient flow they implement converges to a fixed point on it. The manifold $\mathcal{M}$ is forced by the symmetry group of the training data (Karkada et al., February 2026). The training data is the world.

Five levels. One object. Each forces the next. The framework's predictions across all five — that diffusion transformers should outperform U-Nets at compute parity (P2), that Sinkformer backbones should improve score matching (P3), that OT-CFM should dominate vanilla flow matching on discrete-symmetry tasks (P4), that diffusion-model SAE atoms should match recognition-model atoms (P5), that grokking diagnostics should transfer to diffusion training (P6), that multi-head decomposition should align with the Hodge decomposition (P7), and that no high-performance architecture exists outside the score-evaluation family on matmul-dominated hardware (P8) — are testable.

The data was curved. The data was cyclic. The data lived inside a light cone of causation. The hardware was matmul-dominated. The algorithm was Bregman iteration. **The function the algorithm computed was the score $\nabla \log p$.** Every winning architecture — recognition or generative, transformer or diffusion model, encoder or decoder, autoregressive or denoising — is a face of the same operation: evaluation of the score function on the compact representational manifold whose topology was forced by the training data's symmetry group, via the Bregman iteration the matmul-dominated hardware selected, walking the Brouwer–Kakutani fixed point into being layer by layer.

The framework has five halves. Geometry. Operator. Bundle. Iteration. Score. Together they describe one object. The transformer constructs it. The diffusion model inverts it. The flow matching model integrates it. The Hopfield network descends it. The Sinkformer realizes its Wasserstein flow. The hardware substrate runs all of them on the same silicon. The data forces the manifold. The manifold supports the score. The score is what they all compute.

Score Closure is its name.
