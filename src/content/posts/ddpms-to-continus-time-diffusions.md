---
title: "Diffusion models, from DDPMs to continuous time diffusion"
date: 2026-02-26
authors:
  - name: Aaron Cao
    link: https://sites.google.com/view/caozhijun/home/
    image: https://lh3.googleusercontent.com/sitesv/APaQ0SR3CN2St-lhG_c7xRBKLDowSqf8EpjHD-tHzZ8rcD1uz87yCAgCrsVkXFNTTADcxON6lR-WjFXIMNXZXw-2jc5xxpWLed7zCiYFUqUXRu3KmUv2PkTJHvmKjL6CjdC6lb2WwZO8h-_DmlrgFRRpW-bd1otVTy69vVVC9Qgo09efBaEzcoiip8tKb9E=w1280
  - name: Chen Pinghong
  - name: Huang Zifeng
    link: https://github.com/Kowyo
    image: https://github.com/Kowyo.png
  - name: Leo
  - name: easymoneysniper
---

# Part 1: Background and Motivation

## The Reign of Diffusion Models

We have officially entered the era where Generative AI synthesizes incredibly high-fidelity images, audio, and physical states. The journey began with discrete Denoising Diffusion Probabilistic Models (DDPMs), but rapidly shifted towards an elegant, continuous-time framework: Score-based SDEs and flow-based models like Rectified Flow, which construct deterministic, straight-line ODE trajectories for vastly faster sampling.

Hugging Face's `diffusers` library is the undisputed gold standard for building, distributing, and running inference on diffusion models. Its elegant Pipeline abstraction wraps U-Nets, DiTs, ControlNets, and schedulers into just a few lines of code, dramatically lowering the barrier to generative AI development.

Yet this same abstraction is a double-edged sword. The sampling process is mathematically equivalent to solving an ODE/SDE, but diffusers buries these integration steps beneath layers of object-oriented Schedulers. For researchers and engineers who need fine-grained control over solver parameters, step-level observability, or aggressive inference optimization, the encapsulation becomes a bottleneck—forcing them to spend valuable time navigating Pipeline internals just to reach the underlying numerical mechanics.

## Returning to Mathematical First Principles

If generative modeling is fundamentally about evolving probability distributions via differential equations driven by neural networks, why not build it in an environment where differential equations are first-class citizens? This question led us to Julia. By leveraging Julia's world-leading `SciML`/`DifferentialEquations.jl` ecosystem and elegant deep learning via `Flux.jl`, we can strip away the black box. Our goal is to demonstrate how replacing complex Python scheduling logic with transparent, mathematically pure ODE/SDE solvers yields a codebase that is not only highly modular but essentially reads like math.

# Part 2: Principle Review — From Discrete Steps to Continuous Manifolds

Before diving into the code, we must trace the mathematical evolution that underpins modern diffusion models. The progression from discrete DDPM steps to continuous-time SDEs and ODEs is far more than theoretical elegance—it reveals a profound unification. At its core, diffusion models are not merely a bag of tricks for image synthesis; they are a principled framework for learning and manipulating probability distributions.

## The Discrete Origin: DDPM

Introduced by Sohl-Dickstein et al. (2015) and brought to prominence by Ho et al. (2020), DDPM defines a forward Markov chain that gradually corrupts data $x_0$ with Gaussian noise over $T$ steps via a variance schedule $\{\beta_t\} $.  A key insight is that the marginal at any timestep admits a tractable closed form via the reparameterization trick:
$$
q(x_t|x_0) = \mathcal{N}(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1 - \bar{\alpha}_t) I)
$$
The reverse process $p_\theta $ inverts this corruption. Although $q(x_{t-1}|x_t) $ is intractable, Ho et al. showed that conditioning on $x_0 $ yields a closed-form Gaussian posterior $q(x_{t-1}|x_t, x_0) $, and that training a neural network $\epsilon_\theta(x_t, t) $ to predict the added noise via a simplified ELBO reduces to:
$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$
At inference, $p_\theta $ recovers $x_{t-1} $ by first estimating $\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta) $, then sampling from the posterior:
$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\ \mu_\theta(x_t, t),\ \tilde{\beta}_t I)
$$

<img>

Despite achieving remarkable sample quality, this iterative denoising over $T \approx 1000$ steps is computationally costly at inference—a bottleneck addressed by DDIM (Song et al., 2020), which reinterprets the reverse process as a deterministic non-Markovian mapping, slashing sampling steps by an order of magnitude without retraining. More fundamentally, the discrete formulation obscures what continuous-time limit these finite steps are approximating—a question whose answer, as we will see, unlocks a far richer mathematical framework.

## Moving to Continuous Dynamics: Score-Based SDEs

As $N \to \infty $ (i.e., $\Delta t \to 0 $), the discrete Markov chain naturally converges to a continuous-time SDE. Song et al. (2021) recognized that virtually all diffusion model variants—DDPMs, NCNSs, and beyond—can be unified under this single framework by choosing different drift and diffusion coefficients. For the **Variance Preserving (VP-SDE)** case, the forward process is:
$$
dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw
$$
where $dw $ denotes standard Brownian motion and $\beta(t) $ is a continuous noise schedule. The perturbation kernel remains Gaussian at all times, with closed-form mean and variance governed by $\alpha(t) $ and $\sigma(t) $.

## Turning Back Time: The Reverse SDE and the Score Function

Building on Anderson (1982), Song et al. showed that for any forward Itô SDE with sufficiently regular coefficients, an exact time-reversal exists. For the VP-SDE, the reverse process (time flowing from $1 \to 0 $) takes the form:
$$
dx = \left[-\frac{1}{2}\beta(t)x - \beta(t)\nabla_x \log p_t(x)\right]dt + \sqrt{\beta(t)}\,d\bar{w}
$$

<img>

The central quantity is $\nabla_x \log p_t(x)$—the **score function**—which points toward regions of higher data density and fully determines the reverse dynamics. Crucially, it is independent of the unknown normalizing constant of $p_t(x) $, making it learnable. A neural network $s_\theta(x_t, t) $ is trained via **denoising score matching** (Vincent, 2011):
$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, x_0, \epsilon}\left[\lambda(t)\left\|s_\theta(x_t, t) - \nabla_{x_t}\log q(x_t|x_0)\right\|^2\right]
$$
Since the perturbation kernel $q(x_t|x_0) $ is Gaussian, the target score has a closed form: $\nabla_{x_t}\log q(x_t|x_0) = -\epsilon/\sigma(t) $, which directly connects back to the noise prediction network of DDPM via $s_\theta(x_t, t) = -\epsilon_\theta(x_t, t)/\sigma(t) $.

## From Randomness to Certainty: Probability Flow ODE

The reverse SDE generates samples via a score-guided drift corrupted by Brownian noise $d\bar{w}$. However, a remarkable result by Song et al. (2021) proved this randomness is strictly optional. For any diffusion SDE, there exists a unique, deterministic Ordinary Differential Equation (ODE) that induces the exact same marginal distributions $p_t(x)$:
$$
dx = \left[-\frac{1}{2}\beta(t)x - \frac{1}{2}\beta(t)\, s_\theta(x, t)\right] dt
$$
By discarding the stochastic term entirely and halving the score coefficient, we obtain the **Probability Flow ODE**. Integrating this ODE backward in time (from noise at $t=1$ to data at $t=0$) yields a clean sample along a remarkably smooth path.

This formulation carries two profound consequences. First, it reframes generation as a pure numerical integration problem, solvable with off-the-shelf ODE solvers using significantly fewer steps than DDPM's 1000-step chain. Second, it establishes a bijective mapping between noise and data, enabling exact likelihood computation and seamless latent manipulation (with DDIM acting as an early discrete approximation).

## Rectified Flow and Flow-Based Training Paradigm

Lipman et al. (2022) and Liu et al. (2022) independently introduced Flow Matching and **Rectified Flow**. They proposed directly regressing a neural network $v_\theta $ onto a target vector field connecting noise $x_{\text{noise}}$ to data $x_{\text{data}}$. By defining a simple linear interpolation $x_t = t x_{\text{data}} + (1-t) x_{\text{noise}}$ (where time $t$ flows from $0$ to $1$), the training objective becomes elegantly straightforward:
$$
\mathcal{L}_{\text{Flow}} = \mathbb{E}_{t, x_{\text{data}}, x_{\text{noise}}}\left[\|v_\theta(x_t, t) - (x_{\text{data}} - x_{\text{noise}})\|^2\right]
$$
The key insight here is that straight-line trajectories are not just computationally convenient—they are optimal. Linear paths have zero curvature, requiring drastically fewer neural function evaluations (NFEs) to integrate accurately. Rectified Flow further introduced Reflow, an iterative procedure that re-trains the model on its own generated pairs to continuously straighten the trajectories toward a one-step mapping.

<img>

This philosophy—direct vector field regression on straight paths—has now cemented itself as a training paradigm, powering state-of-the-art models like Stable Diffusion 3.5 (by Stability AI), Bagel (by ByteDance) and FLUX (by Black Forest Labs).

# Part 3: Julia Implementation of Continuous-time Diffusion

## Implement SDE-based Diffusion with Brown Bridge

### Core Code

### Visualization

## Implement ODE-based Diffusion with Rectified Flow

### Core Code

### Visualization

### Discussion on Samplers

# Part 4: What else can Julia do, and where are its limitations?

## Generative AI Usage

## Contributions

## Reference
