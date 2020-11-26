# Allostasis

![status](https://img.shields.io/badge/status-development-orange)

## Figures

### Figure 1
Figure 1 presents a model of autonomic responses. Here, interoceptive prediction errors are minimized by autonomic reflexes.

### Figure 2
Figure 2 presents a model of anticipatory responses. Here, an exteroceptive cue signals an imminent change in interoceptive variables. The exterocepive modality acts as a prior probability of interocepive variables, leading to anticipatory responses. Crucially, a non-linear relationship between metabolic cost and autonomic responses means that anticipatory regulation induces benefits in overall metabolic cost. 

### Figure 3
Here, goal-directed behaviors are engaged to minimize interoceptive prediction errors. 

### Figure 4
This figure presents of model of interoceptive dysfunction. We encode a high entropy likelihood and show that suitable goal-directed behaviors are not engaged.

## Issues

The primary issue is the need for both state estimation and set points in the interoceptive domain. There are two solutions:

  - We separate state estimation from prior preferences. Two separate nodes trying to predict interoceptive variables, one with high variance and an active component (set point), and one with unit variance and no action (state estimation).
  - We have set point prior on state estimation variable, and deal with the fact that state estimation will reach equilibrium point and data

The second issue is the notion of interoceptive prediction errors during counterfactual decision making. We will be able to predict what interoceptive sensations we expect, but not clear how to encode prior (related to first issue). One option is that we simply encode prediction error in terms of the KL-divergence between prior and expected beliefs in discrete domain. The second is to separate state estimation from the prior. This would require that _expected beliefs in interoceptive domain predict the expected sensations, and not the set point_. Interoceptive errors are then measured in terms of prior beliefs. 