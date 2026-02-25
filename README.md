# FAST-LEARN-NG-AI
Using multiplicative calculus (non newtonian calculus) in ai
<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Project Title

Using Multiplicative calculus in ai for fast Machine learning 

## Summary

  it is actually a growing area of research! While standard AI (like Neural Networks) is built on Additive Calculus (gradient descent via $w - \eta \nabla L$), researchers have developed Multiplicative Backpropagation and Log-space Neural Networks that use multiplicative principles.


## Background
  It is exceptionally fast at "learning" which experts or features to trust in a massive dataset. It converges much faster than 
additive updates when you have a huge number of dimensions but only a few matter.

Geometric Deep Learning

  In standard Deep Learning, we assume the "data space" is flat (Euclidean). However, many types of data—like social networks, 
chemical molecules, or hierarchical taxonomies—grow exponentially.

Multiplicative calculus is the natural language for Hyperbolic Geometry.

  AI Application: Large Language Models (LLMs) often use hyperbolic embeddings to represent the hierarchy of language. 
Using multiplicative derivatives helps the model navigate these "curved" spaces more efficiently than standard calculus.

  Handling Multiplicative Noise
   
   In computer vision and signal processing, noise isn't always something that is "added" to a pixel (Additive White Gaussian Noise). 
Sometimes the noise is multiplicative (Speckle noise), common in Medical Ultrasound or SAR (Satellite) imaging.  

  AI Application: Using a multiplicative loss function allows a neural network to denoise these images much more effectively than a standard Mean Squared Error (MSE) loss.
  
## How is it used?

To see how this works in practice, let's compare a standard update to a multiplicative update using a simple Python snippet.

In AI, the "Multiplicative Weights Update" (MWU) is the most common implementation. It is often used in Adversarial Learning and Game Theory because it handles "expert advice" extremely well.

Python Code: Standard vs. Multiplicative UpdateThis code simulates a simple scenario: we have a weight $w$ that we want to optimize based on some "loss" (signal).
```
import numpy as np

def standard_update(w, gradient, lr=0.1):
    # Classical: w = w - lr * gradient
    return w - (lr * gradient)

def multiplicative_update(w, gradient, lr=0.1):
    # Multiplicative: w = w * exp(-lr * gradient)
    # This ensures w stays positive and scales relative to its size
    return w * np.exp(-lr * gradient)

# Initial Weight
w_std = 10.0
w_mult = 10.0
gradient = 0.5  # Assume we found a positive gradient (need to decrease weight)

print(f"Initial Weight: {w_std}")
print(f"Standard Update: {standard_update(w_std, gradient):.4f}")
print(f"Multiplicative Update: {multiplicative_update(w_mult, gradient):.4f}")
```
Why the difference matters

  Standard Update: If you set the learning rate too high, the weight could suddenly become negative, which makes no sense if $w$ represents something like "probability" or "importance."
  
  Multiplicative Update: The weight will get smaller and smaller, approaching zero, but it will never cross into negative territory. It preserves the "sign" of your parameters.


## Data sources and AI methods


## Challenges

What does your project _not_ solve? Which limitations and ethical considerations should be taken into account when deploying a solution like this?

## What next?

1- "Madam: A Multiplicative Version of the Adam Optimizer" – This paper shows that multiplicative updates can actually replace Adam in state-of-the-art neural networks without needing to tune the learning rate.
2- "Learning Compositional Functions via Multiplicative Weight Updates" – Discusses why standard gradients fail for deep, complex hierarchies and how multiplicative math fixes it.
3- "Consensus Multiplicative Weights Update" – A very recent look at how AI agents can reach "Nash Equilibrium" faster in multi-player games using this calculus.
## Acknowledgments

* list here the sources of inspiration 
* do not use code, images, data etc. from others without permission
* when you have permission to use other people's materials, always mention the original creator and the open source / Creative Commons licence they've used
  <br>For example: [Sleeping Cat on Her Back by Umberto Salvagnin](https://commons.wikimedia.org/wiki/File:Sleeping_cat_on_her_back.jpg#filelinks) / [CC BY 2.0](https://creativecommons.org/licenses/by/2.0)
* etc
