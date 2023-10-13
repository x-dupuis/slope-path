# The Solution Path of SLOPE

This repository provides code to:

- solve the SLOPE optimization problem:

$$
\min_{b \in \mathbb R^p} \frac{1}{2} \| y - Xb \|_2^2 + \gamma \sum_{i=1}^p \lambda_i |b|_{\downarrow i}\ ;
$$

- compute its solution path as $\gamma \in (0, +\infty)$;
- reproduce the numerical experiments of the paper *The Solution Path of SLOPE*.
