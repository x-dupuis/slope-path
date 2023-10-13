# The Solution Path of SLOPE

This repository provides code to:

- solve the SLOPE optimization problem:

```math
\min_{b \in \mathbb R^p} \frac{1}{2} \| y - Xb \|_2^2 + \gamma \sum_{i=1}^p \lambda_i |b|_{\downarrow i}\ ;
```

- compute its solution path as $\gamma \in (0, +\infty)$;
- minimize exactly the SURE formula along the path;
- reproduce the numerical experiments of the paper *The Solution Path of SLOPE*.

The `module` folder contains the code itself with:

- a solver (for a single $\gamma$) `path_solver` and a full path solver `full_path` for SLOPE;
- a full path solver `full_path_LASSO` for LASSO;
- SURE exact minimization routines `min_SURE` for SLOPE and `min_SURE_LASSO` for LASSO.

The `experiments` folder contains Python scripts and Jupyter notebooks which: 

- compute the full SLOPE solution paths on a toy example and on two real data sets: [`Wine Quality`](http://archive.ics.uci.edu/dataset/186/wine+quality) and [`Riboflavin`](https://www.annualreviews.org/doi/suppl/10.1146/annurev-statistics-022513-115545);
- minimize exactly the SURE formula for both SLOPE and LASSO on the `Wine Quality` data set;
- compare our full path solver to [`genlasso`](https://cran.r-project.org/web/packages/genlasso/index.html) R package to compute the solution path of OSCAR on the `Wine Quality` data set;
- compare our solver (for a single $\gamma$) to the algorithms considered by `Larsson et al.` for the benchmark of their paper [*Coordinate Descent for SLOPE*](https://proceedings.mlr.press/v206/larsson23a.html) (with their implementation available at <https://github.com/jolars/slopecd>) on the `Wine Quality` and `Riboflavin` data sets.

The `environment.yml` file provides a minimal conda environment required to run the code, **except for the subfolder `solvers_benchmarks` which requires to install the code developed by `Larsson et al.` and available at <https://github.com/jolars/slopecd>.**
