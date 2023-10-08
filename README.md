# ClosedLoopReachability.jl

| **Documentation** | **Status** | **Community** | **License** |
|:-----------------:|:----------:|:-------------:|:-----------:|
| [![docs-dev][dev-img]][dev-url] | [![CI][ci-img]][ci-url] [![codecov][cov-img]][cov-url] | [![zulip][chat-img]][chat-url] | [![license][lic-img]][lic-url] |

[dev-img]: https://img.shields.io/badge/docs-latest-blue.svg
[dev-url]: https://juliareach.github.io/ClosedLoopReachability.jl/dev/
[ci-img]: https://github.com/JuliaReach/ClosedLoopReachability.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/JuliaReach/ClosedLoopReachability.jl/actions/workflows/ci.yml
[cov-img]: https://codecov.io/github/JuliaReach/ClosedLoopReachability.jl/coverage.svg
[cov-url]: https://app.codecov.io/github/JuliaReach/ClosedLoopReachability.jl
[chat-img]: https://img.shields.io/badge/zulip-join_chat-brightgreen.svg
[chat-url]: https://julialang.zulipchat.com/#narrow/stream/278609-juliareach
[lic-img]: https://img.shields.io/github/license/mashape/apistatus.svg
[lic-url]: https://github.com/JuliaReach/ClosedLoopReachability.jl/blob/master/LICENSE

This package implements methods to analyze closed-loop control systems using reachability analysis.

Currently we support neural-network controllers.


## 📜 How to cite

If you use this package in your work, please cite it using the metadata [here](CITATION.bib) or below.

<details>
<summary>Click to see BibTeX entry. </summary>

```
@inproceedings{SchillingFG22,
  author    = {Christian Schilling and
               Marcelo Forets and
               Sebasti{\'{a}}n Guadalupe},
  title     = {Verification of Neural-Network Control Systems by Integrating {T}aylor
               Models and Zonotopes},
  booktitle = {{AAAI}},
  pages     = {8169--8177},
  publisher = {{AAAI} Press},
  year      = {2022},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/20790},
  doi       = {10.1609/aaai.v36i7.20790}
}
```

</details>
