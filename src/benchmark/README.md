# Benchmark

## Matrix multiplication

Run `matmul.py`:
```
python matmul.py --algorithm hetal
```
The python script includes our implementations of the baseline matrix multiplication algorithms from the following papers:
* Crockett, "A low-depth homomorphic circuit for logistic regression model training"
* Jin et al., "Secure transfer learning for machine fault diagnosis under different operating conditions"

Here are descriptions of the arguments:

| Argument  | Description |
|-----------|-------------|
| `--algorithm ALGORITHM`    | Algorithm name. `hetal` (DiagABT and DiagATB), `crockett` (ColMajor, Row Major), `privgd`. |
| `--num_iter NUM_ITER`   | Number of iterations for measurements. Defaults to `10`. |

**Update (March 7, 2024)**
- There was a bug in the internal code that counts the number of CMult & Mult used. This problem is fixed in PR #5, and the numbers will be different from the original numbers in Table 7 of the published version of the paper (which are wrong).
The corrected table can be found in the pdf file `hetal_matmul_table_fixed.pdf`. Table 6 in the same pdf file shows the *exact* number of each operation in terms of matrix sizes and $s_0, s_1$.
- Also, the experiments are done with the highest level of the CKKS parameter we used in the paper, which is 12 for both $A$ and $B$.
However, when the level of $A$ is smaller than $B$, then the modified algorithm with $\mathsf{PRotUp}$ increases the number of rotations and constant multiplications (but not multiplications), and such numbers can be also found in the new Table 7 in the pdf file. Note that we save a depth by 1 in this case.
If you want to reproduce these numbers, add the following line before L287 of `matmul.py`:
    ```python
    mat1.level_down(11)
    ```

- We thank Miran Kim for pointing out these errors.

## Softmax approximation

`softmax.py` includes softmax approximation algorithms, implemented in numpy and run without encryption (so the resulting errors do not include those from CKKS scheme itself).

```
python softmax.py --nc 3 --sr 8
```

Here are descriptions of the arguments:

| Argument  | Description |
|-----------|-------------|
| `--nc NUM_CLASS`    | Input dimension of softmax, which corresponds to the number of classes in HETAL. |
| `--sr SAMPLE_RANGE`    | Sampling range of input. `4`, `8`, `32`, or `128`. |
| `--dei DOMAIN_EXTENSION_INDEX`   | Domain extension index for our approximation algorithm. Defaults to `5`. |
| `--ns NUM_SAMPLE` | Determine the number of random samples to estimate approximation error. Defaults to `100000000`. |
| `--fi FI` | Iteration number of `f` for `approx_comp`. Defaults to `1`. |
| `--gi GI` | Iteration number of `g` for `approx_comp`. Defaults to `2`. |
| `--iR INVERSE_R` | Scaling factor `R` for the inverse approximation. Defaults to `100`. |

See Appendix of the paper for details about input sampling.
