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
