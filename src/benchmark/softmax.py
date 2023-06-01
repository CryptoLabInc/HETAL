"""This is a script for comparing precisions of softmax approximations."""
import numpy as np
import argparse
from tqdm import tqdm
from typing import List


def sample_input(num_samples: int, num_classes: int, sr: List[float]) -> np.ndarray:
    x = None
    for r_ in sr:
        x_ = np.random.uniform(-r_, r_, size=(num_samples, num_classes))
        if x is None:
            x = x_
        else:
            x = np.concatenate((x, x_), axis=0)
    return x

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    y = np.exp(x)
    y = y / np.sum(y, axis=1, keepdims=True)
    return y

def approx_exp(x, scale_pow_param: int = 3, deg: int = 8):
    x = x / (2 ** scale_pow_param)
    if deg == 8:
        coeffs = [
            1.000000001945585,
            0.9999999388932228,
            0.499999859398392,
            0.16666798775227673,
            0.04166711228500117,
            0.008328336039797088,
            0.0013881473292497602,
            0.00020492629147503022,
            2.551385701569812e-05,
        ]
    elif deg == 12:
        coeffs = [
            1.00000000e+00,
            1.00000000e+00,
            5.00000000e-01,
            1.66666667e-01,
            4.16666667e-02,
            8.33333323e-03,
            1.38888888e-03,
            1.98413061e-04,
            2.48016226e-05,
            2.75510749e-06,
            2.75519884e-07,
            2.55652816e-08,
            2.12732465e-09,
        ]
    r = coeffs[-1]
    for i in range(len(coeffs) - 1, -1, -1):
        r *= x
        r += coeffs[i]
    for _ in range(scale_pow_param):
        r *= r
    return r

# Our softmax approximation algorithm
def domain_extension(x, r: float, dei: int, der: float, precise: bool = False):
    for i in range(dei - 1, -1, -1):
        x = x - (4 / 27) * (1 / (r * r * (der ** (2 * i)))) * (x ** 3)
    if precise:
        c = (der ** (2 * dei) - 1) / ((der ** (2 * dei - 2)) * (der * der - 1)) * (4 / 27)
        x3 = x ** 3
        x5 = x ** 5
        x = x + (c / (r ** 2)) * x3 - (c / (r ** 4)) * x5
    return x

def approx_inv(x, R: float, n: int):
    y = 2 - x / R
    tmp = y - 1
    for _ in range(n):
        tmp *= tmp
        y *= 1 + tmp
    y /= R
    return y

def approx_comp(a, b, f_iter: int, g_iter: int):
    f = lambda x: (-5/16) * (x ** 7 - (21 / 5) * (x ** 5) + 7 * (x ** 3) - 7 * x)
    g = lambda x: (-12860/1024) * (x ** 7 - (25614 / 12860) * (x ** 5) + (16577 / 12860) * (x ** 3) - (4589 / 12860) * x)
    r = a - b
    for _ in range(g_iter):
        r = g(r)
    for _ in range(f_iter):
        r = f(r)
    r = (r + 1) / 2
    return r

def approx_max(a, b, f_iter: int, g_iter: int):
    comp = approx_comp(a, b, f_iter=f_iter, g_iter=g_iter)
    return a * comp + b * (1 - comp)

def approx_max_row(a: np.ndarray, f_iter: int, g_iter: int):
    # Row-wise approximated max
    # Assume that a is 2-power padded
    logdim = int(np.log2(a.shape[1]))
    ma = a
    for i in range(logdim):
        ma = approx_max(ma, np.roll(ma, -(1 << i)), f_iter=f_iter, g_iter=g_iter)
    return ma[:, :1]

def approx_softmax_original(x, inv_iter: int, inv_R: float):
    # Softmax approximation without normalization nor domain extension
    # Only works well for small interval
    x_exp = np.exp(x)
    z = np.sum(x_exp, axis=1, keepdims=True)
    z = approx_inv(z, R=inv_R, n=inv_iter)
    r = x_exp * z
    return r

def approx_softmax_wide(x, dei: int, der: float, inv_iter: int, inv_R: float, f_iter: int = 3, g_iter: int = 8, r: float = 8, precise: bool = False):
    # Softmax approximation with normalization and domain extension
    Rmax = (r / 2) * (der ** dei)
    num_class = x.shape[1]
    pad_num_class = 1 << int(np.ceil(np.log2(num_class)))

    # normalize
    x_pad = x / (2 * Rmax)
    x_pad = np.pad(x_pad, ((0, 0), (0, pad_num_class - num_class)), constant_values=-1/2)
    mx = approx_max_row(x_pad, f_iter=f_iter, g_iter=g_iter) * (2 * Rmax)
    x_norm = x - mx
    x_clip = domain_extension(x_norm, r=r, dei=dei, der=der, precise=precise)
    x_exp = approx_exp(x_clip)
    z = np.sum(x_exp, axis=1, keepdims=True)
    z = approx_inv(z, R=inv_R, n=inv_iter)
    r = x_exp * z
    return r


# Lee et al. "Privacy-preserving machine learning with fully homomorphic encryption for deep neural network"
def approx_gumbel_softmax(x, inv_iter: int = 8, inv_R: float = 10000, lamb: float = 4.0):
    x = x / lamb
    x_exp = approx_exp(x, scale_pow_param=6, deg=12)
    z = np.sum(x_exp, axis=1, keepdims=True)
    z = approx_inv(z, R=inv_R, n=inv_iter)
    r = x_exp * z
    return r


# Hong et al. "Secure tumor classification by shallow newral network using homomorphic encryption"
def approx_hong_exp(x, r: int, L: float):
    return ((x + 2 ** r) / L) ** (2 ** r)

def approx_hong_softmax(x, exp_r: int = 4, exp_L: float = 32, inv_iter: int = 30, inv_R: float = 80):
    x_exp = approx_hong_exp(x, r=exp_r, L=exp_L)
    z = np.sum(x_exp, axis=1, keepdims=True)
    z = approx_inv(z, R=inv_R, n=inv_iter)
    r = x_exp * z
    return r


# Jin et al. "Secure transfer learning for machine fault diagnosis under different operating conditions"
def approx_sigmoid_privgd(x):
    return 0.5 + 0.15012 * x - 0.00156 * (x ** 3)

def approx_privgd_softmax(x):
    r = []
    for i in range(x.shape[1]):
        t = x[:, i:i+1] - x
        t = approx_sigmoid_privgd(t)
        pi = 1
        for j in range(x.shape[1]):
            if j != i:
                pi *= t[:, j]
        
        r.append(np.expand_dims(pi, axis=1))
    r = np.concatenate(r, axis=1)
    return r


def run_softmax_approx(
    num_classes: int,
    sr: List[float],
    dei: int,
    inv_R: float,
    f_iter: int = 1,
    g_iter: int = 2,
    num_samples: int = 10000000,
    algorithms = ["normal", "precise", "gumbel", "hong", "privgd"],
):
    der = 2.0
    inv_iter = 16
    cover_r = max(sr)
    print(f"input range: [-{cover_r:.4f},{cover_r:.4f}]")
    print(f"sr: {sr}")

    div = 100

    max_err = {alg: 0.0 for alg in algorithms}
    avg_err = {alg: 0.0 for alg in algorithms}
    errs = {alg: [] for alg in algorithms}
    total_num_samples = num_samples * len(sr)
    print("num samples", total_num_samples)
    for _ in tqdm(range(div)):
        x = sample_input(num_samples // div, num_classes, sr=sr)
        y_true = softmax(x)

        for alg in algorithms:
            if alg == "original":  # without domain extension
                y_approx = approx_softmax_wide(x, dei=0, der=1, inv_iter=inv_iter, inv_R=inv_R)
            if alg == "normal":
                y_approx = approx_softmax_wide(x, dei, der, inv_iter, inv_R, f_iter=f_iter, g_iter=g_iter, precise=False)
            elif alg == "precise":
                y_approx = approx_softmax_wide(x, dei, der, inv_iter, inv_R, f_iter=f_iter, g_iter=g_iter, precise=True)
            elif alg == "gumbel":
                y_approx = approx_gumbel_softmax(x)
            elif alg == "hong":
                y_approx = approx_hong_softmax(x)
            elif alg == "privgd":
                y_approx = approx_privgd_softmax(x)

            errors = np.max(np.abs(y_true - y_approx), axis=1)
            max_err[alg] = max(np.max(errors), max_err[alg])
            avg_err[alg] += np.sum(errors)
            errs[alg] += errors.tolist()

    for alg in algorithms:
        avg_err[alg] /= total_num_samples
        print(f"algorithm {alg}: max error {max_err[alg]}, avg error {avg_err[alg]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=int)
    parser.add_argument("--dei", type=int, default=5)
    parser.add_argument("--ns", type=int, default=100000000)
    parser.add_argument("--fi", type=int, default=1)
    parser.add_argument("--gi", type=int, default=2)
    parser.add_argument("--iR", type=float, default=100)
    parser.add_argument("--sr", type=float, default=4)
    args = parser.parse_args()

    sr = args.sr
    if sr == 4:
        algorithms = ["original", "normal", "precise", "gumbel", "hong", "privgd"]  # sr = 4
    elif sr == 8:
        algorithms = ["normal", "precise", "gumbel", "hong"]  # sr = 8
    elif sr == 32:
        algorithms = ["normal", "precise", "gumbel"]  # sr = 32
    elif sr == 128:
        algorithms = ["normal", "precise"] # sr = 128
    print("algorithms: ", algorithms)

    if sr == 4:
        sr = [4]
    elif sr == 8:
        sr = [4, 8]
    elif sr == 32:
        sr = [4, 8, 32]
    elif sr == 128:
        sr = [4, 8, 32, 128]
    else:
        raise ValueError(f"invalid sr: {sr}. Should be one of 4, 8, 32, 128")

    print("c", args.nc)
    np.random.seed(0)
    run_softmax_approx(
        num_classes=args.nc,
        sr=sr,
        dei=args.dei,
        inv_R=args.iR,
        f_iter=args.fi,
        g_iter=args.gi,
        num_samples=args.ns,
        algorithms=algorithms,
    )
