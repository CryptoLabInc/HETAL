import argparse
import timeit
from pathlib import Path

import numpy as np
from typing import Optional, Tuple, List

import heaan_sdk
from heaan_sdk.matrix.matrix import HESubMatrix
from heaan_sdk.matrix.ops import submat_ops as smop
from heaan_sdk import Block


# Jin et al. "Secure transfer learning for machine fault diagnosis under different operating conditions"
def column_pack(context: heaan_sdk.Context, arr: np.ndarray) -> List[Block]:
    block_list = []
    for i in range(arr.shape[1]):
        arr_i = arr[:, i]
        block_i = Block.from_ndarray(context, arr_i)
        block_list.append(block_i)
    return block_list


def column_depack(cp_mat: List[Block]) -> np.ndarray:
    arr = []
    for block in cp_mat:
        arr.append(block.to_ndarray())
    arr = np.vstack(arr).T
    return arr


def replicate_pack(context: heaan_sdk.Context, arr: np.ndarray) -> List[List[Block]]:
    block_list_list = []
    for i in range(arr.shape[0]):
        block_list = []
        for j in range(arr.shape[1]):
            arr_ij = arr[i][j]
            block_ij = Block.fill(context, value=arr_ij)
            block_list.append(block_ij)
        block_list_list.append(block_list)
    return block_list_list


def replicate_depack(rep_mat: List[List[Block]]) -> np.ndarray:
    arr = np.zeros(shape=(len(rep_mat), len(rep_mat[0])))
    for i, block_list in enumerate(rep_mat):
        for j, block in enumerate(block_list):
            arr_ij = block.to_ndarray()[0]
            arr[i][j] = arr_ij
    return arr


def encrypt_column_packed_matrix(cp_mat: List[Block]):
    for block in cp_mat:
        block.encrypt()


def decrypt_column_packed_matrix(cp_mat: List[Block]):
    for block in cp_mat:
        block.decrypt()


def encrypt_replicate_packed_matrix(rep_mat: List[Block]):
    for block_rows in rep_mat:
        for block in block_rows:
            block.encrypt()


def decrypt_replicate_packed_matrix(rep_mat: List[Block]):
    for block_rows in rep_mat:
        for block in block_rows:
            block.decrypt()


def privgd_abt(A: List[Block], B: List[List[Block]]) -> List[Block]:
    # CP x REP -> CP
    # X (n, f+1) x W (c, f+1) -> X * W^T (n, c)
    res = []
    print("privgd_abt", len(A), len(B))
    for b_row in B:
        res_col = None
        for a_col, b in zip(A, b_row):
            if res_col is None:
                res_col = a_col * b
            else:
                res_col += a_col * b
        res.append(res_col)
    return res


def privgd_atb(A: List[Block], B: List[Block]) -> List[List[Block]]:
    # CP x CP -> REP
    # P - Y (n, c) x X (n, f+1) -> (P - Y)^T * X (c, f+1)
    res = []
    for a in A:
        res_row = []
        for b in B:
            ab = a * b
            ab.rotate_sum(inplace=True)
            res_row.append(ab)
        res.append(res_row)
    return res


# Crockett, "A low-depth homomoprhic circuit for logistic regression model training"
# ColMajor and RowMajor algorithm
def _row_mask(context, k: int, unit_shape: Tuple[int, int], const: Optional[float] = None):
    mask = np.zeros(unit_shape)
    mask[k] = 1
    if const is not None:
        mask *= const
    mask = Block.from_ndarray(context, mask)
    return mask


def _col_mask(context, k: int, unit_shape: Tuple[int, int], const: Optional[float] = None):
    mask = np.zeros(unit_shape)
    mask[:, k] = 1
    if const is not None:
        mask *= const
    mask = Block.from_ndarray(context, mask)
    return mask


def colmajor(v1: HESubMatrix, v2: HESubMatrix, const: Optional[float] = None) -> HESubMatrix:
    res = None
    for k in range(v2.num_rows):
        mask_k = _row_mask(v2.context, k, v2.unit_shape, const)
        v2_mask = v2 * mask_k
        v2_tmp = smop.vertical_sum(v2_mask, direction=0, fill=True)
        res_k = v1 * v2_tmp
        res_k = smop.horizontal_sum(res_k, direction=0)
        res_k[0] >>= k
        if k == 0:
            res = res_k
        else:
            res += res_k
    res.shape = (v1.num_rows, v2.num_rows)
    return res


def rowmajor(v1: HESubMatrix, v2: HESubMatrix, const: Optional[float] = None) -> HESubMatrix:
    res = None
    unit_shape = v1.unit_shape
    for k in range(v1.num_cols):
        col_mask_k = _col_mask(v1.context, k, unit_shape, const)
        v1_mask = v1 * col_mask_k
        v1_mask[0] <<= k
        for j in range(int(np.log2(unit_shape[1]))):
            v1_mask[0] += v1_mask[0] >> (1 << j)

        res_k = HESubMatrix(v1.context, v1.unit_shape, shape=(v1.num_cols, v2.num_cols))
        for block in v2:
            res_k.append(v1_mask[0] * block)
        res_k = smop.vertical_sum(res_k, direction=0, fill=True)
        row_mask_k = _row_mask(v1.context, k, v1.unit_shape, const)
        res_k *= row_mask_k
        if k == 0:
            res = res_k
        else:
            res += res_k

    res.shape = (v1.num_cols, v2.num_cols)
    return 


class Duration:
    def __init__(self, name: str, context: heaan_sdk.Context, num_iter: int):
        self.name = name
        self.context = context
        self.num_iter = num_iter

        self.context._cmult_cpu_cnt = 0
        self.context._mult_cpu_cnt = 0
        self.context._rot_cpu_cnt = 0

    def __enter__(self):
        self.st = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = timeit.default_timer() - self.st
        duration /= self.num_iter
        print(f"{self.name}: {duration:.4f}s")
        print(f"- CMult(CPU): {self.context._cmult_cpu_cnt // num_iter}")
        print(f"- Mult(CPU): {self.context._mult_cpu_cnt // num_iter}")
        print(f"- Rot(CPU): {self.context._rot_cpu_cnt // num_iter}")


def setup():
    he_params = heaan_sdk.HEParameter.from_preset("FGb")
    context = heaan_sdk.Context(he_params, make_bootstrappable=True)
    key_dir_path = Path("./keys-matmul")
    rk_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 48, 64, 80, 96, 112, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32512, 32640, 32704, 32736, 32752, 32753, 32754, 32755, 32756, 32757, 32758, 32759, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767]
    context.set_key_dir_path(key_dir_path)
    context.generate_secret_key()
    context.generate_public_key(rk_indices=rk_indices)
    context.load_all_keys()
    context.generate_homevaluator()
    return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, choices=["hetal", "crockett", "privgd"])
    parser.add_argument("--num_iter", type=int, default=10)
    args = parser.parse_args()
    algorithm = args.algorithm
    num_iter = args.num_iter
    print(f"{algorithm=}, {num_iter=}")

    context = setup()

    configs = [
        (128, 128, 4),
        (256, 256, 8),
        (512, 769, 4),
        (1024, 769, 8),
        (2048, 769, 16),
    ]

    for a, b, c in configs:
        unit_shape = (a, context.num_slots // a)

        # A * B^T
        arr1 = np.random.rand(a, b)
        arr2 = np.random.rand(c, b)
        if algorithm != "privgd":
            tiled_arr2 = np.tile(arr2, (a // c, 1))
            mat1 = HESubMatrix.from_ndarray(context, arr1, unit_shape)
            mat2 = HESubMatrix.from_ndarray(context, tiled_arr2, unit_shape)
            mat2.shape = (c, b)
            mat1.encrypt()
            mat2.encrypt()

        if algorithm == "crockett":
            # Crockett, ColMajor
            _ = colmajor(mat1, mat2)
            with Duration(f"ColMajor({(a, b)} x {(b, c)})", context, num_iter):
                for _ in range(num_iter):
                    res_mat_colmajor = colmajor(mat1, mat2)
        elif algorithm == "privgd":
            # PrivGD
            mat_cp = column_pack(context, arr1)
            mat_rep = replicate_pack(context, arr2)
            encrypt_column_packed_matrix(mat_cp)
            encrypt_replicate_packed_matrix(mat_rep)
            _ = privgd_abt(mat_cp, mat_rep)
            with Duration(f"PrivGD ABT({(a, b)} x {(b, c)})", context, num_iter):
                for _ in range(num_iter):
                    res_mat_privgd_abt = privgd_abt(mat_cp, mat_rep)
        else:
            # hetal, DiagABT
            _ = smop.submat_mul_col_tiled(mat1, mat2)
            with Duration(f"DiagABT({(a, b)} x {(b, c)})", context, num_iter):
                for _ in range(num_iter):
                    res_mat_abt = smop.submat_mul_col_tiled(mat1, mat2)

        # A^T * B
        arr1 = np.random.rand(a, c)
        arr2 = np.random.rand(a, b)
        if algorithm != "privgd":
            tiled_arr1 = np.tile(arr1, (1, a // c))
            mat1 = HESubMatrix.from_ndarray(context, tiled_arr1, unit_shape)
            mat1.shape = (a, c)
            mat2 = HESubMatrix.from_ndarray(context, arr2, unit_shape)
            mat1.encrypt()
            mat2.encrypt()

        if algorithm == "crockett":
            # Crockett, RowMajor
            _ = rowmajor(mat1, mat2)
            with Duration(f"RowMajor({(c, a)} x {(a, b)})", context, num_iter):
                for _ in range(num_iter):
                    res_mat_rowmajor = rowmajor(mat1, mat2)
        elif algorithm == "privgd":
            # PrivGD
            mat_cp1 = column_pack(context, arr1)
            mat_cp2 = column_pack(context, arr2)
            encrypt_column_packed_matrix(mat_cp1)
            encrypt_column_packed_matrix(mat_cp2)
            _ = privgd_atb(mat_cp1, mat_cp2)
            with Duration(f"PrivGD ATB({(c, a)} x {(a, b)})", context, num_iter):
                for _ in range(num_iter):
                    res_mat_privgd_atb = privgd_atb(mat_cp1, mat_cp2)
        else:
            # hetal, DiagATB
            _ = smop.submat_mul_row_tiled(mat1, mat2)
            with Duration(f"DiagATB({(c, a)} x {(a, b)})", context, num_iter):
                for _ in range(num_iter):
                    res_mat_atb = smop.submat_mul_row_tiled(mat1, mat2)

        print("=" * 50)
