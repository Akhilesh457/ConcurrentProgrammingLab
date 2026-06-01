# CUDA Python Numba Experiments on Kaggle

This project contains three Jupyter notebooks for CUDA programming using Python, Numba, and Kaggle datasets.

The notebooks are designed to run directly inside Kaggle. They do not download datasets manually and do not depend on local dataset files. All input data is read from Kaggle's mounted dataset directory:

```text
/kaggle/input
```

Notebook outputs and preview files are saved to:

```text
/kaggle/working
```

## Experiments

| DS No. | Experiment | Notebook | Dataset | CUDA Operation |
| --- | --- | --- | --- | --- |
| DS2 | Matrix Addition | `DS2_matrix_addition_cuda_numba_kaggle.ipynb` | Matrix Dataset | Matrix Addition |
| DS1 | Parallel Reduction | `02_parallel_reduction_min_cuda_numba_kaggle.ipynb` | 10 Million Random Number Dataset | Minimum Element |
| DS13 | Radix Sort | `03_radix_sort_cuda_numba_kaggle.ipynb` | 10 Million Random Number Dataset | Radix Sort |

There is also a numbered copy of the DS2 notebook:

```text
01_matrix_addition_cuda_numba_kaggle.ipynb
```

## 1. DS2 - Matrix Addition

### Dataset

- Dataset Name: Matrix Dataset
- Source: Kaggle
- Link: <https://www.kaggle.com/datasets/julianezrasamuel/matrix-dataset>
- File Type: `.pkl`
- Matrix Size: `1024 x 1024`
- Data Type: `float32` / `float64`
- Number of Matrices: multiple matrices available

### Task

The notebook loads two matrices, `A` and `B`, from the Kaggle Matrix Dataset and computes:

```text
C = A + B
```

### CUDA Strategy

One CUDA thread computes one output matrix element.

```text
C[row, col] = A[row, col] + B[row, col]
```

The notebook also verifies the CUDA result using a CPU calculation.

## 2. DS1 - Parallel Reduction for Minimum Element

### Dataset

- Dataset Name: 10 Million Random Number Dataset for ML
- Source: Kaggle
- Link: <https://www.kaggle.com/datasets/mehedihasand1497/10-million-random-number-dataset-for-ml>
- Records: `10,000,000`
- Features: `50`
- Data Type: `float32`

### Task

The notebook finds the minimum element from a large floating-point vector.

### CUDA Strategy

The implementation uses tree-based parallel reduction with shared memory.

Each CUDA block computes a partial minimum, and repeated reduction passes are used until one global minimum remains.

By default, the notebook reduces one feature column with `10,000,000` values. To process all 50 feature columns, change this setting inside the notebook:

```python
REDUCE_ALL_50_FEATURES = True
```

## 3. DS13 - Radix Sort using CUDA

### Dataset

- Dataset Name: 10 Million Random Number Dataset for ML
- Source: Kaggle
- Link: <https://www.kaggle.com/datasets/mehedihasand1497/10-million-random-number-dataset-for-ml>
- Records Used: up to `10,000,000`
- Data Type for Sorting: `uint32`
- Range: `0` to `4,294,967,295`

### Task

The notebook sorts an integer array using CUDA Radix Sort.

The original dataset contains floating-point values between 0 and 1. These values are converted into unsigned 32-bit integer keys:

```python
key = round(value * (2**32 - 1))
```

### CUDA Strategy

The implementation uses Least Significant Digit Radix Sort.

Each bit pass performs:

1. Count zero-bit keys per CUDA block.
2. Compute prefix offsets.
3. Scatter keys into the output array.
4. Swap buffers for the next pass.

## Kaggle Run Instructions

1. Open the notebook in Kaggle.
2. Click **Add Data**.
3. Add the required dataset:
   - DS2: Matrix Dataset
   - DS1 and DS13: 10 Million Random Number Dataset for ML
4. Enable GPU from Notebook Settings.
5. Run **Restart Session and Run All**.

## Requirements

These notebooks are intended for the Kaggle notebook environment.

Main libraries used:

- Python 3
- NumPy
- pandas
- Numba
- CUDA GPU

Kaggle normally provides these libraries by default.

## Output Files

The notebooks print execution details such as:

- detected dataset file
- CUDA device name
- number of processed elements
- correctness check
- elapsed time

Some notebooks save preview files in `/kaggle/working`, such as:

```text
ds2_matrix_addition_cuda_sample.csv
ds13_radix_sort_sorted_sample.csv
```

The radix sort notebook can also save the full sorted array:

```text
ds13_radix_sort_sorted_uint32.npy
```

## Notes

- GPU must be enabled before running the notebooks.
- Datasets must be added using Kaggle's **Add Data** option.
- The notebooks do not download datasets from the internet.
- If Kaggle keeps old variables or functions in memory, use **Restart Session and Run All**.

## Short Report Table

| Experiment | Dataset | Size | Data Type | CUDA Operation |
| --- | --- | --- | --- | --- |
| DS2 Matrix Addition | Matrix Dataset | `1024 x 1024` | Float | Matrix Addition |
| DS1 Parallel Reduction | 10 Million Random Number Dataset | `10M values` | Float | Minimum Element |
| DS13 Radix Sort | 10 Million Random Number Dataset | `10M values` | Integer | Radix Sort |
