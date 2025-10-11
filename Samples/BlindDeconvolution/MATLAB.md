# MATLAB's  **`deconvblind`**

A complex function from MATLAB’s Image Processing Toolbox that implements **blind deconvolution** of images.

---

## **Purpose**

`deconvblind` performs **blind image deconvolution** — it simultaneously estimates:

1. The **original (sharp) image** `J`, and
2. The **point-spread function (PSF)** `P`,

from a **blurred image** `I` when the PSF is **unknown or uncertain**.

This is used in image restoration problems (e.g., astronomy, microscopy, motion blur).

---

## **Algorithm: Blind Lucy–Richardson (Maximum Likelihood) Deconvolution**

The method is based on the **Lucy–Richardson (LR)** algorithm extended for blind estimation (both image and PSF updated iteratively).

Key reference:
Biggs & Andrews, *Applied Optics*, 1997 — “Acceleration of iterative image restoration algorithms.”

---

##  **Major Steps in the Code**

1. Parse and validate inputs.
2. Initialize parameters and FFT components.
3. Run iterative Lucy–Richardson-style updates:
   * Predict image and PSF
   * Apply positivity and normalization constraints
   * Update both image and PSF
   * Optionally apply user-defined PSF constraints
4. Repeat for a fixed number of iterations.
5. Convert results to original data class and output.

## **Mathematical Summary**

The iterative updates implement **maximum likelihood estimation** assuming Poisson noise:
[
J_{k+1} = J_k \cdot \frac{(I / (J_k * P_k)) * P_k^\text{flip}}{\text{scale}}
]
and similarly update `P_k`.

The damping and weighting terms control:

* **DAMPAR:** noise suppression (prevents overfitting to noise).
* **WEIGHT:** ignores unreliable pixels.
* **READOUT:** accounts for camera readout noise.

---

## **Key Features and Design Choices**

| Concept                      | Implementation Detail                        |
| ---------------------------- | -------------------------------------------- |
| **Positivity Constraint**    | Uses `max(..., 0)`                           |
| **PSF Normalization**        | Sum of PSF = 1 enforced                      |
| **Momentum / Acceleration**  | Uses difference of previous two iterations   |
| **FFT-based Convolution**    | Uses `psf2otf`, `ifftn` for fast computation |
| **Damping / Regularization** | Via `DAMPAR` and `corelucy`                  |
| **Custom Constraints**       | User-supplied function `FUN`                 |
| **Resumable Iterations**     | Input/output via cell arrays                 |

---

## **Summary**

The `deconvblind` function:

* Implements **blind Lucy–Richardson deconvolution**.
* Iteratively estimates both **image** and **PSF**.
* Handles **noise suppression**, **weighting**, and **constraints**.
* Uses **Fourier-domain convolution** for efficiency.
* Supports **resumable iterative operation**.

---
