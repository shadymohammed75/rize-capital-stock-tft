#Gauß-Seidel method for the Lyapunov equation AX + XA = C
import numpy as np
def solve_lyapunov():
    # --- 1. Setup Parameters ---
    n = 5  # Dimension n=5

    # Create Matrix A where a_ij = 2^{-|i-j|}
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 2.0 ** (-abs(i - j))

    # Create Matrix C = Identity (id)
    C = np.eye(n)

    # Initialize X with zeros (X^(0) = 0)
    X = np.zeros((n, n))

    # --- 2. Iteration Loop ---
    # We iterate until the sum of corrections d_k <= 10^-8
    k = 0
    tol = 1e-8

    print(f"Starting iteration with n={n}, tolerance={tol}...")

    while True:
        d_k = 0.0  # Variable to store sum of corrections for this step

        # Loop over all indices (i, j) essentially applying Gauß-Seidel
        # We update X[i, j] immediately, so subsequent calculations use the new value (t=k+1)
        for i in range(n):
            for j in range(n):
                # Calculate the matrix product terms (AX)_ij and (XA)_ij
                # This matches the Gauß-Seidel requirement.

                # (AX)_ij = sum_{l} a_il * x_lj
                AX_ij = np.dot(A[i, :], X[:, j])

                # (XA)_ij = sum_{l} x_il * a_lj
                XA_ij = np.dot(X[i, :], A[:, j])

                residual = C[i, j] - AX_ij - XA_ij

                # Calculate the correction term
                denom = A[i, i] + A[j, j]
                correction = residual / denom

                # Update X in place
                X[i, j] += correction

                # Add absolute correction to d_k [cite: 39]
                d_k += abs(correction)

        # Increment iteration counter
        k += 1

        # Check stopping criterion [cite: 39]
        if d_k <= tol:
            break

    return k, X
# --- 3. Execution and Output ---
iterations, final_X = solve_lyapunov()

print("-" * 30)
print(f"Converged after {iterations} iterations.")
print("-" * 30)
print("Final Matrix X:")
np.set_printoptions(precision=4, linewidth=100)  # Format for readability
print(final_X)

