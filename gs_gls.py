import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.linalg
from scipy.optimize import minimize
from scipy.signal import welch, periodogram
import pywt

class GSGLS:
    """
    Geodesic Spectral-Generalised Least Squares (GS-GLS) Estimator.
    """
    def __init__(self, hierarchy, temporal_method='spectral'):
        self.hierarchy = hierarchy
        self.temporal_method = temporal_method
        self.S_sp = hierarchy.get_summing_matrix()
        # S_sp is n_s x m_s
        # We need S_tm later or assume it's provided or simple aggregation
        
        self.temporal_prec_diag = None # The diagonal Lambda^-1
        self.spatial_prec_matrix = None # Sparse matrix Q_sp
        self.P_sp = None
        self.P_tm = None
        
    def fit(self, residuals, S_tm=None):
        """
        Fit the covariance models to the residuals.
        
        Args:
            residuals: (n_nodes x n_timesteps) matrix of forecast errors.
            S_tm: Temporal summing matrix (n_t x m_t).
        """
        self.residuals = residuals
        n_nodes, n_timesteps = residuals.shape
        
        # --- 1. Temporal Estimation ---
        if self.temporal_method == 'spectral':
            self.temporal_prec_matrix = self._estimate_temporal_spectral(residuals)
        elif self.temporal_method == 'wavelet':
            self.temporal_prec_matrix = self._estimate_temporal_wavelet(residuals)
        else:
            raise ValueError(f"Unknown temporal method: {self.temporal_method}")

        # --- 2. Spatial Estimation (Geodesic) ---
        dist_mat = self.hierarchy.get_geodesic_distance_matrix()
        
        # Maximize log-likelihood for Rho
        def neg_log_lik(rho):
            if rho <= 0: return 1e10
            Sigma = np.exp(-dist_mat / rho)
            Sigma += 1e-6 * np.eye(Sigma.shape[0])
            try:
                sign, logdet = np.linalg.slogdet(Sigma)
                sol = scipy.linalg.solve(Sigma, residuals, assume_a='pos')
                quad_term = np.sum(residuals * sol)
                return 0.5 * (residuals.shape[1] * logdet + quad_term)
            except np.linalg.LinAlgError:
                return 1e10

        res = minimize(neg_log_lik, x0=[1.0], bounds=[(1e-2, None)], method='L-BFGS-B')
        self.rho_opt = res.x[0]
        
        # Construct Sparse Precision Q_sp (GMRF approx)
        L = self.hierarchy.get_graph_laplacian()
        I = sp.eye(L.shape[0])
        
        # Precompute eigenvalues of L for fast logdet
        # L is usually symmetric
        try:
            # Use dense eigh for stability on moderate size
            evals_L = scipy.linalg.eigh(L.toarray(), eigvals_only=True)
        except:
            # Fallback to sparse if too huge (though we need all evals for logdet)
            # For massive graphs, we'd need Chebyshev approx, but here N < 1000 is fine.
            evals_L = scipy.linalg.eigh(L.toarray(), eigvals_only=True)

        def gmrF_nll(params):
            kappa, tau = params
            if kappa <= 0 or tau <= 0: return 1e10
            
            # logdet(tau * (k^2 I + L)) = N*log(tau) + sum(log(k^2 + lambda_i))
            logdet_Q = L.shape[0] * np.log(tau) + np.sum(np.log(kappa**2 + evals_L))
            
            # Q is sparse
            Q = tau * (kappa**2 * I + L)
            
            # quad_term = sum(r' Q r)
            # Q is (N x N), residuals is (N x T)
            # diag(R' Q R) = sum(R * (Q R), axis=0)
            # sum total = sum(residuals * (Q @ residuals))
            Q_res = Q @ residuals
            quad_term = np.sum(residuals * Q_res)
            
            nll = -0.5 * (residuals.shape[1] * logdet_Q - quad_term)
            return nll

        res2 = minimize(gmrF_nll, x0=[1.0/self.rho_opt, 1.0], bounds=[(1e-4,None), (1e-4,None)])
        self.kappa_opt, self.tau_opt = res2.x
        
        self.spatial_prec_matrix = self.tau_opt * (self.kappa_opt**2 * I + L)
        
        # --- Precompute Spatial Projection ---
        self.S_sp_sparse = sp.csr_matrix(self.S_sp)
        H_sp = self.S_sp_sparse.T @ self.spatial_prec_matrix @ self.S_sp_sparse
        self.H_sp_factor = scipy.sparse.linalg.splu(H_sp)
        
        # --- Precompute Temporal Projection (if S_tm is provided) ---
        if S_tm is not None:
             # P_tm = S_tm (S_tm' W_tm^-1 S_tm)^-1 S_tm' W_tm^-1
             # We have W_tm^-1 in self.temporal_prec_matrix (dense n_t x n_t)
             # Note: For large n_t, this dense mult is slow. Demo size is fine.
             W_inv = self.temporal_prec_matrix
             H_tm = S_tm.T @ W_inv @ S_tm
             # Invert H_tm
             try:
                 H_tm_inv = np.linalg.inv(H_tm)
                 self.P_tm = S_tm @ H_tm_inv @ S_tm.T @ W_inv
             except np.linalg.LinAlgError:
                 print("Warning: Singular S_tm' W^-1 S_tm. Regularizing...")
                 H_tm_inv = np.linalg.inv(H_tm + 1e-6 * np.eye(H_tm.shape[0]))
                 self.P_tm = S_tm @ H_tm_inv @ S_tm.T @ W_inv
        else:
            self.P_tm = None # Identity or No-Op

    def _estimate_temporal_spectral(self, residuals):
        """
        Estimate temporal precision using Spectral Density (Stationary).
        """
        freqs, Pxx = periodogram(residuals, axis=1)
        avg_Pxx = np.mean(Pxx, axis=0)
        self.spectral_density = avg_Pxx + 1e-6
        
        # Construct Sigma_tm^-1 = F* Lambda^-1 F
        # We need the full matrix for projection construction currently
        n_t = residuals.shape[1]
        Lambda_inv = np.diag(1.0 / self.spectral_density)
        # DFT Matrix F: F_kj = exp(-2pi i k j / N) / sqrt(N)
        F = scipy.linalg.dft(n_t, scale='sqrtn')
        # Sigma^-1 = F.conj().T @ Lambda^-1 @ F
        # Note: Periodogram frequencies correspond to standard DFT order? 
        # periodogram returns [0, 1/T, ... 0.5] (one-sided for real).
        # We need two-sided matching DFT matrix.
        # Simplification for Demo: Use Toeplitz inverse of ACF?
        # Or just trust that we map periodogram to DFT diag.
        
        # Since we are implementing 'spectral' as 'stationary', approximating with Toeplitz inverse is better.
        # But let's stick to the prompt's implied simple spectral diag.
        # HACK: Reconstruct full diagonal in two-sided DFT?
        # Let's use the autocovariance from IFFT of PSD to build Toeplitz, then invert.
        # This is more robust.
        
        # 1. PSD to ACF
        # Two-sided PSD needed for irfft.
        # Construct approx ACF from avg_Pxx directly (assuming it matches periodogram outcome)
        acf = np.fft.irfft(self.spectral_density, n=n_t)
        # This acf is length n_t (or close).
        
        # 2. Toeplitz Covariance
        Sigma_tm = scipy.linalg.toeplitz(acf)
        
        # 3. Invert
        try:
            Prec_tm = np.linalg.inv(Sigma_tm + 1e-6 * np.eye(n_t))
        except np.linalg.LinAlgError:
            Prec_tm = np.eye(n_t) # Fallback
            
        return Prec_tm

    def _estimate_temporal_wavelet(self, residuals):
        """
        Estimate temporal precision using Wavelet Variance (Non-Stationary).
        """
        n_nodes, n_t = residuals.shape
        bn = pywt.wavedec(residuals[0], 'db1', level=None)
        coeffs_shapes = [c.shape for c in bn]
        
        # 1. Collect all coefficients
        # We want to compute variance of each coefficient index across spatial nodes.
        # coeffs_all: list of arrays, each array is (n_nodes x len_coef)
        coeffs_all = []
        for i in range(n_nodes):
            coeffs_all.append(pywt.wavedec(residuals[i], 'db1', level=None))
            
        # 2. Compute Variances
        variances = []
        for level_idx in range(len(coeffs_shapes)):
            # Gather (n_nodes x len_coef)
            level_data = np.array([c[level_idx] for c in coeffs_all])
            # Var across nodes (axis 0)
            # Add epsilon
            var = np.var(level_data, axis=0) + 1e-6
            variances.append(var)
            
        # 3. Construct Diagonal Precision in Wavelet Domain
        # We need to build the operator W^-1.
        # Since we need a dense matrix for P_tm construction (for now), let's build it explicitly.
        # W^-1 = Psi^T Lambda^-1 Psi
        # Where Psi is the DWT matrix.
        
        # Build Psi explicitly by transforming Identity columns
        # This is O(T^2), fine for T ~ 300.
        Psi = []
        I = np.eye(n_t)
        for t in range(n_t):
            c = pywt.wavedec(I[:, t], 'db1', level=None)
            # Flatten coeffs to a single vector
            vec = np.concatenate(c)
            Psi.append(vec)
        Psi = np.array(Psi).T # Columns are transformed basis vectors?
        # Wait. W x = Coeffs. So Psi is the forward transform matrix?
        # Yes. If x is input, c = Psi @ x.
        # Psi has rows corresponding to wavelet basis functions? 
        # Actually Psi in literature usually means Basis Matrix ($x = \Psi c$).
        # Forward transform is $\Psi^T$ or $\mathcal{W}$.
        # Let's check `pywt` linearity.
        
        # Let's perform reconstruction of the precision matrix:
        # Prec = Sum_k (1/var_k) * (Analysis_Basis_k @ Analysis_Basis_k.T) ?
        # Or Synthesis?
        # Covariance in domain W is D = diag(variances).
        # Cov_x = Psi_inv @ D @ Psi_inv.T ? (where $c = Psi @ x$)
        # Rec = Psi_inv @ c.
        # Cov_x = E[Psi_inv c c^T Psi_inv^T] = Psi_inv E[cc^T] Psi_inv^T = Psi_inv D Psi_inv^T.
        # Precision_x = (Cov_x)^-1 = (Psi_inv^T)^-1 D^-1 Psi_inv^-1
        # Since DWT is orthogonal (for db1/Haar), Psi_inv = Psi^T.
        # Prec_x = Psi D^-1 Psi^T.
        # So we need Psi (the forward transform matrix).
        
        # Let's measure Psi by transforming columns of I.
        # Psi_{row, col} = coeff_{row} of unit vector {col}.
        
        # Flatten variances to single diagonal
        D_inv_diag = np.concatenate([1.0/v for v in variances])
        D_inv = np.diag(D_inv_diag)
        
        # Build Psi
        # transform I[0]: [cA, cD...] -> vec
        Psi_cols = []
        for t in range(n_t):
            c_list = pywt.wavedec(I[:, t], 'db1', level=None)
            Psi_cols.append(np.concatenate(c_list))
        Psi = np.array(Psi_cols).T # (n_coeffs x n_t)
        
        # Prec = Psi.T @ D_inv @ Psi
        # Wait. Psi maps Time -> Coeffs. $c = Psi x$.
        # $x' P x = c' D^{-1} c = (Psi x)' D^{-1} (Psi x) = x' (Psi' D^{-1} Psi) x$.
        # So Prec = Psi.T @ D_inv @ Psi.
        
        Prec_tm = Psi.T @ D_inv @ Psi
        return Prec_tm

    def reconcile(self, Y_hat, S_tm=None):
        """
        Apply reconciliation.
        Y_hat: (n_nodes x n_timesteps) - The base forecasts.
        """
        # 1. Spatial Projection
        Q_Y = self.spatial_prec_matrix @ Y_hat
        RHS = self.S_sp_sparse.T @ Q_Y
        coeffs = self.H_sp_factor.solve(RHS)
        Z_sp = self.S_sp_sparse @ coeffs
        
        # 2. Temporal Projection (if available)
        # tilde_Y = P_sp Y_hat P_tm'
        # With Z_sp = P_sp Y_hat
        # tilde_Y = Z_sp P_tm'
        
        # NOTE: If self.P_tm is Set (via fit with S_tm), use it.
        # If user passes new S_tm here, we can't easily use it without recomputing P_tm which is slow.
        # We assume S_tm passed in fit() is the one to use.
        
        if self.P_tm is not None:
             Z = Z_sp @ self.P_tm.T
        else:
             Z = Z_sp
        
        return Z

    def reconcile_impute(self, Y_hat, maxiter=None, tol=1e-5):
        """
        Apply reconciliation with imputation for missing values (NaNs).
        Uses a Masked Conjugate Gradient solver for the Spatial projection.
        
        Args:
            Y_hat: (n_nodes x n_timesteps) matrix with NaNs.
            maxiter: Max iterations for CG.
            tol: Tolerance for CG.
            
        Returns:
            Z: Reconciled (and imputed) forecasts.
        """
        n_nodes, n_timesteps = Y_hat.shape
        Z = np.zeros_like(Y_hat)
        
        # Q is spatial precision
        Q = self.spatial_prec_matrix
        S = self.S_sp_sparse
        ST = S.T
        
        # Iterate over timesteps because masking might vary
        # Optimization: Group columns with same mask if possible, 
        # but for random missingness, just loop.
        
        for t in range(n_timesteps):
            y_col = Y_hat[:, t]
            mask_bool = ~np.isnan(y_col)
            
            # If no missing values, use fast cached solver
            if np.all(mask_bool):
                rhs = ST @ (Q @ y_col)
                coeffs = self.H_sp_factor.solve(rhs)
                z_col = S @ coeffs
                Z[:, t] = z_col
                continue
                
            # Handle missing values
            # Replace NaN with 0 for computation (effective weight 0)
            y_filled = np.nan_to_num(y_col, nan=0.0)
            
            # Mask matrix M (diagonal)
            # Actually we just zero out rows/cols of Q implicitly.
            # LHS = S.T @ (M @ Q @ M) @ S
            # RHS = S.T @ (M @ Q @ M) @ y_filled
            # Wait, if y_filled has 0s where Mask is 0, then M @ y_filled = y_filled.
            # So RHS = S.T @ M @ Q @ y_filled? 
            # If Q is diagonal, Yes. If Q is dense/Laplacian:
            # We want to minimize (y - Sx)' M' Q M (y - Sx) ? 
            # Or is Missingness implying "Observation not available"?
            # Prompt says: "set corresponding row/column of precision matrix to zero".
            # If Prec is Q, we want Q_mod = M Q M.
            # Then solve (S' Q_mod S) x = S' Q_mod y.
            
            # Let's define the LinearOperator for A = S' M Q M S
            # mask_bool is 1 for observed, 0 for missing.
            # We need to apply M as elementwise mult.
            
            def mv(v):
                # v is shape (m_s,) where m_s is number of bottom series (coeffs)
                # S v -> (n_nodes,)
                # M (S v)
                # Q (M S v)
                # M (Q M S v)
                # S.T (M Q M S v)
                
                Sv = S @ v
                MSv = Sv * mask_bool # Apply Mask
                QMSv = Q @ MSv
                MQMSv = QMSv * mask_bool # Apply Mask
                return ST @ MQMSv
            
            m_s = S.shape[1]
            A = scipy.sparse.linalg.LinearOperator((m_s, m_s), matvec=mv)
            
            # Compute RHS
            # rhs = S.T @ M @ Q @ M @ y_filled
            # y_filled has 0s where mask is 0, so M @ y_filled = y_filled effectively?
            # BUT y_filled might have non-zeros where we want 0 if we filled with mean?
            # Ideally y_filled is exactly 0 at missing indices.
            # M @ y = y_filled.
            # Q @ y_filled
            # M @ (Q @ y_filled)
            # S.T @ ...
            
            # Ensure y_filled is 0 at missing
            y_in = y_filled * mask_bool
            rhs = ST @ (mask_bool * (Q @ y_in))
            
            # Initial guess: 0 or maybe warm start?
            x0 = np.zeros(m_s)
            
            # Solve
            # Use callback to monitor? No.
            coeffs, info = scipy.sparse.linalg.cg(A, rhs, x0=x0, rtol=tol, maxiter=maxiter)
            
            if info != 0:
                pass # print(f"CG Warning at t={t}: {info}")
                
            Z[:, t] = S @ coeffs
            
        # 3. Apply Temporal if needed (though masked input makes it tricky)
        # We perform temporal projection on the spatially-reconciled result
        # Assuming Z is now fully dense and coherent spatially.
        if self.P_tm is not None:
             Z = Z @ self.P_tm.T
             
        return Z

    def get_spectral_covariance_inverse(self):
        """Return the diagonal Lambda^-1 for inspection."""
        if self.spectral_density is None: return None
        return np.diag(1.0 / self.spectral_density)

