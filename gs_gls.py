import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.optimize import minimize
from scipy.signal import periodogram
from scipy.fft import fft, ifft
import pywt
import time

class GSGLS:
    """
    Geodesic Spectral-Generalised Least Squares (GS-GLS) Estimator.
    
    Implements the GS-GLS framework for optimal reconciliation of high-dimensional 
    hierarchies, supporting both stationary (Spectral) and non-stationary (Wavelet)
    error processes.
    """
    def __init__(self, hierarchy, mode='stationary', wavelet_family='db1', lambda_reg=1e-6):
        """
        Args:
            hierarchy: Hierarchy object containing spatial structure.
            mode: 'stationary' (FFT-based) or 'non_stationary' (Wavelet-based).
            wavelet_family: Wavelet family to use if mode is 'non_stationary' (e.g., 'db4').
            lambda_reg: Regularization parameter for numerical stability.
        """
        self.hierarchy = hierarchy
        self.mode = mode
        self.wavelet_family = wavelet_family
        self.lambda_reg = lambda_reg
        
        self.S_sp = hierarchy.get_summing_matrix() # n_s x m_s
        self.S_tm = None # Will be set in fit or reconcile if provided
        
        # Precision Operators parameters
        self.spatial_prec_matrix = None # Q_sp
        self.temporal_prec_params = None # (Lambda^-1 or Delta^-1)
        
        # Dimensions
        self.n_s = self.S_sp.shape[0]
        self.m_s = self.S_sp.shape[1]
        
    def fit(self, residuals, S_tm=None):
        """
        Fit the covariance models to the residuals.
        
        Args:
            residuals: (n_s x n_t) matrix of historical forecast errors.
                       Note: Should ideally be the residuals of the incoherent forecasts.
            S_tm: (n_t x m_t) Temporal summing matrix. 
                  If None, assumes Identity (no temporal aggregation, just smoothing).
        """
        self.S_tm = S_tm
        n_s, n_t = residuals.shape
        
        # If S_tm is not provided, we assume strictly m coordinate system for time??
        # Usually MinT requires S to be defined. If S_tm is None, we assume n_t = m_t.
        if self.S_tm is None:
            self.S_tm = sp.eye(n_t)
            
        self.n_t = n_t
        self.m_t = self.S_tm.shape[1]
        
        # --- 1. Temporal Precision Estimation ---
        print(f"Estimating Temporal Precision ({self.mode})...")
        if self.mode == 'stationary':
            self._estimate_spectral_precision(residuals)
        elif self.mode == 'non_stationary':
            self._estimate_wavelet_precision(residuals)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        # --- 2. Spatial Precision Estimation (Geodesic) ---
        print("Estimating Spatial Precision (Geodesic)...")
        self._estimate_spatial_precision(residuals)
        
        print("Fit complete.")
        
    def _estimate_spectral_precision(self, residuals):
        """
        Estimates the spectral density and inverse eigenvalues for stationary errors.
        """
        # 1. Compute Periodogram for each series (row) and average
        # Using periodogram allows us to get the PSD at DFT frequencies
        freqs, Pxx = list(zip(*[periodogram(residuals[i, :], scaling='density', detrend='linear') 
                               for i in range(self.n_s)]))
        avg_Pxx = np.mean(np.array(Pxx), axis=0) # Shape: (n_fft // 2 + 1,)
        
        # We need the full DFT size weights. The periodogram returns one-sided for real signals.
        # We need to reconstruct the full symmetric PSD for the FFT operator.
        # Approximation: Project PSD onto the full FFT grid.
        
        # Simple approach: compute FFT of autocovariance or just map periodogram to FFT bins.
        # Frequencies: 0, 1/N, ... (N/2)/N, ... (N-1)/N
        
        # Let's effectively use the full FFT magnitude squared of the residuals
        # Mean across nodes
        specs = []
        for i in range(self.n_s):
            f_coeff = fft(residuals[i, :])
            specs.append(np.abs(f_coeff)**2)
            
        avg_spec = np.mean(np.array(specs), axis=0)
        avg_spec /= self.n_t # Normalize
        
        # Stabilize (Spectral Floor)
        self.temporal_prec_params = 1.0 / (avg_spec + self.lambda_reg)
        
    def _estimate_wavelet_precision(self, residuals):
        """
        Estimates the wavelet variance scales for non-stationary errors.
        """
        # Determine max level
        max_level = pywt.dwt_max_level(self.n_t, self.wavelet_family)
        
        # Decompose each series to get coefficients structure
        coeffs_example = pywt.wavedec(residuals[0, :], self.wavelet_family, level=max_level)
        
        # We need to compute the variance of coefficients at each scale (and location?)
        # For true non-stationarity, variance can vary with time (location).
        # We will estimate Variance vector matching the flattened coefficients.
        
        all_coeffs_sq = []
        for i in range(self.n_s):
            coeffs = pywt.wavedec(residuals[i, :], self.wavelet_family, level=max_level)
            # Flatten
            flat_c, slices = pywt.coeffs_to_array(coeffs)
            all_coeffs_sq.append(flat_c**2)
            
        avg_var = np.mean(np.array(all_coeffs_sq), axis=0)
        
        # Stabilize
        self.temporal_prec_params = 1.0 / (avg_var + self.lambda_reg)
        self.wavelet_slices = slices # Store to reconstruct
        self.wavelet_max_level = max_level

    def _estimate_spatial_precision(self, residuals):
        """
        Estimates the GMRF parameters on the graph.
        """
        # Heuristic/Simplified Estimation for Stability and Speed:
        # 1. Estimate effective correlation length from data or use default logic if too sparse.
        # For this implementation, we will perform a quick optimization of the GMRF Likelihood
        # using the sample covariance approximation.
        
        L = self.hierarchy.get_graph_laplacian()
        I = sp.eye(L.shape[0])
        n_samples = residuals.shape[1]
        
        # Empirical covariance term: Tr(S * Q) = sum(diag(R'QR)) = sum(r_t' Q r_t)
        # We can precompute r_t' L r_t and r_t' r_t
        
        # Precompute quadratic forms
        # L is sparse
        L_res = L @ residuals # (n_s x n_t)
        term_L = np.sum(residuals * L_res) # sum over all i, t
        term_I = np.sum(residuals * residuals)
        
        def gmrf_score(params):
            # Q = tau * (kappa^2 I + L) 
            # We fix alpha=1 for simplicity (Whittle-Matern)
            kappa, tau = params
            if kappa <= 1e-3 or tau <= 1e-6: return 1e15
            
            # Trace term
            # Tr(R R' Q) = tau * (kappa^2 * term_I + term_L)
            tr_term = tau * (kappa**2 * term_I + term_L)
            
            # LogDet term: N * log|Q| = N * (n_s * log(tau) + log|kappa^2 I + L|)
            # Approximating log|kappa^2 I + L| is expensive. 
            # We'll use the eigenvalues of L if small enough, or simple approximation.
            # L is n_s x n_s. If n_s < 5000, we can compute eigenvalues once.
            
            # For this demo/code, we'll assume we can compute eigenvalues of L once.
            if not hasattr(self, 'L_eigvals'):
                # Dense eigval (okay for < 5k nodes)
                try:
                    self.L_eigvals = scipy.linalg.eigvalsh(L.toarray())
                except:
                    # If too large, assume regular distribution?
                    self.L_eigvals = np.linspace(0, 4, self.n_s) # Dummy
            
            log_det_part = np.sum(np.log(tau * (kappa**2 + self.L_eigvals)))
            
            # NLL = 0.5 * ( tr_term - n_samples * log_det_part )
            # Wait, standard Gaussian likelihood:
            # -0.5 * ( - tr(S Q) + log|Q| ) -> min 0.5 * ( tr(S Q) - log|Q| )
            # Here multiple samples: T * log|Q|
            
            nll = 0.5 * (tr_term - n_samples * log_det_part)
            return nll

        # Optimize
        try:
            res = minimize(gmrf_score, x0=[0.5, 1.0], bounds=[(1e-2, 10.0), (1e-4, 10.0)], method='L-BFGS-B')
            self.kappa_opt, self.tau_opt = res.x
        except:
             # Fallback
            self.kappa_opt, self.tau_opt = 0.1, 1.0

        print(f"Spatial Params: kappa={self.kappa_opt:.4f}, tau={self.tau_opt:.4f}")
        
        # Construct Q_sp sparse matrix
        self.spatial_prec_matrix = self.tau_opt * (self.kappa_opt**2 * I + L)

    def _apply_temporal_precision(self, X):
        """
        Applies Sigma_tm^-1 to each row of X (n_s x n_t).
        """
        if self.mode == 'stationary':
            # FFT -> Mult -> IFFT
            # X is real, but FFT is complex.
            # Sigma^-1 is symmetric, diagonal in Frequency domain.
            # Result should be real.
            
            X_f = fft(X, axis=1)
            # Elementwise multiply by diagonal precision (broadcasting over rows)
            X_filtered_f = X_f * self.temporal_prec_params # params is 1D array of length n_t
            X_out = ifft(X_filtered_f, axis=1).real
            return X_out
            
        elif self.mode == 'non_stationary':
            # Optimized Vectorized Wavelet Filter
            # 1. Decompose all rows at once
            coeffs = pywt.wavedec(X, self.wavelet_family, level=self.wavelet_max_level, axis=1)
            
            # 2. Filter coefficients
            # self.temporal_prec_params is a Flat array corresponding to flattened coeffs of ONE series.
            # We need to reshape/split it to match the structure of 'coeffs' list.
            
            # Reconstruct the parameter structure from flat array using stored slices
            # We do this once or every time? Cheap enough to slice.
            
            coeffs_filtered = []
            
            # Note: pywt.coeffs_to_array gives slices for the flattened concatenation.
            # We assume the structure (lengths) is identical for all series (fixed n_t).
            
            # We iterate over the coeff levels (cA, cD_n, cD_n-1, ...) based on the list returned by wavedec.
            # We use the 'slices' computed during fit to extract the corresponding precision weights.
            
            # slices is a list of slice objects or tuples? pywt returns a list of slices in newer versions? 
            # Actually coeffs_to_array returns (array, slices_list).
            # slices_list[0] corresponds to coeffs[0], etc.
            
            for i, c_block in enumerate(coeffs):
                # Get corresponding weights
                # self.wavelet_slices[i] gives the slice for the flat array
                sl = self.wavelet_slices[i]
                weights = self.temporal_prec_params[sl]
                
                # c_block shape: (n_s, width)
                # weights shape: (width,)
                # Broadcast mult: c_block * weights [None, :] or just broadcast last dim?
                # numpy broadcasts trailing dimensions automatically.
                # (n_s, w) * (w,) -> Works.
                
                coeffs_filtered.append(c_block * weights)
                
            # 3. Reconstruct
            out = pywt.waverec(coeffs_filtered, self.wavelet_family, axis=1)
            
            # Ensure correct length (padding issues sometimes occur)
            if out.shape[1] > X.shape[1]:
                out = out[:, :X.shape[1]]
                
            return out
            
    def _matvec_operator(self, v_flat):
        """
        Computes A * v where A = S^T (Sigma^-1 (x) Q_sp) S.
        v_flat is vector of size m_s * m_t.
        """
        # Defines the Linear Operator for PCG
        
        # 0. Reshape v
        # v corresponds to bottom level forecasts.
        # Shape (m_s, m_t)
        v = v_flat.reshape((self.m_s, self.m_t))
        
        # 1. Up-Projection: U = S v
        # U = S_sp * v * S_tm^T
        # S_sp: (n_s x m_s), S_tm: (n_t x m_t)
        
        # S_sp is sparse: (n_s x m_s)
        # S_tm might be sparse or dense. usually small enough or sparse.
        
        # U = (S_sp @ v) @ S_tm.T
        U_temp = self.S_sp @ v # (n_s x m_t)
        U = U_temp @ self.S_tm.T # (n_s x n_t)
        
        # 2. Apply Precision W^-1
        # Z = Q_sp * U * (Sigma_tm^-1)^T
        # But Sigma_tm^-1 is symmetric.
        # So Z = Q_sp * (Apply Sigma^-1 to rows of U) ???
        # W^-1 = Sigma_tm^-1 (x) Q_sp.
        # vec(Z) = (Sigma^-1 (x) Q_sp) vec(U)
        #        = vec(Q_sp U (Sigma^-1)^T)
        
        # Step 2a: Temporal Filtering (Apply Sigma^-1 to columns? No, from right side)
        # U (Sigma^-1)^T means applying Sigma^-1 to the ROWS of U (since U * S_inv_T).
        # Yes. U is (n_s x n_t). We operate on time axis.
        
        U_temp_filt = self._apply_temporal_precision(U)
        
        # Step 2b: Spatial Filtering (Apply Q_sp to columns)
        # Q_sp @ U_temp_filt. Q_sp is (n_s x n_s).
        Z = self.spatial_prec_matrix @ U_temp_filt
        
        # 3. Down-Projection: w = S^T Z
        # w = S^T vec(Z) -> w_mat = S_sp^T Z S_tm
        
        w_temp = self.S_sp.T @ Z # (m_s x n_t)
        w_mat = w_temp @ self.S_tm # (m_s x m_t)
        
        return w_mat.ravel()

    def reconcile(self, y_hat):
        """
        Reconcile base forecasts.
        
        Args:
            y_hat: (n_s x n_t) Matrix of base forecasts.
        
        Returns:
            y_tilde: (n_s x n_t) Reconciled forecasts.
        """
        # MinT Solution: y~ = S (S' W^-1 S)^-1 S' W^-1 y^
        # Let b = S' W^-1 y^
        # Solve A x = b for x (where x is beta)
        # y~ = S x
        
        # 1. Compute b
        # Apply W^-1 to y_hat
        # Z_hat = Q_sp y_hat (Sigma^-1)^T
        y_hat_temp = self._apply_temporal_precision(y_hat)
        Z_hat = self.spatial_prec_matrix @ y_hat_temp
        
        # Project down: b = S^T Z_hat
        b_temp = self.S_sp.T @ Z_hat # (m_s x n_t)
        b_mat = b_temp @ self.S_tm # (m_s x m_t)
        b = b_mat.ravel()
        
        # 2. Setup PCG
        N_unknowns = self.m_s * self.m_t
        A_op = scipy.sparse.linalg.LinearOperator((N_unknowns, N_unknowns), matvec=self._matvec_operator)
        
        # 3. Solve
        # A x = b
        print(f"Starting PCG Solver (Size {N_unknowns})...")
        t0 = time.time()
        beta_flat, info = scipy.sparse.linalg.cg(A_op, b, rtol=1e-5, maxiter=1000)
        print(f"PCG solved in {time.time()-t0:.4f}s. Info: {info}")
        
        # 4. Project up: y~ = S beta
        beta = beta_flat.reshape((self.m_s, self.m_t))
        y_tilde_temp = self.S_sp @ beta
        y_tilde = y_tilde_temp @ self.S_tm.T
        
        return y_tilde
