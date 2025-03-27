import torch


def svd_flip(u, v=None, u_based_decision=True):
    """
    Adjusts the signs of the singular vectors from the SVD decomposition for
    deterministic output.
    This method ensures that the output remains consistent across different
    runs.
    Args:
        u (torch.Tensor): Left singular vectors tensor.
        v (torch.Tensor): Right singular vectors tensor.
        u_based_decision (bool, optional): If True, uses the left singular
            vectors to determine the sign flipping. Defaults to True.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular
            vectors tensors.
    """
    if u_based_decision:
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
    else:
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
    u *= signs
    if v is not None:
        v *= signs[:, None]
    return u, v


# class SmoothedEig(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, K):
#         eigenvalues, eigenvectors = torch.linalg.eigh(K)
#         ctx.save_for_backward(eigenvalues, eigenvectors)
#         return  eigenvalues,eigenvectors
#     @staticmethod
#     def backward(ctx, grad_eigval, grad_eigvec):
#         eigenvalues, eigenvectors = ctx.saved_tensors
#         grad_eigval = torch.clamp(grad_eigval, -1.0, 1.0)  # Clip eigenvalue gradients
#         return grad_eigvec @ torch.diag(grad_eigval) @ eigenvectors.T
class KernelCenterer():
    def fit(self, K):
        n_samples = K.shape[0]
        self.K_fit_rows_ = torch.sum(K, axis=0) / n_samples
        self.K_fit_all_ = torch.sum(self.K_fit_rows_) / n_samples
        return self

    def transform(self, K, copy=True):
        """Center kernel matrix.
        Parameters
        ----------
        K : ndarray of shape (n_samples1, n_samples2)
            Kernel matrix.
        copy : bool, default=True
            Set to False to perform inplace computation.
        Returns
        -------
        K_new : ndarray of shape (n_samples1, n_samples2)
            Returns the instance itself.
        """
        K_pred_cols = (torch.sum(K, axis=1) / self.K_fit_rows_.shape[0])[:, None]
        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_
        return K


class KernelPCA():
    def __init__(self, n_components=None, eigval_choice='MDS'):
        self.n_components = n_components
        self.eigval_choice = eigval_choice

    def choose_position(self, mode='MDS'):
        if mode == 'MDS':
            sorted_indices = torch.argsort(torch.abs(self.eigenvalues_), descending=True)
            eigenpos = sorted_indices[:self.n_components]
            c_2l = 0  # intednent for compatibility
        if mode == 'MDS_plus':
            # Sort eigenvalues in ascending order
            sorted_indices = torch.argsort(self.eigenvalues_)
            sorted_eigenvalues = self.eigenvalues_[sorted_indices]
            # Initialize tracking variables
            c_1 = torch.sum(sorted_eigenvalues.square())
            c_2l = torch.sum(sorted_eigenvalues)
            pos_ptr = len(sorted_eigenvalues) - 1  # Start from largest eigenvalue
            neg_ptr = 0  # Start from smallest (most negative) eigenvalue
            # Pre-allocate tensors for selected indices
            selected_pos = torch.zeros(self.n_components, dtype=torch.int)
            selected_neg = torch.zeros(self.n_components, dtype=torch.int)
            pos_count = neg_count = 0
            # Main selection loop
            for _ in range(self.n_components):
                if c_2l < 0:
                    # Select negative eigenvalue
                    idx = sorted_indices[neg_ptr]
                    selected_neg[neg_count] = idx
                    val = sorted_eigenvalues[neg_ptr]
                    neg_count += 1
                    neg_ptr += 1
                else:
                    # Select positive eigenvalue
                    idx = sorted_indices[pos_ptr]
                    selected_pos[pos_count] = idx
                    val = sorted_eigenvalues[pos_ptr]
                    pos_count += 1
                    pos_ptr -= 1
                # Update tracking values
                c_1 -= val.square()
                c_2l -= val
            # Trim unused pre-allocated space
            selected_pos = selected_pos[:pos_count]
            selected_neg = selected_neg[:neg_count]
            eigenpos = torch.hstack((selected_pos, selected_neg))
            c_2l = 0  # intednent for compatibility
        if mode == 'MDS_plus_bounds':
            # Sort eigenvalues in ascending order
            sorted_indices = torch.argsort(self.eigenvalues_)
            sorted_eigenvalues = self.eigenvalues_[sorted_indices]
            # Initialize tracking variables
            c_1 = torch.sum(sorted_eigenvalues.square())
            c_2l = torch.sum(sorted_eigenvalues)
            pos_ptr = len(sorted_eigenvalues) - 1  # Start from largest eigenvalue
            neg_ptr = 0  # Start from smallest (most negative) eigenvalue
            # Pre-allocate tensors for selected indices
            selected_pos = torch.zeros(self.n_components, dtype=torch.int)
            selected_neg = torch.zeros(self.n_components, dtype=torch.int)
            pos_count = neg_count = 0
            # Main selection loop
            for _ in range(self.n_components):
                # Calculate hypothetical updates
                pos_val = sorted_eigenvalues[pos_ptr]
                neg_val = sorted_eigenvalues[neg_ptr]

                pos_c_1 = c_1 - pos_val.square()
                pos_c_2l = c_2l - pos_val
                neg_c_1 = c_1 - neg_val.square()
                neg_c_2l = c_2l - neg_val

                # Compare using the bonus formula (CORRECTED CONDITION)
                pos_score = pos_c_1 + pos_c_2l.square() / (self.n_components + 1)
                neg_score = neg_c_1 + neg_c_2l.square() / (self.n_components + 1)
                if pos_score > neg_score:
                    # Select NEGATIVE eigenvalue when positive score is better
                    selected_neg[neg_count] = sorted_indices[neg_ptr]
                    val = neg_val
                    neg_count += 1
                    neg_ptr += 1
                    c_1 = neg_c_1
                    c_2l = neg_c_2l
                else:
                    # Select POSITIVE eigenvalue
                    selected_pos[pos_count] = sorted_indices[pos_ptr]
                    val = pos_val
                    pos_count += 1
                    pos_ptr -= 1
                    c_1 = pos_c_1
                    c_2l = pos_c_2l
            selected_pos = selected_pos[:pos_count]
            selected_neg = selected_neg[:neg_count]
            eigenpos = torch.hstack((selected_pos, selected_neg))
        return eigenpos, c_2l

    def _fit_transform_in_place(self, K):
        """Fit's using kernel K"""
        # center kernel in place
        K = self._centerer.fit(K).transform(K)
        self.eigenvalues_, self.eigenvectors_ = torch.linalg.eigh(K)
        # self.eigenvalues_ = self.eigenvalues_ / (torch.sum(torch.abs(self.eigenvalues_)) + 1e-4)
        # self.eigenvalues_, self.eigenvectors_ = SmoothedEig.apply(K)
        # flip eigenvectors' sign to enforce deterministic output
        # self.eigenvectors_, _ = svd_flip(u=self.eigenvectors_, v=None)
        # sorted_indices = torch.argsort(self.eigenvalues_, descending=True)
        # sorted_indices = torch.argsort(torch.abs(self.eigenvalues_), descending=True)
        self.eigenpos, c_2l = self.choose_position(mode=self.eigval_choice)
        self.eigenvalues_ = self.eigenvalues_[self.eigenpos] + c_2l / (self.n_components + 1)
        self.eigenvectors_ = self.eigenvectors_[:, self.eigenpos]
        # print('eigenvalues chosen {}'.format(self.eigenvalues_.data))

    def fit(self, K):
        self._centerer = KernelCenterer()
        self._fit_transform_in_place(K)
        # self.X_fit_ = K
        return self

    def fit_transform(self, K):
        self.fit(K)
        # no need to use the kernel to transform X, use shortcut expression
        # X_transformed = self.eigenvectors_ * torch.sqrt(self.eigenvalues_)
        X_transformed = self.eigenvectors_ * torch.sqrt(torch.abs(self.eigenvalues_))
        return X_transformed

    def transform(self, K):
        """Transform X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Returns the instance itself.
        """
        if self.eigenvalues_ is None or self.eigenvectors_ is None:
            raise ValueError("The model must be fitted before calling transform.")
        # Compute centered gram matrix between X and training data X_fit_
        K = self._centerer.transform(K)
        # scale eigenvectors (properly account for null-space for dot product)
        non_zeros = torch.nonzero(self.eigenvalues_).reshape(-1)
        scaled_alphas = torch.zeros_like(self.eigenvectors_)
        # scaled_alphas[:, non_zeros] = self.eigenvectors_[:, non_zeros] / torch.sqrt(self.eigenvalues_[non_zeros])
        scaled_alphas[:, non_zeros] = self.eigenvectors_[:, non_zeros] / torch.sqrt(
            torch.abs(self.eigenvalues_[non_zeros]))
        # Project with a scalar product between K and the scaled eigenvectors
        return K @ scaled_alphas