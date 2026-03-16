import numpy as np
from sklearn.base import check_array
from sklearn.utils.metaestimators import available_if
import scipy.sparse as sp

def _parent_has_method(method_name):
    """Check if the parent class has the method."""
    def check(self):
        attr = getattr(super(AutoBatchMixin, self), method_name, None)
        return attr is not None
    return check

class AutoBatchMixin:
    """
    Mixin that adds batch processing to predict, predict_proba, 
    and other common sklearn inference methods.
    """

    predict_batch_size=1

    def _batch_apply(self, method_name, X, **kwargs):
        """Internal helper to slice X and call the base method."""
        # Fallback to standard behavior if batch_size isn't set
        if self.predict_batch_size is None:
            return getattr(super(), method_name)(X, **kwargs)

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], accept_large_sparse=True)
    
        # If it's still not subscriptable (like COO), force CSR
        if sp.issparse(X) and not hasattr(X, '__getitem__'):
            X = X.tocsr()

        n_samples = X.shape[0]
        base_method = getattr(super(), method_name)
        
        # Get the first chunk to determine output shape and dtype
        first_chunk = base_method(X[0:1], **kwargs)

        if isinstance(first_chunk, list):
            # Pre-allocate a list of empty arrays matching each label's output shape
            results = [
                np.empty((n_samples, *out.shape[1:]), dtype=out.dtype)
                for out in first_chunk
            ]
            
            for i in range(0, n_samples, self.predict_batch_size):
                stop = min(i + self.predict_batch_size, n_samples)
                chunk = base_method(X[i:stop], **kwargs)
                # Assign each label's chunk to the correct position in the result list
                for res_arr, chunk_arr in zip(results, chunk):
                    res_arr[i:stop] = chunk_arr
            return results
        
        # Pre-allocate the result array
        # Shape handles both 1D (labels) and 2D (probabilities) outputs
        results = np.empty((n_samples, *first_chunk.shape[1:]), dtype=first_chunk.dtype)
        
        for i in range(0, n_samples, self.predict_batch_size):
            stop = min(i + self.predict_batch_size, n_samples)
            results[i:stop] = base_method(X[i:stop], **kwargs)
            
        return results

    @available_if(_parent_has_method("predict"))
    def predict(self, X, **params):
        return self._batch_apply("predict", X,  **params)

    @available_if(_parent_has_method("predict_proba"))
    def predict_proba(self, X, **params):
        return self._batch_apply("predict_proba", X, **params)
    
    @available_if(_parent_has_method("decision_function"))
    def decision_function(self, X,  **params):
        return self._batch_apply("decision_function", X, **params)