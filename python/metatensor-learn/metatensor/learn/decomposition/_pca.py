import warnings
from typing import List, Optional, Tuple

from .._backend import Array, Labels, TensorBlock, TensorMap
from .._dispatch import (
    abs,
    argmax,
    copy,
    cumsum,
    int_array_like,
    sign,
    sum,
    svd,
)


class EquivariantPCA:
    """
    Scikit-learn-like Principal Component Analysis for TensorMaps
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        verbose: Optional[bool] = True,
        key_lambda_name: Optional[str] = "o3_lambda",
    ) -> None:
        self.n_components = n_components
        self.verbose = verbose
        self.key_lambda_name = key_lambda_name

    @staticmethod
    def _get_mean(values: Array, lambda_key: int) -> float:
        return 0.0
        # if l == 0:
        #     sums = np.sum(values.detach().numpy(), axis=1)
        #     signs = torch.from_numpy(((sums <= 0) - 0.5) * 2.0)
        #     values = signs[:, None] * values
        #     mean = torch.mean(values, dim=0)
        #     return mean
        # else:
        #     return 0.0

    def _svdsolve(self, X) -> Tuple[Array, Array]:
        U, S, Vt = svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = self._svd_flip(U, Vt)
        eigs = S**2 / (X.shape[0] - 1)
        return eigs, Vt.T

    @staticmethod
    def _svd_flip(u: Array, v: Array) -> Tuple[Array, Array]:
        """translated from sklearn implementation"""
        max_abs_cols = argmax(abs(u), axis=0)
        signs = sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, None]
        return u, v

    def _fit(self, values: Array, lambda_key: int) -> Tuple[Array, Array, Array]:
        nsamples, ncomps, nprops = values.shape
        values = values.reshape(nsamples * ncomps, nprops)

        mean = self._get_mean(values, lambda_key)
        eigs, components = self._svdsolve(values - mean)

        return mean, eigs, components

    def fit(self, tensor: TensorMap) -> None:
        values_: List[Array] = []
        explained_variance_: List[Array] = []
        explained_variance_ratio_: List[Array] = []
        components_: List[Array] = []
        retained_components_: List[Array] = []

        for key, block in tensor.items():
            lambda_key = key[self.key_lambda_name]

            values = copy(block.values)
            nsamples, ncomps, nprops = values.shape

            if nsamples <= 1:
                retained = int_array_like(range(0), values)
                eigs = None
                eigs_ratio = None
                components = None
            else:
                # Perform SVD
                mean, eigs, components = self._fit(values, lambda_key=lambda_key)
                eigs_ratio = eigs / sum(eigs)
                eigs_csum = cumsum(eigs_ratio, axis=0)
                if self.n_components is None:
                    max_comp = components.shape[1]
                elif 0 < self.n_components < 1:
                    max_comp = (eigs_csum > self.retain_variance).nonzero()[1, 0]
                elif self.n_components < min(nsamples * ncomps, nprops):
                    max_comp = self.n_components
                else:
                    # Use all the available components
                    warnings.warn(
                        (
                            f"n_components={self.n_components} too big: "
                            "retaining everything"
                        ),
                        stacklevel=3,
                    )
                    max_comp = min(nsamples * ncomps, nprops)
                retained = int_array_like(range(max_comp), eigs_ratio)
                eigs = eigs[retained]
                eigs_ratio = eigs_ratio[retained]
                components = components[:, retained]

                values = values.reshape(nsamples * ncomps, nprops)
                values = (values - mean) @ components
                values = values.reshape(nsamples, ncomps, len(retained))

            # Append values for new TensorMap
            values_.append(values)

            # Append PCA information
            explained_variance_.append(eigs)
            explained_variance_ratio_.append(eigs_ratio)
            components_.append(components)
            retained_components_.append(retained)

        # Update class attributes
        self.values_ = values_
        self.explained_variance_ = explained_variance_
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.components_ = components_
        self.retained_components_ = retained_components_

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError(f"{self} is not fitted.")

    def transform(self, tensor: TensorMap) -> TensorMap:
        self._check_is_fitted()

        blocks: List[TensorBlock] = []

        for idx, block in enumerate(tensor.blocks()):
            values = self.values_[idx]

            n_comp = values.shape[-1]
            if values.shape[0] == 0 and self.n_components:
                n_comp = self.n_components

            properties = Labels(
                names=["pc"],
                values=int_array_like(
                    [[i] for i in range(n_comp)], like=block.properties.values
                ),
            )

            block = TensorBlock(
                values=values,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            blocks.append(block)

        return TensorMap(tensor.keys, blocks)

    def fit_transform(self, tensor: TensorMap) -> TensorMap:
        return self.fit(tensor).transform(tensor)
