"""
Paper-specific GPyTorch GP-BT implementation used by CF-BILD.

This source is deposited with CF-BILD so that the compositional-kernel and
predefined-split interfaces required by the paper do not depend on an
unpublished local package. It is derived from bayesian-gp-cvloss 0.1.7
(Shifa Zhong, MIT License) and uses a GPyTorch backend.

Key change: per-component shared lengthscale (not per-feature ARD) for
compositional kernels — reduces Hyperopt search from ~2000+ dims to ~8 dims.
"""

import pandas as pd
import numpy as np
import torch
import gpytorch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import logging
import gc

logger = logging.getLogger(__name__)

DEFAULT_KERNELS = {
    "Matern32": "Matern32",
    "Matern52": "Matern52",
    "RBF": "RBF",
    "RationalQuadratic": "RationalQuadratic",
}


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _make_base_kernel(kernel_name, lengthscale, variance, active_dims=None,
                      ard=False, dtype=torch.float32):
    """Build a GPyTorch ScaleKernel(BaseKernel).

    Args:
        kernel_name: kernel type string
        lengthscale: scalar (shared) or 1D array (ARD)
        variance: output scale
        active_dims: list of active dimensions or None
        ard: if True and lengthscale is scalar, still create non-ARD kernel
        dtype: tensor dtype
    """
    # Determine ARD dims
    if isinstance(lengthscale, (list, np.ndarray)) and len(lengthscale) > 1:
        ard_num_dims = len(lengthscale)
    else:
        ard_num_dims = None  # shared/isotropic lengthscale

    if kernel_name == "RBF":
        base = gpytorch.kernels.RBFKernel(
            ard_num_dims=ard_num_dims, active_dims=active_dims)
    elif kernel_name == "Matern52":
        base = gpytorch.kernels.MaternKernel(
            nu=2.5, ard_num_dims=ard_num_dims, active_dims=active_dims)
    elif kernel_name == "Matern32":
        base = gpytorch.kernels.MaternKernel(
            nu=1.5, ard_num_dims=ard_num_dims, active_dims=active_dims)
    elif kernel_name == "RationalQuadratic":
        base = gpytorch.kernels.RQKernel(
            ard_num_dims=ard_num_dims, active_dims=active_dims)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    # Set lengthscale
    ls_val = np.atleast_1d(np.asarray(lengthscale, dtype=np.float32))
    base.lengthscale = torch.tensor(ls_val, dtype=dtype).unsqueeze(0)

    scaled = gpytorch.kernels.ScaleKernel(base)
    scaled.outputscale = torch.tensor(float(variance), dtype=dtype)
    return scaled


def _build_compositional_kernel(kernel_name, params, dim_cat, dim_an,
                                 kernel_form, num_features, dtype=torch.float32):
    """Build compositional kernel with per-component SHARED lengthscale.

    Params expected:
        ls_cat: shared lengthscale for cation sub-kernel
        ls_an: shared lengthscale for anion sub-kernel
        ls_cross: shared lengthscale for cross sub-kernel
        ls_env_T, ls_env_P: lengthscales for T and P (if extra dims exist)
        kernel_variance: output variance
    """
    total_dims = num_features or (dim_cat + dim_an)
    extra_dims = total_dims - dim_cat - dim_an

    cat_dims = list(range(dim_cat))
    an_dims = list(range(dim_cat, dim_cat + dim_an))
    all_fp_dims = cat_dims + an_dims
    env_dims = list(range(dim_cat + dim_an, total_dims)) if extra_dims > 0 else []

    ls_cat = float(params['ls_cat'])
    ls_an = float(params['ls_an'])
    variance = float(params['kernel_variance'])
    has_cross = kernel_form in ("product", "additive")
    n_components = 3.0 if has_cross else 2.0
    is_product = kernel_form in ("product", "product_no_cross")
    var_component = variance ** (1.0 / n_components) if is_product else variance / n_components

    k_cat = _make_base_kernel(kernel_name, ls_cat, var_component,
                               active_dims=cat_dims, dtype=dtype)
    k_an = _make_base_kernel(kernel_name, ls_an, var_component,
                              active_dims=an_dims, dtype=dtype)
    if kernel_form == "product":
        ls_cross = float(params['ls_cross'])
        k_cross = _make_base_kernel(kernel_name, ls_cross, var_component,
                                     active_dims=all_fp_dims, dtype=dtype)
        k_struct = gpytorch.kernels.ProductKernel(k_cat, k_an) + k_cross
    elif kernel_form == "additive":
        ls_cross = float(params['ls_cross'])
        k_cross = _make_base_kernel(kernel_name, ls_cross, var_component,
                                     active_dims=all_fp_dims, dtype=dtype)
        k_struct = k_cat + k_an + k_cross
    elif kernel_form == "product_no_cross":
        k_struct = gpytorch.kernels.ProductKernel(k_cat, k_an)
    elif kernel_form == "additive_no_cross":
        k_struct = k_cat + k_an
    else:
        raise ValueError(f"Unknown kernel_form: {kernel_form}")

    # If there are environmental dims (T, P), add a separate kernel multiplied in
    if extra_dims > 0:
        env_ls = []
        for i, dim_idx in enumerate(env_dims):
            env_ls.append(float(params.get(f'ls_env_{i}', 1.0)))
        k_env = _make_base_kernel(kernel_name, env_ls, 1.0,
                                   active_dims=env_dims, dtype=dtype)
        return gpytorch.kernels.ProductKernel(k_struct, k_env)

    return k_struct


def _build_standard_kernel(kernel_name, params, num_features, dtype=torch.float32):
    """Build standard (non-compositional) kernel with per-feature ARD."""
    ls = np.array([float(params[f'lengthscales_{i}']) for i in range(num_features)],
                  dtype=np.float32)
    variance = float(params['kernel_variance'])
    return _make_base_kernel(kernel_name, ls, variance, dtype=dtype)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPCrossValidatedOptimizer:
    """
    Optimizes GP hyperparameters via Hyperopt with k-fold CV.
    Uses per-component shared lengthscales for compositional kernels.
    """
    def __init__(self, X_train, y_train,
                 hyperopt_space=None,
                 kernels=None,
                 lengthscale_bounds=None,
                 kernel_variance_bounds=None,
                 noise_variance_bounds=None,
                 n_splits=5, random_state=None,
                 predefined_cv_splits=None,
                 compositional_kernel_dims=None,
                 kernel_form="standard",
                 device=None):

        if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
            raise ValueError("X_train must be a pandas DataFrame or NumPy ndarray.")
        if not isinstance(y_train, (pd.Series, np.ndarray)):
            raise ValueError("y_train must be a pandas Series or NumPy ndarray.")

        if isinstance(X_train, np.ndarray) and len(X_train.shape) != 2:
            raise ValueError("X_train as NumPy array must be 2D.")

        _y = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)
        if _y.ndim == 2 and _y.shape[1] == 1:
            _y = _y.flatten()
        if _y.ndim != 1:
            raise ValueError("y_train must be 1D.")

        if X_train.shape[0] != _y.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples.")

        self.X_train = X_train
        self.y_train_1d = _y
        self.y_train_mean_ = np.mean(self.y_train_1d)
        self.n_splits = n_splits
        self.random_state = random_state
        self.num_features = X_train.shape[1]
        self.device = device or _get_device()

        # Predefined CV splits
        self.predefined_cv_splits = predefined_cv_splits
        if predefined_cv_splits is not None:
            self.n_splits = len(predefined_cv_splits)

        # Compositional kernel
        self.compositional_kernel_dims = compositional_kernel_dims
        self.kernel_form = kernel_form
        if compositional_kernel_dims is not None:
            dim_cat, dim_an = compositional_kernel_dims
            self.extra_dims = self.num_features - dim_cat - dim_an
            logger.info(f"Compositional kernel: cat={dim_cat}, an={dim_an}, "
                        f"extra={self.extra_dims}, form={kernel_form}")
        else:
            self.extra_dims = 0

        # Kernels
        if kernels is not None:
            invalid = [k for k in kernels if k not in DEFAULT_KERNELS]
            if invalid:
                raise ValueError(f"Unknown kernel(s): {invalid}")
            self._active_kernels = {k: DEFAULT_KERNELS[k] for k in kernels}
        else:
            self._active_kernels = dict(DEFAULT_KERNELS)

        # Bounds
        self._lengthscale_bounds = self._validate_bounds(lengthscale_bounds, "lengthscale_bounds")
        self._kernel_variance_bounds = self._validate_bounds(kernel_variance_bounds, "kernel_variance_bounds")
        self._noise_variance_bounds = self._validate_bounds(noise_variance_bounds, "noise_variance_bounds")

        # Hyperopt space
        if hyperopt_space is not None:
            self.hyperopt_space = hyperopt_space
        else:
            self.hyperopt_space = self._get_default_data_dependent_space()

        self.trials = Trials()
        self.best_params = None
        self.best_model_ = None
        self.best_likelihood_ = None
        self._iteration_count = 0

    @staticmethod
    def _validate_bounds(bounds, name):
        if bounds is None:
            return None
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(f"{name} must be a (low, high) tuple")
        low, high = bounds
        if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
            raise ValueError(f"{name} values must be numeric")
        if low <= 0 or high <= 0:
            raise ValueError(f"{name} values must be positive")
        if low >= high:
            raise ValueError(f"{name} requires low < high")
        return (float(low), float(high))

    def _get_default_data_dependent_space(self):
        """Generate compact hyperopt space with per-component shared lengthscales."""
        logger.info("Generating per-component shared-lengthscale hyperparameter space.")

        X_data = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train

        # Compute global feature std for lengthscale bounds
        feat_stds = np.std(X_data, axis=0)
        global_std = np.median(feat_stds[feat_stds > 1e-9]) if np.any(feat_stds > 1e-9) else 1.0

        if self._lengthscale_bounds is None:
            ls_lo = max(1e-3, global_std * 0.01)
            ls_hi = max(ls_lo * 10, global_std * 100.0)
        else:
            ls_lo, ls_hi = self._lengthscale_bounds

        space = {}

        if self.compositional_kernel_dims is not None:
            # Per-component shared lengthscales
            space['ls_cat'] = hp.loguniform('ls_cat', np.log(ls_lo), np.log(ls_hi))
            space['ls_an'] = hp.loguniform('ls_an', np.log(ls_lo), np.log(ls_hi))
            if self.kernel_form in ("product", "additive"):
                space['ls_cross'] = hp.loguniform(
                    'ls_cross', np.log(ls_lo), np.log(ls_hi))

            # Environmental dims (T, P) get individual lengthscales
            for i in range(self.extra_dims):
                space[f'ls_env_{i}'] = hp.loguniform(
                    f'ls_env_{i}', np.log(ls_lo), np.log(ls_hi))
        else:
            # Per-feature ARD for standard (non-compositional) kernel
            for i in range(self.num_features):
                space[f'lengthscales_{i}'] = hp.loguniform(
                    f'lengthscales_{i}', np.log(ls_lo), np.log(ls_hi))

        # Kernel variance
        y_var = np.var(self.y_train_1d)
        if self._kernel_variance_bounds is None:
            kv_lo = 1e-6
            kv_hi = max(kv_lo * 10, float(2.0 * y_var))
        else:
            kv_lo, kv_hi = self._kernel_variance_bounds
        space['kernel_variance'] = hp.loguniform('kernel_variance',
                                                  np.log(kv_lo), np.log(kv_hi))

        # Noise variance
        y_std = np.std(self.y_train_1d)
        if self._noise_variance_bounds is None:
            if y_std > 1e-9:
                nv_lo = max(1e-9, (y_std / 100.0) ** 2)
                nv_hi = max(nv_lo * 1.1 + 1e-9, y_std ** 2)
            else:
                nv_lo, nv_hi = 1e-9, 1e-2
        else:
            nv_lo, nv_hi = self._noise_variance_bounds
        space['likelihood_noise_variance'] = hp.loguniform(
            'likelihood_noise_variance', np.log(nv_lo), np.log(nv_hi))

        # Kernel choice
        space['kernel_name'] = hp.choice('kernel_name', list(self._active_kernels.keys()))

        n_params = len(space)
        logger.info(f"Search space: {n_params} parameters "
                    f"(ls_bounds=[{ls_lo:.4f}, {ls_hi:.4f}])")
        return space

    def _build_kernel(self, params, dtype=torch.float32):
        kernel_name = params['kernel_name']
        if kernel_name not in DEFAULT_KERNELS:
            return None

        if self.compositional_kernel_dims is not None:
            dim_cat, dim_an = self.compositional_kernel_dims
            return _build_compositional_kernel(
                kernel_name, params, dim_cat, dim_an,
                self.kernel_form, self.num_features, dtype=dtype)
        else:
            return _build_standard_kernel(
                kernel_name, params, self.num_features, dtype=dtype)

    def _evaluate_fold(self, X_train_np, y_train_np, X_val_np, y_val_np, params):
        device = self.device
        noise_var = max(float(params['likelihood_noise_variance']), 1e-4)

        y_mean = np.mean(y_train_np)
        y_tr_c = y_train_np - y_mean
        y_va_c = y_val_np - y_mean

        dtype = torch.float32
        X_tr = torch.tensor(X_train_np, dtype=dtype, device=device)
        y_tr = torch.tensor(y_tr_c, dtype=dtype, device=device)
        X_va = torch.tensor(X_val_np, dtype=dtype, device=device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
        likelihood.noise = torch.tensor(noise_var, dtype=dtype)
        likelihood = likelihood.to(device)

        kernel = self._build_kernel(params, dtype=dtype)
        if kernel is None:
            return np.inf, np.inf

        model = ExactGPModel(X_tr, y_tr, likelihood, kernel).to(device)

        for p in model.parameters():
            p.requires_grad = False

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_val = likelihood(model(X_va))
            y_pred_val = pred_val.mean.cpu().numpy()
            val_rmse = np.sqrt(mean_squared_error(y_va_c, y_pred_val))

            pred_tr = likelihood(model(X_tr))
            y_pred_tr = pred_tr.mean.cpu().numpy()
            train_rmse = np.sqrt(mean_squared_error(y_tr_c, y_pred_tr))

        del model, likelihood, X_tr, y_tr, X_va, kernel
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return val_rmse, train_rmse

    def _objective(self, params):
        self._iteration_count += 1
        iteration_num = self._iteration_count
        kernel_name = params.get('kernel_name', '?')

        # Determine CV folds
        if self.predefined_cv_splits is not None:
            cv_folds = self.predefined_cv_splits
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=True,
                       random_state=self.random_state)
            X_data = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
            y_data = self.y_train_1d
            cv_folds = [
                (X_data[tri], y_data[tri], X_data[vai], y_data[vai])
                for tri, vai in kf.split(X_data)
            ]

        fold_rmses, fold_train_rmses = [], []

        for fold_idx, (X_tr, y_tr, X_va, y_va) in enumerate(cv_folds):
            X_tr = X_tr.values if isinstance(X_tr, pd.DataFrame) else np.asarray(X_tr)
            X_va = X_va.values if isinstance(X_va, pd.DataFrame) else np.asarray(X_va)
            y_tr = np.asarray(y_tr).flatten()
            y_va = np.asarray(y_va).flatten()

            try:
                val_rmse, train_rmse = self._evaluate_fold(X_tr, y_tr, X_va, y_va, params)
                fold_rmses.append(val_rmse)
                fold_train_rmses.append(train_rmse)
            except Exception as e:
                logger.warning(f"Fold {fold_idx+1} error: {e}")
                fold_rmses.append(np.inf)
                fold_train_rmses.append(np.inf)
                break

        avg_cv = np.mean(fold_rmses) if np.all(np.isfinite(fold_rmses)) else np.inf
        avg_tr = np.mean(fold_train_rmses) if np.all(np.isfinite(fold_train_rmses)) else np.inf

        noise = float(params.get('likelihood_noise_variance', 0))
        # Build compact log string
        ls_info = ""
        if self.compositional_kernel_dims is not None:
            ls_info = (f"ls_cat={params.get('ls_cat',0):.3f} "
                       f"ls_an={params.get('ls_an',0):.3f}")
            if 'ls_cross' in params:
                ls_info += f" ls_cross={params['ls_cross']:.3f}"
        logger.info(f"Iter {iteration_num:>3} | CV: {avg_cv:<8.4f} | "
                    f"Train: {avg_tr:<8.4f} | {kernel_name} | "
                    f"{ls_info} | Noise: {noise:.6f}")

        return {
            'loss': avg_cv, 'status': STATUS_OK, 'params': params,
            'iteration': iteration_num, 'train_loss': avg_tr
        }

    def optimize(self, max_evals=100, tpe_algo=tpe.suggest,
                 early_stop_fn=None, rstate_seed=None):
        self._iteration_count = 0
        if rstate_seed is None and self.random_state is not None:
            rstate_seed = self.random_state

        rstate = np.random.default_rng(rstate_seed) if rstate_seed is not None else None

        self.best_params_raw_ = fmin(
            fn=self._objective,
            space=self.hyperopt_space,
            algo=tpe_algo,
            max_evals=max_evals,
            trials=self.trials,
            early_stop_fn=early_stop_fn,
            rstate=rstate
        )

        logger.info(f"Optimization finished.")

        if (self.trials.best_trial and 'result' in self.trials.best_trial
                and self.trials.best_trial['result']['status'] == STATUS_OK):
            self.best_params = self.trials.best_trial['result']['params']
            best_loss = self.trials.best_trial['result']['loss']
            best_train = self.trials.best_trial['result']['train_loss']
            logger.info(f"Best CV RMSE: {best_loss:.4f}, Train RMSE: {best_train:.4f}")
            self._calibrate_variance()
            self.refit_best_model()
        else:
            self.best_params = None
            logger.warning("Optimization did not yield a valid best trial.")

        return self.best_params

    def _calibrate_variance(self, target_coverage=0.95):
        """Calibrate predictive variance using CV residuals.

        Computes a scaling factor so that the 95% CI achieves ~95% coverage
        on the CV validation folds. This corrects for over/under-confident σ.
        """
        if not self.best_params or self.predefined_cv_splits is None:
            self.variance_scale_ = 1.0
            return

        params = self.best_params
        all_residuals = []
        all_pred_std = []

        # Clear GPU cache before calibration
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        for X_tr, y_tr, X_va, y_va in self.predefined_cv_splits:
            X_tr = np.asarray(X_tr)
            X_va = np.asarray(X_va)
            y_tr = np.asarray(y_tr).flatten()
            y_va = np.asarray(y_va).flatten()

            try:
                noise_var = max(float(params['likelihood_noise_variance']), 1e-4)
                y_mean = np.mean(y_tr)
                y_tr_c = y_tr - y_mean

                dtype = torch.float32
                device = self.device
                X_t = torch.tensor(X_tr, dtype=dtype, device=device)
                y_t = torch.tensor(y_tr_c, dtype=dtype, device=device)
                X_v = torch.tensor(X_va, dtype=dtype, device=device)

                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
                likelihood.noise = torch.tensor(noise_var, dtype=dtype)
                likelihood = likelihood.to(device)

                kernel = self._build_kernel(params, dtype=dtype)
                model = ExactGPModel(X_t, y_t, likelihood, kernel).to(device)
                for p in model.parameters():
                    p.requires_grad = False
                model.eval()
                likelihood.eval()

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred = likelihood(model(X_v))
                    pred_mean = pred.mean.cpu().numpy() + y_mean
                    pred_var = pred.variance.cpu().numpy()

                residuals = np.abs(y_va - pred_mean)
                pred_stds = np.sqrt(pred_var)

                all_residuals.extend(residuals.tolist())
                all_pred_std.extend(pred_stds.tolist())

                del model, likelihood, kernel, X_t, y_t, X_v, pred
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger.warning(f"Calibration fold error: {e}")
                continue

        if len(all_residuals) == 0:
            self.variance_scale_ = 1.0
            return

        # Find scale factor: we want P(|residual| < 1.96 * scale * pred_std) ≈ 0.95
        all_residuals = np.array(all_residuals)
        all_pred_std = np.array(all_pred_std)

        # Compute z-scores with current σ
        z_scores = all_residuals / (all_pred_std + 1e-10)

        # Find the scale factor so that 95% of z-scores fall within 1.96 * scale
        z_95 = np.percentile(z_scores, target_coverage * 100)
        scale = z_95 / 1.96

        # Ensure scale >= 1 (never shrink variance)
        self.variance_scale_ = max(scale, 1.0) ** 2  # square because we scale variance
        logger.info(f"Variance calibration: scale={self.variance_scale_:.4f} "
                    f"(raw z95={z_95:.4f})")

    def refit_best_model(self):
        if not self.best_params:
            self.best_model_ = None
            return None

        params = self.best_params
        noise_var = max(float(params['likelihood_noise_variance']), 1e-4)

        X_data = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        y_centered = (self.y_train_1d - self.y_train_mean_)

        dtype = torch.float32
        device = self.device
        X_t = torch.tensor(X_data, dtype=dtype, device=device)
        y_t = torch.tensor(y_centered, dtype=dtype, device=device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
        likelihood.noise = torch.tensor(noise_var, dtype=dtype)
        likelihood = likelihood.to(device)

        kernel = self._build_kernel(params, dtype=dtype)
        model = ExactGPModel(X_t, y_t, likelihood, kernel).to(device)

        for p in model.parameters():
            p.requires_grad = False

        model.eval()
        likelihood.eval()

        self.best_model_ = model
        self.best_likelihood_ = likelihood
        self._train_x = X_t
        self._train_y = y_t
        logger.info("Model refitted on full training data.")
        return model

    def predict(self, X_new_processed):
        if self.best_model_ is None:
            logger.error("No best model. Run optimize() first.")
            return None, None

        if isinstance(X_new_processed, pd.DataFrame):
            X_new_processed = X_new_processed.values
        X_new_processed = np.asarray(X_new_processed, dtype=np.float64)

        if X_new_processed.shape[1] != self.num_features:
            raise ValueError(
                f"X_new has {X_new_processed.shape[1]} features, "
                f"model expects {self.num_features}.")

        device = self.device
        X_new = torch.tensor(X_new_processed, dtype=torch.float32, device=device)

        self.best_model_.eval()
        self.best_likelihood_.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.best_likelihood_(self.best_model_(X_new))
            mean_centered = pred.mean.cpu().numpy()
            variance = pred.variance.cpu().numpy()

        # Apply variance calibration
        scale = getattr(self, 'variance_scale_', 1.0)
        variance = variance * scale

        mean_original = mean_centered + self.y_train_mean_
        return mean_original.reshape(-1, 1), variance.reshape(-1, 1)

    def get_optimization_results(self):
        return self.trials
