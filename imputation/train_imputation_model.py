"""
The Implementation of a missing data imputation model. The model is a mixture of
 categorical distributions, trained via EM.
Model:
p(x) = p(h) * p(x|h) = p(h) * prod_i p(x_i|h)
where h is a latent variable, x is the observed variable, and x_i is the i-th feature
 of x. Features are assumed to be independent given h. Each p(x_i|h) is a categorical
   distribution. Features are originally continuous, but we discretize them into bins.

Author: Ahmed H. Shahin
Date: 31/8/2023
"""

import pickle

import numpy as np
from scipy.special import logsumexp, softmax


def condp(arr, axis=None):
    """Computes conditional probability"""
    if axis is None:
        return arr / arr.sum()
    return arr / arr.sum(axis=axis, keepdims=True)


def condexp(arr, dist_var=None):
    """
    Compute the softmax of x along dist_var. (useful for computing conditional
      probability from unnormalized log probability)

    Args:
        arr: a vector of unnormalized log probabilities
        dist_var: index of the variable to be normalized (default: None, normalize all
          variables, i.e., x is a joint distribution)

    Returns:
        a vector of conditional probabilities
    """
    if dist_var is None:
        return softmax(arr)
    return softmax(arr, axis=dist_var)


def predict_missing_values(p_xm_given_xo, combs, missing_idx, selection):
    """
    Predict the value for each missing feature according to the distribution p(x_m|x_o).

    Args:
        p_xm_given_xo: the distribution of x_m given x_o (np.array of shape
          (num_combinations,)).
        combs: all possible combinations of missing features (np.array of shape
          (num_combinations, num_missing_features)).
        missing_idx: the indices of missing features (np.array of shape
          (num_missing_features,)).
        selection: the method to select the value for each missing feature according to
          the distribution.

    Returns:
        y: the predicted value for each missing feature.
    """
    pred = np.zeros_like(missing_idx, dtype=int)

    if selection == "max":
        pred = combs[np.argmax(p_xm_given_xo)]
    elif selection == "sample":
        pred = combs[np.random.choice(np.arange(len(combs)), p=p_xm_given_xo)]
    elif selection == "mean":
        pred = combs[int((p_xm_given_xo * np.arange(len(combs))).sum().round())]

    return pred


class Discretizer:
    """
    Discretize continuous features into vins, using equal-frequency binning.

    Args:
        cont_feats_idx (list): list of indices of continuous features
        n_bins (int or (list or tuple)): number of bins for each continuous feature.
          If int, the same number of bins is used for all features.
    """

    def __init__(self, cont_feats_idx=(0, 4, 5), n_bins=10):
        assert isinstance(cont_feats_idx, (list, tuple)), "cont_feats_idx must be a list or tuple"
        assert isinstance(n_bins, (int, list)), "n_bins must be an int or a list"
        self.cont_feats_idx = cont_feats_idx
        self.n_bins = (
            n_bins if isinstance(n_bins, list) else [n_bins] * len(cont_feats_idx)
        )

        self.bins = []
        # representative values for each bin, to be used for imputation and converting
        # back to continuous values
        self.representative_values = []

    def fit(self, arr):
        """
        Fit the discretizer to the data to decide on the bins.

        Args:
            arr (np.array): array of data
        """
        print("Fitting discretizer...")
        for i, feat_idx in enumerate(self.cont_feats_idx):
            col = arr[:, feat_idx]
            _bins = np.nanquantile(col, np.linspace(0, 1, self.n_bins[i] + 1))
            _bins[0] = -np.inf
            _bins[-1] = np.inf
            self.bins.append(_bins)

            # representative values are the mean of each bin
            col = col[~np.isnan(col)]
            col_disc = np.digitize(col, _bins, right=True) - 1
            _rep_value = np.array(
                [np.mean(col[col_disc == bin]) for bin in range(self.n_bins[i])]
            )
            self.representative_values.append(_rep_value)

    def transform(self, arr):
        """
        Transform the data into discrete values using the bins.

        Args:
            arr (np.array): array of continuous data

        Returns:
            np.array: array of discretized data
        """
        arr = arr.copy()
        # ignore nan values when discretizing
        for i, feat_idx in enumerate(self.cont_feats_idx):
            arr[:, feat_idx][~np.isnan(arr[:, feat_idx])] = (
                np.digitize(
                    arr[:, feat_idx][~np.isnan(arr[:, feat_idx])],
                    self.bins[i],
                    right=True,
                )
                - 1
            )
        return arr

    def inverse_transform(self, arr):
        """
        Transform the data back to continuous values using the representative values.

        Args:
            arr (np.array): array of discretized data

        Returns:
            np.array: array of continuous data
        """
        arr = arr.copy()
        for i, feat_idx in enumerate(self.cont_feats_idx):
            arr[:, feat_idx][~np.isnan(arr[:, feat_idx])] = self.representative_values[
                i
            ][arr[:, feat_idx][~np.isnan(arr[:, feat_idx])].astype(int)]
        return arr

    def __repr__(self) -> str:
        return (
            f"Discretizer(cont_feats_idx={self.cont_feats_idx}, n_bins={self.n_bins})"
        )


class EM:
    """
    Fitting the latent variable model using EM algorithm.

    Args:
        num_latent_states (int): number of states of the latent variable
        n_iter (int): number of iterations
        num_categories (np.array): number of categories for each feature
        discretizer (Discretizer): discretizer object
        train_data (np.array): training data (discretized)
        tol (float): tolerance for convergence
    """

    def __init__(
        self,
        num_latent_states: int,
        n_iter: int,
        num_categories: np.array,
        discretizer: Discretizer,
        train_data: np.array,
        tol: float = 1e-3,
    ) -> None:
        self.num_latent_states = num_latent_states
        self.n_iter = n_iter
        self.tol = tol
        self.num_categories = num_categories
        self.num_feats = len(num_categories)
        self.discretizer = discretizer
        self.train_data = train_data
        self.eps = 1e-10
        self.hist = {"train_loglikelihood": [], "val_loglikelihood": []}
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initializes parameters"""
        self.p_h = condp(np.random.rand(self.num_latent_states), axis=0)
        # the line below would be more efficient if number of categories is the same for
        # all features, then we can use a single array
        self.p_x_given_h = [
            condp(
                np.random.rand(self.num_latent_states, self.num_categories[i]), axis=1
            )
            for i in range(self.num_feats)
        ]

    def __repr__(self) -> str:
        return f"EM(H={self.num_latent_states}, n_iter={self.n_iter},\
              num_categories={self.num_categories}, tol={self.tol})"

    def fit(self, val_data: np.array) -> None:
        """
        Fits the model to the data

        Args:
            val_data (np.array): validation data (discretized)
        """
        best_val_loglikelihood = -np.inf
        train_data = self.train_data
        for iteration in range(self.n_iter):
            train_loglikelihood = self.em_step(train_data)
            val_loglikelihood = self.compute_likelihood(val_data)
            self.hist["train_loglikelihood"].append(train_loglikelihood)
            self.hist["val_loglikelihood"].append(val_loglikelihood)

            if val_loglikelihood > best_val_loglikelihood:
                best_val_loglikelihood = val_loglikelihood
                self.best_params = self.p_h, self.p_x_given_h

            print(
                f"Iteration {iteration+1}/{self.n_iter}: train log-likelihood = \
                    {train_loglikelihood:.4f}, val log-likelihood = \
                        {val_loglikelihood:.4f}"
            )

            if (
                iteration > 0
                and np.abs(
                    self.hist["train_loglikelihood"][-1]
                    - self.hist["train_loglikelihood"][-2]
                )
                < self.tol
            ):
                break

    def em_step(self, x_train) -> None:
        """
        Performs one EM step on the data.

        Args:
            x_train (np.array): training data (discretized)

        Returns:
            log-likelihood of the training data
        """
        p_h_tot = np.zeros(self.num_latent_states)
        p_x_given_h_tot = [
            np.zeros((self.num_latent_states, self.num_categories[i]))
            for i in range(self.num_feats)
        ]
        loglikelihood = 0

        for tr_sample in x_train:
            observed_idx = np.where(~np.isnan(tr_sample))[0]
            log_q_h_given_x = np.log(self.p_h.copy() + self.eps)
            for obs_feat_idx in observed_idx:
                log_q_h_given_x += np.log(
                    self.p_x_given_h[obs_feat_idx][
                        :, tr_sample[obs_feat_idx].astype(int)
                    ]
                    + self.eps
                )

            q_h_given_x = condexp(log_q_h_given_x)
            loglikelihood += logsumexp(log_q_h_given_x)
            p_h_tot += q_h_given_x

            for obs_feat_idx in observed_idx:
                p_x_given_h_tot[obs_feat_idx][
                    :, tr_sample[obs_feat_idx].astype(int)
                ] += q_h_given_x

        self.p_h = condp(p_h_tot)
        for feat_idx in range(self.num_feats):
            self.p_x_given_h[feat_idx] = condp(p_x_given_h_tot[feat_idx], axis=1)
        return loglikelihood / len(x_train)

    def compute_likelihood(self, arr) -> float:
        """Computes the likelihood of given array of data"""
        loglikelihood = 0
        for record in arr:
            observed_idx = np.where(~np.isnan(record))[0]
            log_q_h_given_x = np.log(self.p_h.copy() + self.eps)
            for i in observed_idx:
                log_q_h_given_x += np.log(
                    self.p_x_given_h[i][:, record[i].astype(int)] + self.eps
                )
            loglikelihood += logsumexp(log_q_h_given_x)
        return loglikelihood / len(arr)

    def predict(self, arr, method="argmax", use_best=False) -> np.array:
        """
        Predict the missing values of given array of data
        p(x_m | x_o) = sum_h p(x_m | h) * p(h | x_o)
          \\propto sum_h p(x_m | h) * p(x_o | h) * p(h)
        log p(x_m | x_o) = logsumexp_h log p(x_m | h) + log p(x_o | h) + log p(h)

        Args:
            arr (np.array): array of data
            method (str): method for selecting the missing values.
              Can be 'argmax', 'sample', or 'mean'
            use_best (bool): whether to use the best parameters or the current
              parameters

        Returns:
            np.array: array of data with missing values imputed
        """
        assert method in ["argmax", "sample", "mean"], "Invalid selection method"
        assert arr.ndim == 2, "Input must be a 2D array"
        arr = arr.copy()

        if use_best:
            p_h, p_x_given_h = self.best_params
        else:
            p_h, p_x_given_h = self.p_h, self.p_x_given_h

        output = []
        for record in arr:
            observed_idx = np.where(~np.isnan(record))[0]
            missing_idx = np.where(np.isnan(record))[0]
            if len(missing_idx) == 0:
                output.append(record)
                continue

            combs = np.array(
                np.meshgrid(*[np.arange(self.num_categories[i]) for i in missing_idx])
            ).T.reshape(-1, len(missing_idx))
            logp_xm_given_xo = (
                np.log(p_h.copy() + self.eps).reshape(1, -1).repeat(len(combs), axis=0)
            )
            for i in observed_idx:
                logp_xm_given_xo += np.log(
                    p_x_given_h[i][:, record[i].astype(int)] + self.eps
                )
            for i, comb in enumerate(combs):
                record[missing_idx] = comb
                for j in missing_idx:
                    logp_xm_given_xo[i] += np.log(
                        p_x_given_h[j][:, record[j].astype(int)] + self.eps
                    )
            p_xm_given_xo = condexp(logsumexp(logp_xm_given_xo, axis=1))

            imputed_record = record.copy()
            imputed_record[missing_idx] = predict_missing_values(
                p_xm_given_xo, combs, missing_idx, method
            )

            output.append(imputed_record)

        return np.array(output)

    def save_model(self, path):
        """Save the model to a pickle file"""
        with open(path, "wb") as file:
            pickle.dump(self, file)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import pandas as pd

    parser = ArgumentParser()
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-save_path', type=str)
    parser.add_argument('-n_iter', type=int, default=100)
    parser.add_argument('-n_latent_states', type=int, default=10)
    parser.add_argument('-n_bins', type=int, default=10)
    parser.add_argument('-tol', type=float, default=1e-3)
    parser.add_argument('-val_size', type=float, default=0.2)
    parser.add_argument('-cont_feats_idx', type=int, nargs='+', default=[0, 4, 5])
    args = parser.parse_args()

    data = pd.read_csv(args.data_path).values # missing values are represented by nan
    train_data = data[:int((1-args.val_size)*len(data))]
    val_data = data[int((1-args.val_size)*len(data)):]
    num_categories = np.array([args.n_bins]*data.shape[1])

    discretizer = Discretizer(args.cont_feats_idx, args.n_bins)
    discretizer.fit(train_data)

    train_data = discretizer.transform(train_data)
    val_data = discretizer.transform(val_data)

    em = EM(args.n_latent_states, args.n_iter, num_categories, discretizer, train_data, args.tol)
    em.fit(val_data)
    em.save_model(args.save_path)
