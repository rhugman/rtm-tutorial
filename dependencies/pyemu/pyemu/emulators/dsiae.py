"""
Data Space Inversion Autoencoder (DSIAE) emulator.
"""
from __future__ import print_function, division
import os
import numpy as np
import pandas as pd

from .dsi import DSI
from pyemu.en import ParameterEnsemble

try:
    import tensorflow as tf
except ImportError:
    tf = None


class DSIAE(DSI):
    """A non-linear Data Space Inversion emulator.

    A variational autoencoder replaces the SVD of :class:`DSI`. The KL term
    pulls the encoded prior toward ``N(0, I)``, and a post-fit affine
    calibration of the encoder output recenters and rescales each latent
    coordinate, so the latent-prior statement that DSI writes (``parval1=0``
    and a unit-standard-deviation parcov) holds exactly for the first two
    marginal moments. The raw (pre-calibration) encoded-prior moments are
    reported as fit diagnostics in :attr:`fit_report`.
    """

    # finite backstop for a nonlinear decoder: ~10 sigma under the unit-normal
    # latent prior, so it never binds during sane updates but stops pathological
    # excursions into extrapolation territory where phi feedback is unreliable.
    _latent_par_bound = 10.0

    def __init__(self,
                 pst=None,
                 data=None,
                 transforms=None,
                 latent_dim=None,
                 energy_threshold=1.0,
                 verbose=False):
        """Initialize the DSIAE emulator.

        Parameters
        ----------
        pst : Pst or pandas.DataFrame, optional
            A Pst object or observation_data DataFrame.
        data : pandas.DataFrame or ObservationEnsemble, optional
            An ensemble of simulated observations used as training data.
        transforms : list of dict, optional
            Feature transform specifications (see :class:`DSI`).
        latent_dim : int, optional
            Number of latent coordinates. If None, chosen from
            ``energy_threshold`` via PCA.
        energy_threshold : float, optional
            Cumulative explained-variance threshold for the automatic latent
            dimension. Default is 1.0.
        verbose : bool, optional
            If True, enable verbose logging. Default is False.
        """
        if tf is None:
            raise ImportError(
                "DSIAE requires tensorflow, which is not installed")
        super().__init__(pst=pst, data=data, transforms=transforms,
                         energy_threshold=energy_threshold, verbose=verbose)
        self.verbose = verbose
        self._latent_dim = (int(latent_dim) if latent_dim is not None
                            else self._calc_latent_dim())

    @property
    def latent_dim(self):
        """Number of latent coordinates.

        Returns
        -------
        int
            The latent dimensionality of the autoencoder.
        """
        return self._latent_dim

    def _calc_latent_dim(self):
        """Choose the latent dimension from the energy threshold via PCA.

        Returns
        -------
        int
            The smallest number of principal components whose cumulative
            explained variance reaches ``energy_threshold``, capped at the
            number of features.
        """
        from sklearn.decomposition import PCA

        pca = PCA()
        pca.fit(self.data_transformed.values.astype(float))
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, self.energy_threshold) + 1)
        k = min(k, self.data_transformed.shape[1])
        self.logger.statement(f"selected latent dimension {k} from energy "
                              f"threshold {self.energy_threshold}")
        return k

    def compute_projection_matrix(self, energy_threshold=None):
        """Not applicable to DSIAE; the autoencoder is trained in :meth:`fit`."""
        raise NotImplementedError(
            "DSIAE has no projection matrix; call fit() to train the autoencoder")

    def fit(self, hidden_dims=(128, 64), lr=1e-3, beta=1e-3, epochs=300,
            batch_size=128, validation_split=0.1, early_stopping=True,
            patience=10, random_state=42):
        """Train the variational autoencoder on the transformed training data.

        Parameters
        ----------
        hidden_dims : tuple of int, optional
            Hidden layer sizes for the encoder (reversed for the decoder).
        lr : float, optional
            Adam learning rate.
        beta : float, optional
            Weight on the KL term in the VAE loss.
        epochs : int, optional
            Maximum number of training epochs.
        batch_size : int, optional
            Training batch size.
        validation_split : float, optional
            Fraction of training data held out for validation.
        early_stopping : bool, optional
            Stop early on validation loss when a validation split exists.
        patience : int, optional
            Early-stopping patience in epochs.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        DSIAE
            The fitted emulator. The ``latent_*`` entries of
            :attr:`fit_report` describe the raw encoder output, before the
            affine calibration that makes the encoded prior exactly
            zero-mean/unit-std per coordinate.
        """
        X = self.data_transformed
        if X is None:
            raise ValueError("No transformed training data available")

        # internal per-column standardization is the nonlinear analog of DSI
        # centering deviations inside the SVD: it makes the MSE/KL balance
        # scale-free.  inverted inside _reconstruct.
        self._vae_mean = X.mean()
        self._vae_std = X.std(ddof=1).replace(0.0, 1.0)
        Xs = ((X - self._vae_mean) / self._vae_std).values.astype("float32")

        keras_verbose = 2 if self.verbose else 0
        encoder, decoder, history = _train_vae(
            Xs, self.latent_dim, hidden_dims, lr, beta, epochs, batch_size,
            validation_split, early_stopping, patience, random_state,
            keras_verbose)
        self._encoder = encoder
        self._decoder = decoder

        # affine latent calibration: absorb the encoded prior's per-dimension
        # mean/std so the prior statement written to the control file
        # (parval1=0, unit-std parcov) holds exactly, not just by KL pressure.
        # encode() applies it; _reconstruct() inverts it before decoding.
        z_mean = np.asarray(encoder(Xs, training=False)[0], dtype=float)
        self._z_shift = z_mean.mean(axis=0)
        self._z_scale = z_mean.std(axis=0, ddof=1)
        self._z_scale[self._z_scale == 0.0] = 1.0

        recon = np.asarray(decoder(z_mean.astype("float32"), training=False),
                           dtype=float)
        recon = recon * self._vae_std.values + self._vae_mean.values

        latent_std = z_mean.std(axis=0, ddof=1)
        data_std = X.std(ddof=1).values
        safe = data_std != 0.0
        ratio = np.full(data_std.shape, np.nan)
        ratio[safe] = recon.std(axis=0, ddof=1)[safe] / data_std[safe]
        ratio = ratio[np.isfinite(ratio)]

        self.fit_report = {
            "latent_abs_mean_max": float(np.max(np.abs(z_mean.mean(axis=0)))),
            "latent_std_min": float(latent_std.min()),
            "latent_std_max": float(latent_std.max()),
            "recon_std_ratio_min": float(ratio.min()) if ratio.size else np.nan,
            "recon_std_ratio_median": float(np.median(ratio)) if ratio.size else np.nan,
        }
        if history is not None and history.history.get("loss"):
            self.fit_report["final_loss"] = float(history.history["loss"][-1])
        if history is not None and history.history.get("val_loss"):
            self.fit_report["final_val_loss"] = float(history.history["val_loss"][-1])

        self.logger.statement(
            "vae fit: raw latent_std=[{0:.3f}, {1:.3f}] (calibrated to unit "
            "std) recon_std_ratio_median={2:.3f}"
            .format(self.fit_report["latent_std_min"],
                    self.fit_report["latent_std_max"],
                    self.fit_report["recon_std_ratio_median"]))
        if self.fit_report["latent_std_min"] < 0.1:
            self.logger.warn(
                "an encoded latent dimension is near collapse (raw std < 0.1); "
                "calibration will amplify noise in that coordinate")
        if (np.isfinite(self.fit_report["recon_std_ratio_median"]) and
                self.fit_report["recon_std_ratio_median"] < 0.7):
            self.logger.warn(
                "reconstruction std ratio median below 0.7; reconstructions "
                "may be under-dispersed")

        self.fitted = True
        return self

    def encode(self, X):
        """Encode physical-space observations into latent coordinates.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Observations in physical (untransformed) space. DataFrame indices
            are preserved; ndarrays get a RangeIndex.

        Returns
        -------
        pandas.DataFrame
            Calibrated latent coordinates with columns
            ``p_0 .. p_{latent_dim-1}``; the encoded training ensemble has
            exactly zero mean and unit std per coordinate.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before encoding")

        if isinstance(X, pd.DataFrame):
            index = X.index
        else:
            X = pd.DataFrame(np.atleast_2d(np.asarray(X)),
                             columns=self.data.columns)
            index = X.index

        if self.transforms is not None:
            X = self.transformer_pipeline.transform(X)
        Xs = ((X[self._vae_mean.index] - self._vae_mean) / self._vae_std)
        Xs = Xs.values.astype("float32")
        mu = np.asarray(self._encoder(Xs, training=False)[0], dtype=float)
        mu = (mu - self._z_shift) / self._z_scale
        cols = [f"p_{i}" for i in range(self.latent_dim)]
        return pd.DataFrame(mu, index=index, columns=cols)

    def _reconstruct(self, pvals):
        """Decode latent coordinates into transformed-space observations.

        Parameters
        ----------
        pvals : numpy.ndarray
            A 2-D float array of shape ``(n_realizations, latent_dim)`` of
            calibrated latent coordinates (the space :meth:`encode` returns).

        Returns
        -------
        numpy.ndarray
            A 2-D float array of shape ``(n_realizations, n_obs)`` in
            transformed space (un-standardized, before inverse transforms).
        """
        z = pvals * self._z_scale + self._z_shift
        out = np.asarray(self._decoder(z.astype("float32"), training=False),
                         dtype=float)
        return out * self._vae_std.values + self._vae_mean.values

    def _configure_pst_object(self, pst_new, pst_old=None, observation_data=None,
                              t_d=None):
        """Configure the Pst object and write the encoded latent prior ensemble.

        Parameters
        ----------
        pst_new : Pst
            The control file being prepared.
        pst_old : Pst, optional
            The original control file (for option carry-over).
        observation_data : pandas.DataFrame, optional
            Observation data to merge into ``pst_new``.
        t_d : str, optional
            Template directory in which to write ``latent_prior.jcb``.

        Returns
        -------
        Pst
            The configured control file.
        """
        super()._configure_pst_object(pst_new, pst_old,
                                       observation_data=observation_data, t_d=t_d)

        Z = self.encode(self.data)
        par_names = pst_new.parameter_data.index.tolist()
        assert Z.shape[1] == len(par_names), (
            "latent dim {0} does not match number of parameters {1} in pst_new"
            .format(Z.shape[1], len(par_names)))
        Z.columns = par_names

        pe = ParameterEnsemble(pst_new, Z)
        pe.to_binary(os.path.join(t_d or ".", "latent_prior.jcb"))
        pst_new.pestpp_options["ies_parameter_ensemble"] = "latent_prior.jcb"
        return pst_new


def _train_vae(X, latent_dim, hidden_dims, lr, beta, epochs, batch_size,
               validation_split, early_stopping, patience, random_state,
               verbose):
    """Build and train a variational autoencoder.

    Parameters
    ----------
    X : numpy.ndarray
        Standardized training data of shape ``(n_samples, n_features)``.
    latent_dim : int
        Latent dimensionality.
    hidden_dims : tuple of int
        Encoder hidden layer sizes (reversed for the decoder).
    lr : float
        Adam learning rate.
    beta : float
        Weight on the KL term.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Training batch size.
    validation_split : float
        Validation hold-out fraction.
    early_stopping : bool
        Enable early stopping when a validation split exists.
    patience : int
        Early-stopping patience.
    random_state : int
        Random seed.
    verbose : int
        Keras verbosity.

    Returns
    -------
    tuple
        ``(encoder, decoder, history)`` where encoder and decoder are vanilla
        keras functional models and history is the keras training history.
    """
    tf.random.set_seed(random_state)
    np.random.seed(random_state)
    n_features = X.shape[1]

    inp = tf.keras.Input(shape=(n_features,))
    h = inp
    for units in hidden_dims:
        h = tf.keras.layers.Dense(units, activation="relu")(h)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(h)
    encoder = tf.keras.Model(inp, [z_mean, z_log_var], name="encoder")

    latent_inp = tf.keras.Input(shape=(latent_dim,))
    d = latent_inp
    for units in reversed(hidden_dims):
        d = tf.keras.layers.Dense(units, activation="relu")(d)
    out = tf.keras.layers.Dense(n_features, activation=None)(d)
    decoder = tf.keras.Model(latent_inp, out, name="decoder")

    class _VAETrainer(tf.keras.Model):
        def __init__(self, encoder, decoder, beta):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.beta = beta
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        @property
        def metrics(self):
            return [self.loss_tracker]

        def _compute_loss(self, x, training, sample):
            z_mean, z_log_var = self.encoder(x, training=training)
            if sample:
                eps = tf.random.normal(tf.shape(z_mean))
                z = z_mean + tf.exp(0.5 * z_log_var) * eps
            else:
                z = z_mean
            recon = self.decoder(z, training=training)
            mse = tf.reduce_mean(tf.square(x - recon))
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            return mse + self.beta * kl

        def train_step(self, data):
            x = tf.keras.utils.unpack_x_y_sample_weight(data)[0]
            with tf.GradientTape() as tape:
                loss = self._compute_loss(x, training=True, sample=True)
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def test_step(self, data):
            # evaluate at the posterior mean so val_loss (and early stopping)
            # is deterministic rather than a noisy single draw.
            x = tf.keras.utils.unpack_x_y_sample_weight(data)[0]
            loss = self._compute_loss(x, training=False, sample=False)
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

    trainer = _VAETrainer(encoder, decoder, beta)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(lr))

    callbacks = []
    if early_stopping and validation_split > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True))

    history = trainer.fit(X, validation_split=validation_split, epochs=epochs,
                          batch_size=batch_size, callbacks=callbacks,
                          verbose=verbose, shuffle=True)
    return encoder, decoder, history
