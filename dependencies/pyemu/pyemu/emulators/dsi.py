"""
Data Space Inversion (DSI) emulator implementation.
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import os
import shutil
from pyemu.pst.pst_handler import Pst
from pyemu.en import ObservationEnsemble
from .base import Emulator
from .transformers import AutobotsAssemble, RowWiseMinMaxScaler

class DSI(Emulator):
    """
    Data Space Inversion (DSI) emulator class. Based on DSI as described in Sun &
    Durlofsky (2017) and Sun et al (2017).

    """

    # latent parameter bound magnitude written to the control file; subclasses may narrow it
    _latent_par_bound = 1.0e10

    @property
    def latent_dim(self):
        """Number of latent data-space coordinates.

        Returns
        -------
        int
            The number of columns in the projection matrix.

        Raises
        ------
        Exception
            If the emulator has not been fitted (no projection matrix yet).
        """
        if getattr(self, "pmat", None) is None:
            raise Exception("latent_dim is undefined before fit (no projection matrix yet)")
        return self.pmat.shape[1]

    def __init__(self,
                pst=None,
                data=None,
                transforms=None,
                energy_threshold=1.0,
                rowwise_groups=None,
                rowwise_fit_groups=None,
                feature_range=(-1, 1),
                verbose=False):
        """
        Initialize the DSI emulator.

        If rowwise_groups is provided, training data are row-wise scaled per-group
        before SVD. Predictions are returned in scaled space and then inverse-scaled
        using per-row parameters derived from truth values found in
        pst.observation_data.

        Parameters
        ----------
        pst : Pst, optional
            A Pst object. If provided, the emulator will be initialized with the
            information from the Pst object.
        data : DataFrame or ObservationEnsemble, optional
            An ensemble of simulated observations. If provided, the emulator will
            be initialized with the information from the ensemble.
        transforms : list of dict, optional
            List of transformation specifications. Each dict should have:
            - 'type': str - Type of transformation (e.g.,'log10', 'normal_score').
            - 'columns': list of str,optional - Columns to apply the transformation to. If not supplied, transformation is applied to all columns.
            - Additional kwargs for the transformation (e.g., 'extrapolation' for normal score transform).
            Example:
            transforms = [
                {'type': 'log10', 'columns': ['obs1', 'obs2']},
                {'type': 'normal_score', 'extrapolation': 'quadratic'}
            ]
            Default is None, which means no transformations will be applied.
        energy_threshold : float, optional 
            The energy threshold for the SVD. Default is 1.0, no truncation.
        rowwise_groups : dict, optional
            Dictionary mapping groups to column lists for row-wise scaling.
        rowwise_fit_groups : dict, optional
            Dictionary mapping groups to column lists for fitting row-wise scalers.
        feature_range : tuple, optional
            Feature range for row-wise scaling. Default is (-1, 1).
        verbose : bool, optional
            If True, enable verbose logging. Default is False.
        """

        super().__init__(verbose=verbose)

        if isinstance(pst,Pst):
            self.observation_data = pst.observation_data.copy() if pst is not None else None
        elif isinstance(pst, pd.DataFrame):
            self.observation_data = pst.copy()
        else:
             self.observation_data = None
        if self.observation_data is not None:
            # keep obs names in the lowercase PEST namespace (see Emulator._lowercase_intake)
            if self.observation_data.index.dtype == object:
                self.observation_data.index = self.observation_data.index.str.lower()
            if "obsnme" in self.observation_data.columns:
                self.observation_data["obsnme"] = self.observation_data["obsnme"].str.lower()

        #self.__org_parameter_data = pst.parameter_data.copy() if pst is not None else None
        #self.__org_control_data = pst.control_data.copy() #breaks pickling
        if isinstance(data, ObservationEnsemble):
            data = data._df.copy()
        # set all data to be floats
        data = data.astype(float) if data is not None else None
        #self.__org_data = data.copy() if data is not None else None
        self.data = data.copy() if data is not None else None
        self.energy_threshold = energy_threshold
        if transforms is not None:
            # shared list/dict/'type'/'columns' checks (raises ValueError)
            self._validate_transforms(transforms)
            for t in transforms:
                if 'columns' in t and self.data is not None:
                    missing = [col for col in t['columns'] if col not in self.data.columns]
                    if missing:
                        raise ValueError(f"transform columns not found in the data: {missing}")
                if t.get('type') == 'normal_score' and 'quadratic_extrapolation' in t:
                    if not isinstance(t['quadratic_extrapolation'], bool):
                        raise ValueError("'quadratic_extrapolation' must be a boolean")
        self.transforms = transforms

        # Row-wise scaling config (optional)
        self.rowwise_groups = rowwise_groups
        self.rowwise_fit_groups = rowwise_fit_groups if rowwise_fit_groups is not None else rowwise_groups
        # rowwise group column lists must track the lowercased data columns
        if self.rowwise_groups is not None:
            self.rowwise_groups = {g: [str(c).lower() for c in cols]
                                   for g, cols in self.rowwise_groups.items()}
        if self.rowwise_fit_groups is not None:
            self.rowwise_fit_groups = {g: [str(c).lower() for c in cols]
                                       for g, cols in self.rowwise_fit_groups.items()}
        self.feature_range = feature_range
        self._rowwise_train_scaler = None

        self.fitted = False

        self.data_transformed = self._prepare_training_data()

        # If row-wise scaling is enabled and truth is available, pre-fit truth scaler once
        self._truth_rowwise_scaler = None
        if self.rowwise_groups is not None and self.observation_data is not None:
            # truth values are present: a failure here (e.g. a truth value
            # outside the fitted transform domain) is a data error the user
            # must fix now, not a condition to defer to predict time
            self._truth_rowwise_scaler = self._prefit_truth_rowwise_scaler()
        else:
            self._truth_row_index = 'truth'

    def _prepare_training_data(self):
        """
        Prepare and transform training data for model fitting.
        
        Parameters
        ----------
        self : DSI
            The DSI emulator instance.
            
        Returns
        -------
        tuple
            Processed data ready for model fitting.
        """
        data = self.data
        if data is None:
            raise ValueError("No data stored in the emulator")

        # lowercase all name-keyed state at intake (see Emulator._lowercase_intake)
        self._lowercase_intake()
        data = self.data

        self.logger.statement("applying feature transforms")
        # Always use the base class transformation method for consistency
        if self.transforms is not None:
            self.data_transformed = self._fit_transformer_pipeline(data, self.transforms)
        else:
            # Still need to set up a dummy transformer for inverse operations
            self.transformer_pipeline = AutobotsAssemble(data.copy())
            self.data_transformed = data.copy()

        # 2) Optional row-wise scaling for training
        if self.rowwise_groups is not None:
            self.logger.statement("applying row-wise min-max scaling (training)")
            self._rowwise_train_scaler = RowWiseMinMaxScaler(
                feature_range=self.feature_range,
                groups=self.rowwise_groups,
                fit_groups=self.rowwise_fit_groups
            )
            # Fit on transformed data (e.g. log-transformed) and transform
            self.data_transformed = self._rowwise_train_scaler.fit_transform(self.data_transformed)
    
        return self.data_transformed

    def _get_emulator_parameters(self, pst=None):
        """
        Get the parameters (inputs) for the DSI emulator.
        Returns a DataFrame with columns: parnme, parval1, parlbnd, parubnd, pargp
        """
        if not self.fitted:
            raise Exception("Emulator must be fitted before calling prepare_pestpp")

        # In DSI, parameters are the projections in latent space (p_0, p_1, ...)
        # Number of parameters = dimensionality of projection matrix (columns)
        num_pars = self.latent_dim

        par_names = [f"p_{i}" for i in range(num_pars)]

        df = pd.DataFrame(index=par_names)
        df["parnme"] = par_names
        df["parval1"] = 0.0 # DSI assumes centered parameters (mean 0)
        df["parlbnd"] = -self._latent_par_bound # Effectively unbounded, but good to have ranges
        df["parubnd"] = self._latent_par_bound
        df["pargp"] = "dsi_pars"
        df["partrans"] = "none"
        
        return df

    def _get_emulator_observations(self, pst=None):
        """
        Get the observations (outputs) for the DSI emulator.
        Returns a DataFrame with columns: obsnme, obsval, weight, obgnme
        """
        #if self.observation_data is not None:
        #     df = self.observation_data.copy()
        #     df = df.loc[self.data.columns]  # Ensure order matches training data
        #     return df
        
        # Use columns from data (assuming they represent observations)
        if self.data is not None:
            cols = self.data.columns
            df = pd.DataFrame(index=cols)
            df["obsnme"] = cols
            df["obsval"] = self.data.mean(axis=0) # Use mean as dummy value
            df["weight"] = 0.0
            df["obgnme"] = "obgnme"
            return df
            
        raise Exception("No observation data available to generate instruction files")

    def _build_truth_rowwise_scaler(self, truth_df_transformed):
        """Build a RowWiseMinMaxScaler fitted on provided truth values."""
        scaler = RowWiseMinMaxScaler(
            feature_range=self.feature_range,
            groups=self.rowwise_groups,
            fit_groups=self.rowwise_fit_groups,
        )
        scaler.fit(truth_df_transformed)
        return scaler

    def _prefit_truth_rowwise_scaler(self):
        """Fit a truth-based RowWiseMinMaxScaler once, using pst.observation_data.

        Uses only rowwise_fit_groups columns (intersected with availability) so that
        future/forecast columns are not required in truth.
        """
        if self.rowwise_groups is None:
            return None

        obsdf = self.observation_data
        #obsdf = obsdf.loc[obsdf.weight > 0]
        if obsdf is None or 'obsval' not in obsdf.columns:
            raise ValueError("pst.observation_data with 'obsval' required for truth-based row-wise scaling.")

        # Determine which columns to use from truth: union of fit groups
        fit_cols_union = []
        if self.rowwise_fit_groups is not None:
            for cols in self.rowwise_fit_groups.values():
                fit_cols_union.extend(cols)
        else:
            # this shouldn't happen if groups are set, but just in case
            fit_cols_union = obsdf.index.tolist()

        # Intersect with available columns in training-transformed data and truth index
        available_cols = [c for c in fit_cols_union if c in self.data_transformed.columns and c in obsdf.index]
        if not available_cols:
            raise ValueError("No intersection between rowwise_fit_groups and pst.observation_data.")

        # Build single-row truth DataFrame
        truth_df = obsdf.loc[available_cols, 'obsval'].to_frame().T
        # Use a specific index name we can track. 
        self._truth_row_index = 'truth'
        truth_df.index = [self._truth_row_index]
        
        # Apply feature transforms to truth
        truth_transformed = self.transformer_pipeline.transform(truth_df)

        # Trim fit groups per availability
        fit_groups = {}
        if self.rowwise_fit_groups is not None:
            for g, cols in self.rowwise_fit_groups.items():
                # keep only columns that exist in both truth and training data
                fit_groups[g] = [c for c in cols if c in available_cols]
        
        empty = [g for g, cols in fit_groups.items() if len(cols) == 0]
        if empty:
            self.logger.warn(f"The following row-wise fit groups have no available truth data: {empty}")

        scaler = RowWiseMinMaxScaler(
            feature_range=self.feature_range,
            groups=self.rowwise_groups,
            fit_groups=fit_groups,
        )
        scaler.fit(truth_transformed)
        return scaler
        
    def compute_projection_matrix(self, energy_threshold=None):
        """
        Compute the projection matrix using SVD.
        
        Parameters
        ----------
        energy_threshold : float, optional
            Energy threshold for truncation. Default is None, which uses the threshold from initialization.
            
        Returns
        -------
        None
        """
        self.logger.statement("normalizing data")
        # normalize the data by subtracting the mean and dividing by the standard deviation
        X = self.data_transformed.copy()
        deviations = X - X.mean()
        z = deviations / np.sqrt(float(X.shape[0] - 1))
        if isinstance(z, pd.DataFrame):
            z = z.values

        self.logger.statement("undertaking SVD")
        u, s, v = np.linalg.svd(z, full_matrices=False)
        org_num_components = len(s)
        us = np.dot(v.T, np.diag(s))
        if energy_threshold is None:
            energy_threshold = self.energy_threshold
        if energy_threshold<1.0:
            self.logger.statement("applying energy truncation")
            # compute the cumulative energy of the singular values
            cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
            # find the number of components needed to reach the energy threshold
            num_components = np.argmax(cumulative_energy >= energy_threshold) + 1
            # keep only the first num_components singular values and vectors
            us = us[:, :num_components]
            s = s[:num_components]
            u = u[:, :num_components]
            #print(f"Truncated from {len(s)} to {num_components} components while retaining {energy_threshold*100:.1f}% of variance")
            self.logger.statement(f"truncated from {org_num_components} to {num_components} components while retaining {energy_threshold*100:.1f}% of variance")
            if num_components<=1:
                #print(f"Warning: only {num_components} component retained, you may need to check the data")
                self.logger.warning(f"only {num_components} component retained, you may need to check the data")
        self.logger.statement("calculating us matrix")
        
        # store components needed for forward run
        # store mean vector
        self.ovals = self.data_transformed.mean(axis=0)
        # store proj matrix and singular values
        self.pmat = us
        self.s = s
        return
    
    def fit(self):
        """
        Fit the emulator to training data.
        
        Parameters
        ----------
        self : DSI
            The DSI emulator instance.
            
        Returns
        -------
        self : DSI
            The fitted emulator.
        """
        
        if self.data_transformed is None:
            self.logger.statement("transforming training data")
            self.data_transformed = self._prepare_training_data()

        # Compute projection matrix
        self.compute_projection_matrix()
        self.fitted = True
        return self
    
    def _reconstruct(self, pvals):
        """Reconstruct transformed-space observations from latent coordinates.

        Parameters
        ----------
        pvals : numpy.ndarray
            A 2-D float array of shape (n_realizations, latent_dim).

        Returns
        -------
        numpy.ndarray
            A 2-D float array of shape (n_realizations, n_obs) holding the
            reconstructed values in transformed space, that is before row-wise
            inverse scaling and before inverse feature transforms.

        Notes
        -----
        Subclasses override this to swap the linear map for another decoder.
        """
        ovals = self.ovals.values if hasattr(self.ovals, 'values') else self.ovals
        return (ovals[:, np.newaxis] + np.dot(self.pmat, pvals.T)).T

    def predict(self, pvals, pst: Pst = None):
        """
        Generate predictions from the emulator.
        
        Parameters
        ----------
        pvals : numpy.ndarray or pandas.Series
            Parameter values for prediction.
        pst : Pst, optional
            If provided (or if self.observation_data exists), used to obtain
            truth values for inverse row-wise scaling (if enabled).

        Returns
        -------
        pandas.Series
            Predicted observation values.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before prediction")
            
        if self.transforms is not None and (not hasattr(self, 'transformer_pipeline') or self.transformer_pipeline is None):
            raise ValueError("Emulator must be fitted and have valid transformations before prediction")
        
        # Handle different input types and convert to numpy array
        if isinstance(pvals, pd.Series):
            pvals = pvals.values.reshape(1, -1)  # Single realization
            single_realization = True
        elif isinstance(pvals, pd.DataFrame):
            realization_names = pvals.index.tolist()
            pvals = pvals.values  # Multiple realizations
            single_realization = False
        else:
            pvals = np.asarray(pvals)
            if pvals.ndim == 1:
                pvals = pvals.reshape(1, -1)  # Single realization
                single_realization = True
            else:
                realization_names = [f"real_{i}" for i in range(pvals.shape[0])]
                single_realization = False
        
        # Validate dimensions
        if pvals.shape[1] != self.latent_dim:
            raise ValueError(f"pvals must have {self.latent_dim} parameters, got {pvals.shape[1]}")

        # Reconstruct transformed-space observations (n_realizations x n_obs)
        sim_vals_arr = self._reconstruct(pvals)

        # Determine column names (observations). For DSI, ovals.index is exactly
        # data_transformed.columns; DSIAE has no ovals, so resolve from the data.
        obs_names = self.data_transformed.columns

        # Convert to pandas structure (n_realizations x n_obs)
        if single_realization:
            # Return Series for single realization
            sim_vals = pd.Series(sim_vals_arr[0], index=obs_names)
            sim_vals.index.name = 'obsnme'
            sim_vals.name = "obsval"

            # Temporary DataFrame for unified processing
            sim_df = sim_vals.to_frame().T
            sim_df.index = [getattr(self, '_truth_row_index', 'truth')] # mimic truth index for 1-row case
        else:
            # Return DataFrame for multiple realizations
            sim_df = pd.DataFrame(sim_vals_arr,
                                columns=obs_names,
                                index=realization_names,
                                )
            sim_df.index.name = 'realization'

        # --- Row-wise Inverse Scaling (Logic from dsi copy.py adapted for broadcasting) ---
        if self.rowwise_groups is not None:
            # use the pre-fitted truth scaler, else fit once from truth
            # values provided at predict time
            truth_scaler = self._truth_rowwise_scaler
            if truth_scaler is None:
                if pst is not None:
                    self.observation_data = pst.observation_data.copy()
                if self.observation_data is None:
                    raise ValueError(
                        "row-wise scaling requires truth values to "
                        "inverse-scale predictions; provide them via the "
                        "pst argument")
                truth_scaler = self._prefit_truth_rowwise_scaler()
                self._truth_rowwise_scaler = truth_scaler

            if truth_scaler is not None:
                 # Apply inverse row-wise scaling efficiently
                 # Truth scaler has params for ONE row (the truth). We apply this to ALL rows.
                 f_min, f_max = self.feature_range
                 result_df = sim_df.copy() # Start with current (scaled) predictions
                 
                 for group_name, group_cols in self.rowwise_groups.items():
                    valid_cols = [col for col in group_cols if col in sim_df.columns]
                    if not valid_cols:
                        continue
                    
                    # Get the min and max for the TRUTH row (fitted in truth_scaler)
                    # truth_scaler.row_params is {group: (min_series, max_series)}
                    # These series have index ['truth'] (or whatever _truth_row_index is)
                    row_min_series, row_max_series = truth_scaler.row_params[group_name]
                    
                    # Extract scalar values from the series (since there's only 1 truth)
                    t_min = row_min_series.iloc[0]
                    t_max = row_max_series.iloc[0]
                    
                    t_range = t_max - t_min
                    if t_range == 0: t_range = 1.0

                    # Get data for this group (n_samples, n_cols)
                    group_data = sim_df[valid_cols]
                    
                    # Inverse formula: x_orig = (x_scaled - f_min)/(f_max - f_min) * (t_max - t_min) + t_min
                    # Broadcast: (group_data - scalar) / scalar * scalar + scalar
                    group_std = (group_data - f_min) / (f_max - f_min)
                    
                    # Apply truth range to all rows
                    result_df[valid_cols] = group_std * t_range + t_min
                 
                 sim_df = result_df

        # --- Feature Inverse Transforms ---
        # Apply inverse transforms if needed
        if self.transforms is not None:
            pipeline = self.transformer_pipeline
            # Apply inverse transform to each realization
            sim_df = pipeline.inverse(sim_df)

        self.sim_vals = sim_df if not single_realization else sim_df.iloc[0]
        return self.sim_vals
    
    def check_for_pdc(self):
        """Check for Prior data conflict."""
        #TODO
        return

    def _write_forward_run_script(self, filename, emu_file, input_file, output_file, class_name, pst_name=None):
        """Generates the python script that PEST++ runs for DSI."""
        from pyemu.utils.helpers import dsi_file_forward_run, dsi_runstore_forward_run, dsi_forward_run

        use_runstor = getattr(self, "_use_runstor", False)
        if use_runstor:
            target_func = "dsi_runstore_forward_run"
            call_args = f"pst_name='{pst_name}'" if pst_name is not None else ""
        else:
            target_func = "dsi_file_forward_run"
            call_args = f"'{emu_file}', '{input_file}', '{output_file}'"

        self._write_forward_run_script_body(
            filename,
            [dsi_forward_run, dsi_file_forward_run, dsi_runstore_forward_run],
            target_func,
            call_args,
        )

    def prepare_pestpp(self, t_d, observation_data=None, use_runstor=False, pst=None, verbose=False):
        """
        Prepare PEST++ interface for DSI.
        Overrides base method to handle specific DSI arguments like use_runstor
        """
        self._use_runstor = use_runstor

        # Maintain backward compatibility with explicit observation_data argument
        if observation_data is not None:
             if isinstance(observation_data, pd.DataFrame):
                 self.observation_data = observation_data
             # If passed, we update our internal reference so the hook uses it
        
        # 1. Call Generic Base Logic
        # This creates files and standard Pst object
        pst_obj = super().prepare_pestpp(t_d, pst=pst, verbose=verbose, 
                                         tpl_filename="dsi_pars.csv.tpl",
                                         input_filename="dsi_pars.csv",
                                         ins_filename="dsi_sim_vals.csv.ins",
                                         output_filename="dsi_sim_vals.csv",
                                         emu_filename="dsi.pickle",
                                         observation_data=self.observation_data,
                                         use_runstor=self._use_runstor)
        
        with open(os.path.join(t_d,"dsi.unc"),'w') as f:
            f.write("START STANDARD_DEVIATION\n")
            for p in pst_obj.par_names:
                f.write("{0} 1.0\n".format(p))
            f.write("END STANDARD_DEVIATION")
        pst_obj.pestpp_options['parcov'] = "dsi.unc"

        # Write the control file so the prepared template dir is complete.
        # Callers may further modify the returned pst and write it again; their
        # write simply overwrites this one.
        pst_obj.write(os.path.join(t_d, "dsi.pst"), version=2)
        return pst_obj
