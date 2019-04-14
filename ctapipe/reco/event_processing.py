import numpy as np
import astropy.units as u
from copy import deepcopy

from abc import ABC, abstractmethod

import sklearn
import sklearn.ensemble
import pandas as pd
import joblib

from ctapipe.image import tailcuts_clean, hillas_parameters
from ctapipe.io import EventSourceFactory
from ctapipe.io.containers import ReconstructedContainer, ReconstructedEnergyContainer, ParticleClassificationContainer
from ctapipe.reco import EnergyRegressor
from ctapipe.reco.HillasReconstructor import TooFewTelescopesException


class HillasFeatureSelector(ABC):
    """
    The base class that handles the event Hillas parameter extraction
    for future use with the random forest energy and classification pipelines.
    """

    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        """
        Constructor. Stores the settings that will be used during the parameter
        extraction.

        Parameters
        ----------
        hillas_params_to_use: list
            A list of Hillas parameter names that should be extracted.
        hillas_reco_params_to_use: list
            A list of the Hillas "stereo" parameters (after HillasReconstructor),
            that should also be extracted.
        telescopes: list
            List of telescope identifiers. Only events triggering these will be processed.
        """

        self.hillas_params_to_use = hillas_params_to_use
        self.hillas_reco_params_to_use = hillas_reco_params_to_use
        self.telescopes = telescopes

        n_features_per_telescope = len(hillas_params_to_use) + len(hillas_reco_params_to_use)
        self.n_features = n_features_per_telescope * len(telescopes)

    @staticmethod
    def _get_param_value(param):
        """
        An internal method that extracts the parameter value from both
        float and Quantity instances.

        Parameters
        ----------
        param: float or astropy.unit.Quantity
            A parameter whos value should be extracted.

        Returns
        -------
        float:
            An extracted value. For float the param itself is returned,
            for Quantity the Quantity.value is taken.

        """

        if isinstance(param, u.Quantity):
            return param.value
        else:
            return param

    @abstractmethod
    def fill_event(self, event, hillas_reco_result, target):
        """
        A dummy function to process an event.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        hillas_reco_result: ReconstructedShowerContainer
            A container with the computed shower direction properties.
        target: float
            A target variable for future regression/classification model.

        Returns
        -------

        """

        pass


class EventFeatureSelector(HillasFeatureSelector):
    """
    A class that performs the selects event features for further reconstruction with
    ctapipe.reco.RegressorClassifierBase.
    """

    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        """
        Constructor. Stores the settings that will be used during the parameter
        extraction.

        Parameters
        ----------
        hillas_params_to_use: list
            A list of Hillas parameter names that should be extracted.
        hillas_reco_params_to_use: list
            A list of the Hillas "stereo" parameters (after HillasReconstructor),
            that should also be extracted.
        telescopes: list
            List of telescope identifiers. Only events triggering these will be processed.
        """

        super(EventFeatureSelector, self).__init__(hillas_params_to_use, hillas_reco_params_to_use, telescopes)

        self.events = []
        self.event_targets = []

    def fill_event(self, event, hillas_reco_result, target=None):
        """
        This method fills the event features to the feature list.
        Optionally it can add a "target" value, which can be used
        to check the accuracy of the reconstruction.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        hillas_reco_result: ReconstructedShowerContainer
            A container with the computed shower direction properties.
        target: float
            A target variable for future regression/classification model.

        Returns
        -------

        """

        event_record = dict()
        for tel_id in self.telescopes:
            feature_entry = []

            for param_name in self.hillas_params_to_use:
                param = event.dl1.tel[tel_id].hillas_params[param_name]

                feature_entry.append(self._get_param_value(param))

            for param_name in self.hillas_reco_params_to_use:
                param = hillas_reco_result[param_name]

                feature_entry.append(self._get_param_value(param))

            if np.all(np.isfinite(feature_entry)):
                event_record[tel_id] = [feature_entry]

        self.events.append(event_record)
        self.event_targets.append(target)


class EventFeatureTargetSelector(HillasFeatureSelector):
    """
    A class that performs the selects event features for further training of the
    ctapipe.reco.RegressorClassifierBase model.
    """

    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        """
        Constructor. Stores the settings that will be used during the parameter
        extraction.

        Parameters
        ----------
        hillas_params_to_use: list
            A list of Hillas parameter names that should be extracted.
        hillas_reco_params_to_use: list
            A list of the Hillas "stereo" parameters (after HillasReconstructor),
            that should also be extracted.
        telescopes: list
            List of telescope identifiers. Only events triggering these will be processed.
        """

        super(EventFeatureTargetSelector, self).__init__(hillas_params_to_use, hillas_reco_params_to_use, telescopes)

        self.features = dict()
        self.targets = dict()
        self.events = []

        for tel_id in self.telescopes:
            self.features[tel_id] = []
            self.targets[tel_id] = []

    def fill_event(self, event, hillas_reco_result, target):
        """
        This method fills the event features to the feature list;
        "target" values are added to their own "target" list.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        hillas_reco_result: ReconstructedShowerContainer
            A container with the computed shower direction properties.
        target: float
            A target variable for future regression/classification model.

        Returns
        -------

        """

        event_record = dict()
        for tel_id in self.telescopes:
            feature_entry = []

            for param_name in self.hillas_params_to_use:
                param = event.dl1.tel[tel_id].hillas_params[param_name]

                feature_entry.append(self._get_param_value(param))

            for param_name in self.hillas_reco_params_to_use:
                param = hillas_reco_result[param_name]

                feature_entry.append(self._get_param_value(param))

            if np.all(np.isfinite(feature_entry)):
                self.features[tel_id].append(feature_entry)
                self.targets[tel_id].append(self._get_param_value(target))

                event_record[tel_id] = [feature_entry]
        self.events.append(event_record)


class EventProcessor:
    """
    This class is meant to represents the DL0->DL2 analysis pipeline.
    It handles event loading, Hillas parameter estimation and storage
    (DL0->DL1), stereo (with >=2 telescopes) event direction/impact etc.
    reconstruction and RF energy estimation.
    """

    def __init__(self, calibrator, hillas_reconstructor, min_survived_pixels=10):
        """
        Constructor. Sets the calibration / Hillas processing workers.

        Parameters
        ----------
        calibrator: ctapipe.calib.CameraCalibrator
            A desired camera calibrator instance.
        hillas_reconstructor: ctapipe.reco.HillasReconstructor
            A "stereo" (with >=2 telescopes) Hillas reconstructor instance that
            will be used to determine the event direction/impact etc.
        min_survived_pixels: int, optional
            Minimal number of pixels in the shower image, that should survive
            image cleaning. Hillas parameters are not computed for events falling
            below this threshold.
            Defaults to 10.
        """

        self.calibrator = calibrator
        self.hillas_reconstructor = hillas_reconstructor
        self.min_survived_pixels = min_survived_pixels

        self.events = []
        self.reconstruction_results = []

    def _dl1_process(self, event):
        """
        Internal method that performs DL0->DL1 event processing.
        This involves image cleaning and Hillas parameter calculation.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL0 event data.

        Returns
        -------
        DataContainer:
            Event with computed Hillas parameters.

        """

        tels_with_data = list(event.r1.tels_with_data)

        for tel_id in tels_with_data:
            subarray = event.inst.subarray
            camera = subarray.tel[tel_id].camera

            self.calibrator.calibrate(event)

            event_image = event.dl1.tel[tel_id].image[1]

            mask = tailcuts_clean(camera, event_image,
                                  picture_thresh=6, boundary_thresh=6)
            event_image_cleaned = event_image.copy()
            event_image_cleaned[~mask] = 0

            n_survived_pixels = event_image_cleaned[mask].size

            # Calculate hillas parameters
            # It fails for empty images, so we apply a cut on the number
            # of survived pixels
            if n_survived_pixels > self.min_survived_pixels:
                try:
                    event.dl1.tel[tel_id].hillas_params = hillas_parameters(camera, event_image_cleaned)
                except:
                    print('Failed')
                    pass

        return event

    def _update_event_direction(self, event, reco_container):
        """
        Internal method used to compute the shower direction/impact etc. from
        intersection of the per-telescope image planes (from Hillas parameters)
        and store them to the provided reconstruction container.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        reco_container: ReconstructedContainer
            A container that will hold the computed shower properties.

        Returns
        -------
        ReconstructedContainer:
            Updated shower reconstruction container.

        """

        # Performing a geometrical direction reconstruction
        try:
            reco_container.shower['hillas'] = self.hillas_reconstructor.predict_from_dl1(event)
        except TooFewTelescopesException:
            # No reconstruction possible. Resetting to defaults
            reco_container.shower['hillas'].reset()

        return reco_container

    def _update_event_energy(self, event, reco_container):
        """
        Internal method used to compute the shower energy from a pre-trained RF
        and store it to the provided reconstruction container.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        reco_container: ReconstructedContainer
            A container that will hold the computed shower energy.

        Returns
        -------
        ReconstructedContainer:
            Updated shower reconstruction container.

        """

        return reco_container

    def _update_event_classification(self, event, reco_container):
        """
        Internal method used to compute the classify the using the pre-trained RF
        and store it to the provided reconstruction container.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        reco_container: ReconstructedContainer
            A container that will hold the computed shower class.

        Returns
        -------
        ReconstructedContainer:
            Updated shower reconstruction container.

        """

        return reco_container

    def _load_events(self, file_name, append_to_existing_events):
        """
        Internal method that takes care of the event loading from the specified file
        and DL1 processing. The DL1 events are also stored in the self.events list
        for future usage; their R0/R1/DL0 containers are reset to save memory.

        Parameters
        ----------
        file_name: str
            A file name from which to read the events.
        append_to_existing_events: bool
            Defines whether the previously filled event list should be cleared
            or if the new events should be appended to it.

        Returns
        -------

        """

        with EventSourceFactory.produce(input_url=file_name) as event_factory:
            event_generator = event_factory._generator()

            if not append_to_existing_events:
                self.events = []

            # Running the parameter computation over the event list
            for event in event_generator:
                event = self._dl1_process(event)

                event.r0.reset()
                event.r1.reset()
                event.dl0.reset()

                self.events.append(deepcopy(event))

    def process_file(self, file_name,
                     append_to_existing_events=True,
                     do_direction_reconstruction=True,
                     do_energy_reconstruction=True,
                     do_classification=True):

        """
        This method represents the file processing pipeline: data loading,
        DL0->DL1 processing and the subsequent direction/energy/classification.

        Parameters
        ----------
        file_name: str
            A file name from which to read the events.
        append_to_existing_events: bool
            Defines whether the previously filled event list should be cleared
            or if the new events should be appended to it.
        do_direction_reconstruction: bool, optional
            Sets whether the direction reconstruction should be performed.
            Defaults to True.
        do_energy_reconstruction: bool, optional
            Sets whether the energy reconstruction should be performed.
            Requires a trained energy random forest.
            Defaults to False. NOT YES IMPLEMENTED
        do_classification: bool, optional
            Sets whether the event classification should be performed.
            Requires a trained classifier random forest.
            Defaults to False. NOT YES IMPLEMENTED

        Returns
        -------

        """

        self._load_events(file_name, append_to_existing_events=append_to_existing_events)

        self.reconstruction_results = []

        # Running the parameter computation over the event list
        for event in self.events:
            # Run the event properties reconstruction
            reco_results = ReconstructedContainer()

            if do_direction_reconstruction:
                reco_results = self._update_event_direction(event, reco_results)

            if do_energy_reconstruction:
                reco_results = self._update_event_energy(event, reco_results)

            if do_classification:
                reco_results = self._update_event_classification(event, reco_results)

            self.reconstruction_results.append(reco_results)


class EnergyEstimator:
    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        """
        Constructor. Stores the settings that will be used during the parameter
        extraction.

        Parameters
        ----------
        hillas_params_to_use: list
            A list of Hillas parameter names that should be extracted.
        hillas_reco_params_to_use: list
            A list of the Hillas "stereo" parameters (after HillasReconstructor),
            that should also be extracted.
        telescopes: list
            List of telescope identifiers. Only events triggering these will be processed.
        """

        self.hillas_params_to_use = hillas_params_to_use
        self.hillas_reco_params_to_use = hillas_reco_params_to_use
        self.telescopes = telescopes

        n_features_per_telescope = len(hillas_params_to_use) + len(hillas_reco_params_to_use)
        self.n_features = n_features_per_telescope * len(telescopes)

        self.train_features = dict()
        self.train_targets = dict()
        self.train_events = []

        for tel_id in self.telescopes:
            self.train_features[tel_id] = []
            self.train_targets[tel_id] = []

        self.regressor = EnergyRegressor(cam_id_list=self.telescopes)

    @staticmethod
    def _get_param_value(param):
        """
        An internal method that extracts the parameter value from both
        float and Quantity instances.

        Parameters
        ----------
        param: float or astropy.unit.Quantity
            A parameter whose value should be extracted.

        Returns
        -------
        float:
            An extracted value. For float the param itself is returned,
            for Quantity the Quantity.value is taken.

        """

        if isinstance(param, u.Quantity):
            return param.value
        else:
            return param

    def add_train_event(self, event, reco_result):
        """
        This method fills the event features to the feature list;
        "target" values are added to their own "target" list.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        reco_result: ReconstructedContainer
            A container with the already reconstructed event properties.
            Its 'shower' part must have the 'hillas' key.

        Returns
        -------

        """

        event_record = dict()

        for tel_id in self.telescopes:
            feature_entry = []

            for param_name in self.hillas_params_to_use:
                param = event.dl1.tel[tel_id].hillas_params[param_name]

                feature_entry.append(self._get_param_value(param))

            for param_name in self.hillas_reco_params_to_use:
                param = reco_result.shower['hillas'][param_name]

                feature_entry.append(self._get_param_value(param))

            if np.all(np.isfinite(feature_entry)):
                event_energy = event.mc.energy.to(u.TeV).value
                self.train_features[tel_id].append(feature_entry)
                self.train_targets[tel_id].append(np.log10(event_energy))

                event_record[tel_id] = [feature_entry]

        self.train_events.append(event_record)

    def process_event(self, event, reco_result):
        event_record = dict()
        for tel_id in self.telescopes:
            feature_entry = []

            for param_name in self.hillas_params_to_use:
                param = event.dl1.tel[tel_id].hillas_params[param_name]

                feature_entry.append(self._get_param_value(param))

            for param_name in self.hillas_reco_params_to_use:
                param = reco_result.shower['hillas'][param_name]

                feature_entry.append(self._get_param_value(param))

            if np.all(np.isfinite(feature_entry)):
                event_record[tel_id] = [feature_entry]

        predicted_energy_dict = self.regressor.predict_by_event([event_record])

        reconstructed_energy = 10**predicted_energy_dict['mean'].value * u.TeV
        std = predicted_energy_dict['std'].value
        rel_uncert = 0.5 * (10 ** std - 1 / 10 ** std)

        energy_container = ReconstructedEnergyContainer()
        energy_container.energy = reconstructed_energy
        energy_container.energy_uncert = energy_container.energy * rel_uncert
        energy_container.is_valid = True
        energy_container.tel_ids = list(event_record.keys())

        return energy_container

    def fit_model(self):
        _ = self.regressor.fit(self.train_features, self.train_targets)

    def save_model(self):
        pass

    def load_model(self):
        pass


class EnergyEstimatorPandas:
    """
    This class trains/applies the random forest regressor for event energy,
    using as the input Hillas and stereo parameters, stored in a Pandas data frame.
    It trains a separate regressor for each telescope. Further another "consolidating"
    regressor is applied to combine the per-telescope predictions.
    """

    def __init__(self, feature_names, target_name):
        """
        Constructor. Gets basic settings.

        Parameters
        ----------
        feature_names: tuple
            Feature names (str type) to be used by the regressor. Must correspond to the
            columns of the data frames that will be processed.
        target_name: str
            The target variable for the regressor. Likely this should be 'log10_true_energy'.
        """

        self.feature_names = feature_names
        self.target_name = target_name

        self.telescope_regressors = dict()
        self.consolidating_regressor = None

    def fit(self, shower_data):
        """
        Fits the regressor model.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.

        Returns
        -------
        None

        """

        self.train_per_telescope_rf(shower_data)

        shower_data_with_energy = self.apply_per_telescope_rf(shower_data, output_prefix='log10_est_energy')

        energy_feature_cols = list(filter(lambda s: 'log10_est_energy_' in s, shower_data_with_energy.columns))
        energy_target_col = 'log10_true_energy'

        features = shower_data_with_energy[energy_feature_cols]
        features = features.fillna(0).groupby(['obs_id', 'event_id']).sum()
        features = features.values

        target = shower_data_with_energy[energy_target_col].groupby(['obs_id', 'event_id']).mean().values

        self.consolidating_regressor = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
        self.consolidating_regressor.fit(features, target)

    def predict(self, shower_data, output_prefix):
        """
        Applies the trained regressor to the data.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.
        output_prefix: str, optional
            Prefix to the new data frame columns, that will host the regressors
            predictions. Columns will have names "{base_output_name}_{tel_id}".
            Defaults to 'est_energy'.

        Returns
        -------
        pandas.DataFrame:
            Updated data frame with the computed shower energies.

        """

        shower_data_with_energy = self.apply_per_telescope_rf(shower_data, output_prefix='log10_est_energy')

        energy_feature_cols = list(filter(lambda s: output_prefix + '_' in s, shower_data_with_energy.columns))

        features = shower_data_with_energy[energy_feature_cols]
        features = features.fillna(0).groupby(['obs_id', 'event_id']).sum()
        index = features.index
        features = features.values

        predictions = self.consolidating_regressor.predict(features)

        est_energy_series = pd.Series(np.repeat(np.nan, len(features)),
                                      name='log10_est_energy', dtype=np.float32,
                                      index=index)

        est_energy_series.loc[:] = predictions

        est_energy_series = est_energy_series.reindex(shower_data_with_energy.index)

        shower_data_with_energy = shower_data_with_energy.join(est_energy_series)

        return shower_data_with_energy

    def _get_per_telescope_features(self, shower_data):
        """
        Extracts the shower features specific to each telescope of
        the available ones.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.

        Returns
        -------
        output: dict
            output['feature']: dict
                Shower features for each telescope (keys - telescope IDs).
            output['targets']: dict
                Regressor targets for each telescope (keys - telescope IDs).

        """

        tel_ids = shower_data.index.levels[2]

        output = dict()
        output['features'] = dict()
        output['targets'] = dict()
        output['event_ids'] = dict()

        for tel_id in tel_ids:
            selected_columns = self.feature_names + (self.target_name,)

            this_telescope = shower_data.loc[(slice(None), slice(None), tel_id), selected_columns]
            this_telescope = this_telescope.dropna()

            output['features'][tel_id] = this_telescope[list(self.feature_names)].values
            output['targets'][tel_id] = this_telescope[self.target_name].values

        return output

    def train_per_telescope_rf(self, shower_data):
        """
        Trains the energy regressors for each of the available telescopes.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.

        Returns
        -------
        None

        """

        input_data = self._get_per_telescope_features(shower_data)

        tel_ids = input_data['features'].keys()

        self.telescope_regressors = dict()

        for tel_id in tel_ids:
            x_train = input_data['features'][tel_id]
            y_train = input_data['targets'][tel_id]

            regressor = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
            regressor.fit(x_train, y_train)

            self.telescope_regressors[tel_id] = regressor

    def apply_per_telescope_rf(self, shower_data, output_prefix='est_energy'):
        """
        Applies the regressors, trained per each telescope.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.
        output_prefix: str, optional
            Prefix to the new data frame columns, that will host the regressors
            predictions. Columns will have names "{base_output_name}_{tel_id}".
            Defaults to 'est_energy'.

        Returns
        -------
        pandas.DataFrame:
            Updated data frame with the computed shower energies.

        """

        input_data = self._get_per_telescope_features(shower_data)

        tel_ids = input_data['features'].keys()

        shower_data_with_energy = shower_data.copy()

        for tel_id in tel_ids:
            selected_columns = self.feature_names + (self.target_name,)

            this_telescope = shower_data.loc[(slice(None), slice(None), tel_id), selected_columns]
            this_telescope = this_telescope.dropna()
            index = this_telescope.index.remove_unused_levels()

            data_series_name = "{:s}_{:d}".format(output_prefix, tel_id)
            est_energy_series = pd.Series(np.repeat(np.nan, len(this_telescope)),
                                          name=data_series_name, dtype=np.float32,
                                          index=index)

            predictions = self.telescope_regressors[tel_id].predict(input_data['features'][tel_id])

            est_energy_series.loc[(slice(None), slice(None), tel_id)] = predictions

            est_energy_series = est_energy_series.reindex(shower_data.index)

            shower_data_with_energy = shower_data_with_energy.assign(**{data_series_name: est_energy_series.values})

        return shower_data_with_energy

    def save(self, file_name):
        """
        Saves trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        output = dict()
        output['feature_names'] = self.feature_names
        output['target_name'] = self.target_name
        output['telescope_regressors'] = self.telescope_regressors
        output['consolidating_regressor'] = self.consolidating_regressor

        joblib.dump(output, file_name)

    def load(self, file_name):
        """
        Loads pre-trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        data = joblib.load(file_name)

        self.feature_names = data['feature_names'] 
        self.target_name = data['target_name'] 
        self.telescope_regressors = data['telescope_regressors'] 
        self.consolidating_regressor = data['consolidating_regressor'] 


class DirectionEstimatorPandas:
    """
    This class trains/applies the random forest regressor for event energy,
    using as the input Hillas and stereo parameters, stored in a Pandas data frame.
    It trains a separate regressor for each telescope. Further another "consolidating"
    regressor is applied to combine the per-telescope predictions.
    """

    def __init__(self, feature_names, target_name, **rf_settings):
        """
        Constructor. Gets basic settings.

        Parameters
        ----------
        feature_names: list
            Feature names (str type) to be used by the regressor. Must correspond to the
            columns of the data frames that will be processed.
        target_name: str
            The target variable for the regressor. Likely this should be 'log10_true_energy'.
        """

        self.feature_names = feature_names
        self.target_name = target_name

        self.rf_settings = rf_settings

        self.telescope_regressors = dict()
        self.consolidating_regressor = None

    def fit(self, shower_data):
        """
        Fits the regressor model.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.

        Returns
        -------
        None

        """

        self.train_per_telescope_rf(shower_data, **self.rf_settings)

        # shower_data_with_energy = self.apply_per_telescope_rf(shower_data, output_prefix='log10_est_energy')
        #
        # energy_feature_cols = list(filter(lambda s: 'log10_est_energy_' in s, shower_data_with_energy.columns))
        # energy_target_col = 'log10_true_energy'
        #
        # features = shower_data_with_energy[energy_feature_cols]
        # features = features.fillna(0).groupby(['obs_id', 'event_id']).sum()
        # features = features.values
        #
        # target = shower_data_with_energy[energy_target_col].groupby(['obs_id', 'event_id']).mean().values
        #
        # self.consolidating_regressor = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
        # self.consolidating_regressor.fit(features, target)

    def predict(self, shower_data, output_prefix):
        """
        Applies the trained regressor to the data.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.
        output_prefix: str, optional
            Prefix to the new data frame columns, that will host the regressors
            predictions. Columns will have names "{base_output_name}_{tel_id}".
            Defaults to 'est_energy'.

        Returns
        -------
        pandas.DataFrame:
            Updated data frame with the computed shower energies.

        """

        shower_data_with_energy = self.apply_per_telescope_rf(shower_data, output_prefix='log10_est_energy')

        # energy_feature_cols = list(filter(lambda s: output_prefix + '_' in s, shower_data_with_energy.columns))
        #
        # features = shower_data_with_energy[energy_feature_cols]
        # features = features.fillna(0).groupby(['obs_id', 'event_id']).sum()
        # index = features.index
        # features = features.values
        #
        # predictions = self.consolidating_regressor.predict(features)
        #
        # est_energy_series = pd.Series(np.repeat(np.nan, len(features)),
        #                               name='log10_est_energy', dtype=np.float32,
        #                               index=index)
        #
        # est_energy_series.loc[:] = predictions
        #
        # est_energy_series = est_energy_series.reindex(shower_data_with_energy.index)
        #
        # shower_data_with_energy = shower_data_with_energy.join(est_energy_series)
        #
        # return shower_data_with_energy

    def _get_per_telescope_features(self, shower_data):
        """
        Extracts the shower features specific to each telescope of
        the available ones.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.

        Returns
        -------
        output: dict
            output['feature']: dict
                Shower features for each telescope (keys - telescope IDs).
            output['targets']: dict
                Regressor targets for each telescope (keys - telescope IDs).

        """

        tel_ids = shower_data.index.levels[2]

        output = dict()
        output['features'] = dict()
        output['targets'] = dict()
        output['event_ids'] = dict()

        for tel_id in tel_ids:
            selected_columns = self.feature_names + (self.target_name,)

            this_telescope = shower_data.loc[(slice(None), slice(None), tel_id), selected_columns]
            this_telescope = this_telescope.dropna()

            output['features'][tel_id] = this_telescope[list(self.feature_names)].values
            output['targets'][tel_id] = this_telescope[self.target_name].values

        return output

    def train_per_telescope_rf(self, shower_data, **rf_settings):
        """
        Trains the energy regressors for each of the available telescopes.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.

        Returns
        -------
        None

        """

        idx = pd.IndexSlice

        tel_ids = shower_data.index.levels[2]

        self.telescope_regressors = dict()

        for tel_id in tel_ids:
            print(f'Training telescope {tel_id}...')

            input_data = shower_data.loc[idx[:, :, tel_id], self.feature_names + [self.target_name]]
            input_data.dropna(inplace=True)

            x_train = input_data[self.feature_names].values
            y_train = input_data[self.target_name].values

            regressor = sklearn.ensemble.RandomForestRegressor(**rf_settings)
            regressor.fit(x_train, y_train)

            self.telescope_regressors[tel_id] = regressor

    def apply_per_telescope_rf(self, shower_data, output_prefix='rf_disp'):
        """
        Applies the regressors, trained per each telescope.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.
        output_prefix: str, optional
            Prefix to the new data frame columns, that will host the regressors
            predictions. Columns will have names "{base_output_name}_{tel_id}".
            Defaults to 'est_energy'.

        Returns
        -------
        pandas.DataFrame:
            Updated data frame with the computed shower energies.

        """

        idx = pd.IndexSlice

        tel_ids = shower_data.index.levels[2]

        shower_data_with_direction = shower_data.copy()

        disp_az = pd.Series(np.repeat(np.nan, len(shower_data)),
                            name=output_prefix, dtype=np.float32,
                            index=shower_data.index)

        for tel_id in tel_ids:
            selected_columns = self.feature_names + [self.target_name]

            input_data = shower_data.loc[idx[:, :, tel_id], selected_columns]
            input_data = input_data.fillna(-100)

            predictions = self.telescope_regressors[tel_id].predict(input_data[self.feature_names])

            disp_az.loc[idx[:, :, tel_id]] = predictions

            disp_az = disp_az.reindex(shower_data_with_direction.index)

            shower_data_with_direction = shower_data_with_direction.assign(**{disp_az.name: disp_az.values})

        return shower_data_with_direction

    def save(self, file_name):
        """
        Saves trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        output = dict()
        output['feature_names'] = self.feature_names
        output['target_name'] = self.target_name
        output['telescope_regressors'] = self.telescope_regressors
        output['consolidating_regressor'] = self.consolidating_regressor

        joblib.dump(output, file_name)

    def load(self, file_name):
        """
        Loads pre-trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        data = joblib.load(file_name)

        self.feature_names = data['feature_names']
        self.target_name = data['target_name']
        self.telescope_regressors = data['telescope_regressors']
        self.consolidating_regressor = data['consolidating_regressor']
