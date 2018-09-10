import numpy as np
import astropy.units as u
from copy import deepcopy

from abc import ABC, abstractmethod

from ctapipe.io import EventSourceFactory
from ctapipe.image import tailcuts_clean, hillas_parameters
from ctapipe.reco.HillasReconstructor import TooFewTelescopesException
from ctapipe.io.containers import ReconstructedContainer, ReconstructedEnergyContainer, ParticleClassificationContainer


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
    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        super(EventFeatureSelector, self).__init__(hillas_params_to_use, hillas_reco_params_to_use, telescopes)

        self.events = []
        self.event_targets = []

    def fill_event(self, event, hillas_reco_result, target=None):
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
    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        super(EventFeatureTargetSelector, self).__init__(hillas_params_to_use, hillas_reco_params_to_use, telescopes)

        self.features = dict()
        self.targets = dict()
        self.events = []

        for tel_id in self.telescopes:
            self.features[tel_id] = []
            self.targets[tel_id] = []

    def fill_event(self, event, hillas_reco_result, target):
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

