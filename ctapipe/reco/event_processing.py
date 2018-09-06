import numpy as np
import astropy.units as u
from copy import deepcopy

from ctapipe.io import EventSourceFactory
from ctapipe.image import tailcuts_clean, hillas_parameters
from ctapipe.reco.HillasReconstructor import TooFewTelescopesException
from ctapipe.io.containers import ReconstructedContainer, ReconstructedEnergyContainer, ParticleClassificationContainer


class HillasFeatureSelector:
    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        self.hillas_params_to_use = hillas_params_to_use
        self.hillas_reco_params_to_use = hillas_reco_params_to_use
        self.telescopes = telescopes

        n_features_per_telescope = len(hillas_params_to_use) + len(hillas_reco_params_to_use)
        self.n_features = n_features_per_telescope * len(telescopes)

    @staticmethod
    def _get_param_value(param):
        if isinstance(param, u.Quantity):
            return param.value
        else:
            return param

    def fill_event(self, event, hillas_reco_result, target):
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
    def __init__(self, calibrator, hillas_reconstructor, min_survived_pixels=10):
        self.calibrator = calibrator
        self.hillas_reconstructor = hillas_reconstructor
        self.min_survived_pixels = min_survived_pixels

        self.events = []
        self.reconstruction_results = []

    def _dl1_process(self, event):
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
        # Performing a geometrical direction reconstruction
        try:
            reco_container.shower['hillas'] = self.hillas_reconstructor.predict_from_dl1(event)
        except TooFewTelescopesException:
            # No reconstruction possible. Resetting to defaults
            reco_container.shower['hillas'].reset()

        return reco_container

    def _update_event_energy(self, event, reco_container):
        return reco_container

    def _update_event_classification(self, event, reco_container):
        return reco_container

    def _load_events(self, file_name, append_to_existing_events):
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

