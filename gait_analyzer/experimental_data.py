import os
import ezc3d
import numpy as np
from pyomeca import Analogs

from gait_analyzer.model_creator import ModelCreator
from gait_analyzer.operator import Operator
from gait_analyzer.subject import Subject


class ExperimentalData:
    """
    This class contains all the experimental data from a trial (markers, EMG, force plates data, gait parameters).
    """

    def __init__(
        self,
        c3d_file_name: str,
        result_folder: str,
        model_creator: ModelCreator,
        markers_to_ignore: list[str],
        analogs_to_ignore: list[str],
        animate_c3d_flag: bool,
    ):
        """
        Initialize the ExperimentalData.
        .
        Parameters
        ----------
        c3d_file_name: str
            The name of the trial's c3d file.
        subject: Subject
            The subject to analyze.
        result_folder: str
            The folder where the results will be saved. It should look like result_folder/subject_name.
        model_creator: ModelCreator
            The subject's personalized biorbd model.
        markers_to_ignore: list[str]
            Supplementary markers to ignore in the analysis.
        analogs_to_ignore: list[str]
            Supplementary analogs to ignore in the analysis (e.g., EMG signals).
        animate_c3d_flag: bool
            If True, the c3d file will be animated.
        """
        # Checks
        if not isinstance(c3d_file_name, str):
            raise ValueError("c3d_file_name must be a string")
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")

        # Threshold for removing force values
        # TODO: Validate because this value is high !
        self.force_threshold = 15  # N

        # Initial attributes
        self.c3d_full_file_path = c3d_file_name
        self.model_creator = model_creator
        self.markers_to_ignore = markers_to_ignore
        self.analogs_to_ignore = analogs_to_ignore
        self.result_folder = result_folder

        # Extended attributes
        self.c3d = None
        self.model_marker_names = None
        self.marker_sampling_frequency = None
        self.markers_dt = None
        self.marker_units = None
        self.nb_marker_frames = None
        self.markers_sorted = None
        self.analogs_sampling_frequency = None
        self.normalized_emg = None
        self.analog_names = None
        self.platform_corners = None
        self.analogs_dt = None
        self.nb_analog_frames = None
        self.f_ext_sorted = None
        self.f_ext_sorted_filtered = None
        self.markers_time_vector = None
        self.analogs_time_vector = None

        # Extract data from the c3d file
        print(f"Reading experimental data from file {self.c3d_full_file_path} ...")
        self.perform_initial_treatment()
        self.extract_gait_parameters(nb_cycle=None)
        if animate_c3d_flag:
            self.animate_c3d()

    def perform_initial_treatment(self):
        """
        Extract important information and sort the data
        """

        def load_model():
            self.model_marker_names = [
                m.to_string()
                for m in self.model_creator.biorbd_model.markerNames()
                if m.to_string() not in self.markers_to_ignore
            ]

        def sort_markers():
            self.c3d = ezc3d.c3d(self.c3d_full_file_path, extract_forceplat_data=True)
            markers = self.c3d["data"]["points"]
            self.marker_sampling_frequency = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]  # Hz
            self.markers_dt = 1 / self.c3d["header"]["points"]["frame_rate"]
            self.nb_marker_frames = markers.shape[2]
            exp_marker_names = [
                m for m in self.c3d["parameters"]["POINT"]["LABELS"]["value"] if m not in self.markers_to_ignore
            ]

            self.marker_units = 1
            if self.c3d["parameters"]["POINT"]["UNITS"]["value"][0] == "mm":
                self.marker_units = 0.001
            if len(self.model_marker_names) > len(exp_marker_names):
                supplementary_marker_names = [name for name in self.model_marker_names if name not in exp_marker_names]
                raise ValueError(
                    f"The markers {supplementary_marker_names} are not in the c3d file, but are in the model."
                )
            elif len(self.model_marker_names) < len(exp_marker_names):
                supplementary_marker_names = [name for name in exp_marker_names if name not in self.model_marker_names]
                raise ValueError(f"The markers {supplementary_marker_names} are in the c3d file, but not in the model.")

            markers_sorted = np.zeros((3, len(self.model_marker_names), self.nb_marker_frames))
            markers_sorted[:, :, :] = np.nan
            for i_marker, name in enumerate(exp_marker_names):
                if name not in self.markers_to_ignore:
                    marker_idx = self.model_marker_names.index(name)
                    markers_sorted[:, marker_idx, :] = markers[:3, i_marker, :] * self.marker_units
            self.markers_sorted = markers_sorted

        def sort_analogs():
            """
            Sort the analogs data from the c3d file.
            Extract the EMG signals, filter, and normalize (using MVC).
            """

            # Get an array of the experimental muscle activity
            analogs = self.c3d["data"]["analogs"]
            self.nb_analog_frames = analogs.shape[2]
            self.analogs_sampling_frequency = self.c3d["parameters"]["ANALOG"]["RATE"]["value"][0]  # Hz
            self.analogs_dt = 1 / self.c3d["header"]["analogs"]["frame_rate"]
            self.analog_names = [
                name
                for name in self.c3d["parameters"]["ANALOG"]["LABELS"]["value"]
                if name not in self.analogs_to_ignore
            ]

            self.emg_units = 1
            for i_analog, name in enumerate(self.c3d["parameters"]["ANALOG"]["LABELS"]["value"]):
                if name not in self.analogs_to_ignore:
                    if self.c3d["parameters"]["ANALOG"]["UNITS"]["value"][i_analog] == "V":
                        self.emg_units = 1_000_000  # Convert to microV

            # Make sure all MVC are declared

            if not self.model_creator.mvc_values:
                print("No MVC values available, skipping EMG normalization.")
                self.normalized_emg = None  # ou un tableau de NaN selon ce qu'attend le reste du pipeline
                return
            for analog_name in self.analog_names:
                if analog_name not in self.model_creator.mvc_values.keys():
                    raise RuntimeError(
                        f"There was not MVC trial for muscle {analog_name}, available muscles are {self.model_creator.mvc_values.keys()}. Please check that the MVC trials are correctly named and placed in the folder {self.model_creator.mvc_trials_path}."
                    )

            # Process the EMG signals
            normalized_emg = np.zeros((len(self.analog_names), self.nb_analog_frames))
            for i_muscle, muscle_name in enumerate(self.analog_names):
                emg = Analogs.from_c3d(
                    self.c3d_full_file_path, suffix_delimiter=".", usecols=[muscle_name]
                )
                emg = emg.interpolate_na(dim="time", method="linear")
                emg_data_clean = np.nan_to_num(np.array(emg.data), nan=0.0)  # garde la forme (1, 120000)
                emg.data[:] = emg_data_clean
                emg_processed = (
                    # emg.meca.interpolate_missing_data()
                    emg.meca.band_pass(order=2, cutoff=[10, 425])
                    .meca.center()
                    .meca.abs()
                    .meca.low_pass(order=4, cutoff=5, freq=emg.rate)
                ) * self.emg_units
                normalized_emg[i_muscle, :] = (
                    np.array(emg_processed) / self.model_creator.mvc_values[muscle_name]
                )
                normalized_emg[i_muscle, normalized_emg[i_muscle, :] < 0] = (
                    0  # There are still small negative values after meca.abs()
                )

            self.normalized_emg = normalized_emg

            if np.any(self.normalized_emg > 1):
                # raise RuntimeError("The experimental trial reached EMG values above the MVC, which is not expected. ")
                for i_emg in range(self.normalized_emg.shape[0]):
                    if np.nanmax(self.normalized_emg[i_emg, :]) > 1:
                        print(
                            f"Muscle {self.analog_names[i_emg]} reached {np.nanmax(self.normalized_emg[i_emg, :])}... renormalizing with this new maximum."
                        )
                        self.normalized_emg[i_emg, :] /= np.nanmax(self.normalized_emg[i_emg, :])

        def extract_force_platform_data():
            """
            Extracts the force platform data from the c3d file and filters it.
            The F_ext output is of the form [cop, moments, forces].
            """

            platforms = self.c3d["data"]["platform"]
            nb_platforms = len(platforms)
            units = self.marker_units  # We assume that the all position units are the same as the markers'
            self.platform_corners = []
            # for platform in platforms:
            #     self.platform_corners += [platform["corners"] * units]
            # Positions fixes des plateformes (en mm, issues de LEM03 comme référence)
            # I fix the platform coordinates because someone messed up the QTM setup in the lab...
            fixed_corners = [
                np.array([
                    [1614., 1614., -44.3, -44.3],
                    [1038., 527., 527., 1038.],
                    [0., 0., 0., 0.]
                ]) * units,  # Platform 1
                np.array([
                    [1614., 1614., -44.3, -44.3],
                    [494., -17., -17., 494.],
                    [0., 0., 0., 0.]
                ]) * units,  # Platform 2
            ]
            for corners in fixed_corners:
                self.platform_corners += [corners]
            # Initialize arrays for storing external forces and moments
            force_filtered = np.zeros((nb_platforms, 3, self.nb_analog_frames))
            moment_filtered = np.zeros((nb_platforms, 3, self.nb_analog_frames))
            tz_filtered = np.zeros((nb_platforms, 3, self.nb_analog_frames))
            cop_filtered = np.zeros((nb_platforms, 3, self.nb_analog_frames))
            f_ext_sorted = np.zeros((nb_platforms, 9, self.nb_analog_frames))
            f_ext_sorted_filtered = np.zeros((nb_platforms, 9, self.nb_analog_frames))

            # Process force platform data
            for i_platform in range(nb_platforms):

                # Get the data
                force = platforms[i_platform]["force"]
                moment = platforms[i_platform]["moment"] * units
                tz = platforms[i_platform]["Tz"] * units
                tz[:2, :] = 0  # This is the intended behavior (no moments on X and Y at the CoP)

                # Filter forces and moments
                # TODO: Charbie -> Antoine is supposed to send a ref for this filtering
                force_filtered[i_platform, :, :] = Operator.apply_filtfilt(
                    force, order=2, sampling_rate=self.analogs_sampling_frequency, cutoff_freq=10
                )
                moment_filtered[i_platform, :, :] = Operator.apply_filtfilt(
                    moment, order=2, sampling_rate=self.analogs_sampling_frequency, cutoff_freq=10
                )
                tz_filtered[i_platform, :, :] = Operator.apply_filtfilt(
                    tz, order=2, sampling_rate=self.analogs_sampling_frequency, cutoff_freq=10
                )

                # Remove the values when the force is too small since it is likely only noise
                null_idx = np.where(np.linalg.norm(force_filtered[i_platform, :, :], axis=0) < self.force_threshold)[0]
                moment_filtered[i_platform, :, null_idx] = np.nan
                force_filtered[i_platform, :, null_idx] = np.nan
                tz_filtered[i_platform, :, null_idx] = np.nan

                # Do not trust the CoP from ezc3d and recompute it after filtering the forces and moments
                cop_ezc3d = platforms[i_platform]["center_of_pressure"] * units

                r_z = 0  # In our case the reference frame of the platform is at its surface, so the height is 0
                r_z = np.mean(
                    self.platform_corners[i_platform][2, :])  # Height of the platform origin (non-zero for some setups)
                cop_filtered[i_platform, 0, :] = (
                    -(moment_filtered[i_platform, 1, :] - force_filtered[i_platform, 0, :] * r_z)
                    / force_filtered[i_platform, 2, :]
                )
                cop_filtered[i_platform, 1, :] = (
                    moment_filtered[i_platform, 0, :] + force_filtered[i_platform, 1, :] * r_z
                ) / force_filtered[i_platform, 2, :]
                cop_filtered[i_platform, 2, :] = r_z
                # The CoP must be expressed relatively to the center of the platforms
                cop_filtered[i_platform, :, :] += np.tile(
                    np.mean(self.platform_corners[i_platform], axis=1), (self.nb_analog_frames, 1)
                ).T

                # Store output in a biorbd compatible format
                f_ext_sorted[i_platform, :3, :] = cop_ezc3d[:, :]
                f_ext_sorted_filtered[i_platform, :3, :] = cop_filtered[i_platform, :, :]
                f_ext_sorted[i_platform, 3:6, :] = tz[:, :]
                # f_ext_sorted_filtered[i_platform, 3:6, :] = moment_filtered[i_platform, :, :] #Tz
                f_ext_sorted_filtered[i_platform, 3:6, :] = tz_filtered[i_platform, :, :]  # comme f_ext_sorted
                f_ext_sorted[i_platform, 6:9, :] = force[:, :]
                f_ext_sorted_filtered[i_platform, 6:9, :] = force_filtered[i_platform, :, :]

                # Check if the ddata is computed the same way in ezc3d and in this code
                is_good_trial = True
                for i_component in range(3):
                    bad_index = np.where(cop_ezc3d[i_component, :] - cop_filtered[i_platform, i_component, :] > 0.05)
                    if len(bad_index) > 0 and bad_index[0].shape[0] > self.nb_analog_frames / 100:
                        is_good_trial = False
                    cop_filtered[i_platform, i_component, bad_index] = np.nan
                if np.nanmean(cop_ezc3d[:2, :] - cop_filtered[i_platform, :2, :]) > 1e-3:
                    is_good_trial = False

                if not is_good_trial:
                    import matplotlib.pyplot as plt

                    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

                    axs[0].plot(cop_ezc3d[0, :], "-b", label="CoP ezc3d raw")
                    axs[1].plot(cop_ezc3d[1, :], "-b")
                    axs[2].plot(cop_ezc3d[2, :], "-b")

                    axs[0].plot(cop_filtered[i_platform, 0, :], "--r", label="CoP recomputed (from filtered F and M)")
                    axs[1].plot(cop_filtered[i_platform, 1, :], "--r")
                    axs[2].plot(cop_filtered[i_platform, 2, :], "--r")

                    axs[0].set_xlim(0, 25000)
                    axs[1].set_xlim(0, 25000)
                    axs[2].set_xlim(0, 25000)

                    axs[0].set_ylim(-1, 1)
                    axs[1].set_ylim(-1, 1)
                    axs[2].set_ylim(-0.01, 0.01)

                    axs[3].plot(np.linalg.norm(cop_ezc3d[:2, :] - cop_filtered[i_platform, :2, :], axis=0))
                    axs[3].plot(np.array([0, cop_ezc3d.shape[1]]), np.array([1e-3, 1e-3]), "--k")
                    axs[3].set_ylabel("Error (m)")

                    axs[0].legend()
                    fig.savefig("CoP_filtering_error.png")
                    fig.show()
                    raise NotImplementedError(
                        "The force platform data is not computed the same way in ezc3d than in this code, see the CoP graph."
                    )

            self.f_ext_sorted = f_ext_sorted
            self.f_ext_sorted_filtered = f_ext_sorted_filtered

        def compute_time_vectors():
            self.markers_time_vector = np.linspace(0, self.markers_dt * self.nb_marker_frames, self.nb_marker_frames)
            self.analogs_time_vector = np.linspace(0, self.analogs_dt * self.nb_analog_frames, self.nb_analog_frames)

        # Perform the initial treatment
        load_model()
        sort_markers()
        sort_analogs()
        extract_force_platform_data()
        compute_time_vectors()

    def animate_c3d(self):
        try:
            from pyorerun import BiorbdModel, PhaseRerun
        except:
            raise RuntimeError("To animate the .c3d, you first need to install Pyorerun.")
        raise NotImplementedError("Animation of c3d files is not implemented yet.")
        pass

    def extract_gait_parameters(self, threshold: float = None, nb_cycle: int = None):
        """
        Detect gait events from force platforms + CAL markers and compute gait parameters.
        - threshold: threshold on vertical force (N). If None, use self.force_threshold.
        - nb_cycle: number of cycles to compute. If None, select up to 5 (or available cycles).
        Results are stored as:
          self.gait_parameters_all = {'right_leg': {...}, 'left_leg': {...}}
          self.gait_parameters_meanstd = {'right_leg': {...}, 'left_leg': {...}}
        """
        if threshold is None:
            threshold = self.force_threshold

        try:
            idx_RCAL = self.model_marker_names.index("RCAL")
        except ValueError:
            idx_RCAL = None
        try:
            idx_LCAL = self.model_marker_names.index("LCAL")
        except ValueError:
            idx_LCAL = None

        if idx_RCAL is None and idx_LCAL is None:
            raise RuntimeError("Neither RCAL nor LCAL markers found in model_marker_names.")

        if idx_RCAL is not None:
            traj_RCAL = self.markers_sorted[:, idx_RCAL, :]  # shape (3, nframes)
        else:
            traj_RCAL = None
        if idx_LCAL is not None:
            traj_LCAL = self.markers_sorted[:, idx_LCAL, :]
        else:
            traj_LCAL = None

        if self.f_ext_sorted_filtered is None:
            raise RuntimeError("Force data (f_ext_sorted_filtered) not computed.")

        nb_platforms = self.f_ext_sorted_filtered.shape[0]
        if nb_platforms < 1:
            raise RuntimeError("No force platforms found in f_ext_sorted_filtered.")

        force1 = self.f_ext_sorted_filtered[0, 6:9, :].copy() if nb_platforms >= 1 else None
        force2 = self.f_ext_sorted_filtered[1, 6:9, :].copy() if nb_platforms >= 2 else None
        if force1 is not None:
            force1 = np.where(np.isnan(force1), 0.0, force1)
        if force2 is not None:
            force2 = np.where(np.isnan(force2), 0.0, force2)
        if force1 is None and force2 is None:
            raise RuntimeError("Could not find force arrays on platforms 1 or 2.")

        # TODO: move this code to avoid duplication with the biomechanics quantities computation

        def _gait_parameters_calculation(foot: int, nbcycle: int | None, threshold_local: float):
            if foot == 1:
                fv_foot1 = force1[2, :].copy() if force1 is not None else np.zeros(self.nb_analog_frames)
                fv_foot2 = force2[2, :].copy() if force2 is not None else np.zeros(self.nb_analog_frames)
                opp_cal = traj_RCAL if traj_RCAL is not None else traj_LCAL
                study_cal = traj_LCAL if traj_LCAL is not None else traj_RCAL
            else:
                fv_foot1 = force2[2, :].copy() if force2 is not None else np.zeros(self.nb_analog_frames)
                fv_foot2 = force1[2, :].copy() if force1 is not None else np.zeros(self.nb_analog_frames)
                opp_cal = traj_LCAL if traj_LCAL is not None else traj_RCAL
                study_cal = traj_RCAL if traj_RCAL is not None else traj_LCAL

            baseline1 = np.nanmean(fv_foot1[fv_foot1 < 20])
            baseline2 = np.nanmean(fv_foot2[fv_foot2 < 20])

            fv_foot1 = fv_foot1 - baseline1
            fv_foot2 = fv_foot2 - baseline2

            fs_force = self.analogs_sampling_frequency
            fs_mks = self.marker_sampling_frequency

            above = fv_foot1 > threshold_local
            idx_all = np.where(np.diff(above.astype(int)) == 1)[0] + 1

            if idx_all.size > 1:
                diffs = np.diff(idx_all)
                mean_diff = np.mean(diffs)
                idx_err = np.where(diffs < 0.75 * mean_diff)[0]
                if idx_err.size > 0:
                    idx_all = np.delete(idx_all, idx_err + 1)

            if idx_all.size < 4:
                return {}


            idx_deb = idx_all
            idx_end = idx_all[1:]
            start_idx = 2
            available_cycles = len(idx_end) - start_idx
            if nbcycle is None:
                nbcycle = available_cycles
            else:
                nbcycle = min(nbcycle, max(0, available_cycles))

            use_idx_deb = idx_deb[start_idx: start_idx + nbcycle]
            use_idx_end = idx_end[start_idx: start_idx + nbcycle]

            idx_TpfToTframe = fs_mks / fs_force if fs_force != 0 else 1.0

            end_contact_foot_study = np.zeros(nbcycle, dtype=int)
            end_contact_foot_opp = np.zeros(nbcycle, dtype=int)
            start_contact_foot_opp = np.zeros(nbcycle, dtype=int)

            for ii in range(nbcycle):
                a = use_idx_deb[ii]
                b = use_idx_end[ii]
                seg = fv_foot1[a: b + 1]
                rel = np.where(seg > threshold_local)[0]
                if rel.size == 0:
                    end_contact_foot_study[ii] = a
                else:
                    end_contact_foot_study[ii] = a + rel[-1]

                seg2 = fv_foot2[a: b + 1]
                rel2 = np.where(seg2 < threshold_local)[0]
                if rel2.size == 0:
                    end_contact_foot_opp[ii] = a
                else:
                    end_contact_foot_opp[ii] = a + rel2[0]

                pre_seg = fv_foot2[: use_idx_end[ii] + 1]
                rel3 = np.where(pre_seg < threshold_local)[0]
                if rel3.size == 0:
                    start_contact_foot_opp[ii] = 0
                else:
                    start_contact_foot_opp[ii] = rel3[-1]

            if len(use_idx_deb) >= 2:
                stride_time = np.diff(use_idx_deb) / fs_force
            else:
                stride_time = np.array([])

            single_support_time = (start_contact_foot_opp[: nbcycle - 1] - end_contact_foot_opp[
                                                                           : nbcycle - 1]) / fs_force
            double_support_time = (end_contact_foot_study[: nbcycle - 1] - start_contact_foot_opp[
                                                                           : nbcycle - 1]) / fs_force
            stance_time = single_support_time + double_support_time
            swing_time = stride_time - stance_time
            frequency = 1.0 / stride_time if stride_time.size > 0 else np.array([])

            step_length = np.array([])
            step_width = np.array([])
            opp_step_length = np.array([])
            stride_length = np.array([])
            velocity = np.array([])

            if study_cal is not None and opp_cal is not None:

                marker_idx2 = np.round(start_contact_foot_opp * idx_TpfToTframe).astype(int)
                marker_idx2 = np.clip(marker_idx2, 0, study_cal.shape[1] - 1)
                StepLength = np.abs(opp_cal[0, marker_idx2] - study_cal[0, marker_idx2])
                StepLength = np.where(StepLength > 100, StepLength / 1000.0, StepLength)
                StrideLength = StepLength[: StepLength.size - 1] + StepLength[: StepLength.size - 1]
                # step width: mean(abs(opp_cal(2,:) - study_cal(2,:))) / 1000
                StepWidth = np.abs(opp_cal[1, marker_idx2] - study_cal[1, marker_idx2])

                if stride_time.size > 0:
                    Velocity = StrideLength / stride_time
                else:
                    Velocity = np.array([])

                step_length = StepLength
                step_width = StepWidth
                stride_length = StrideLength
                velocity = Velocity

            gait_parameters = {
                "StrideTime": stride_time,
                "SingleSupportTime": single_support_time,
                "DoubleSupportTime": double_support_time,
                "StanceTime": stance_time,
                "SwingTime": swing_time,
                "Frequence": frequency,
                "StepLength": step_length,
                "StepWidth": step_width,
                "StrideLength": stride_length,
                "Velocity": velocity,
                "idx_deb": use_idx_deb,
                "idx_end": use_idx_end
            }
            return gait_parameters

        # compute for left and right (foot 1 and 2 in matlab naming)
        gait_left = _gait_parameters_calculation(1, nb_cycle, threshold)
        gait_right = _gait_parameters_calculation(2, nb_cycle, threshold)
        self.gait_parameters_all = {"right_leg": gait_right, "left_leg": gait_left}
        self.gait_parameters_meanstd = self._estimation_mean_std_abs_diff_per(self.gait_parameters_all)


    def _estimation_mean_std_abs_diff_per(self, data):
        """
        Equivalent of estimation_mean_std_abs_diff_per in MATLAB.
        data is dict with keys 'right_leg' and 'left_leg'; each contains dict of arrays.
        Returns dict with mean and std for each variable and each leg.
        """
        res = {}
        for leg in ("right_leg", "left_leg"):
            res[leg] = {}
            if leg not in data or not data[leg]:
                continue
            for var, arr in data[leg].items():
                try:
                    # if arr is scalar or empty, handle safely
                    arr_np = np.asarray(arr)
                    if arr_np.size == 0:
                        mean_v = np.nan
                        std_v = np.nan
                    else:
                        mean_v = np.nanmean(arr_np)
                        std_v = np.nanstd(arr_np)
                except Exception:
                    mean_v = np.nan
                    std_v = np.nan
                res[leg][var] = (mean_v, std_v)
        return res

    def inputs(self):
        return {
            "c3d_full_file_path": self.c3d_full_file_path,
            "model_creator": self.model_creator,
        }

    def outputs(self):
        return {
            "model_marker_names": self.model_marker_names,
            "marker_sampling_frequency": self.marker_sampling_frequency,
            "markers_dt": self.markers_dt,
            "nb_marker_frames": self.nb_marker_frames,
            "markers_sorted": self.markers_sorted,
            "analogs_sampling_frequency": self.analogs_sampling_frequency,
            "analogs_dt": self.analogs_dt,
            "nb_analog_frames": self.nb_analog_frames,
            "f_ext_sorted": self.f_ext_sorted,
            "f_ext_sorted_filtered": self.f_ext_sorted_filtered,
            "markers_time_vector": self.markers_time_vector,
            "analogs_time_vector": self.analogs_time_vector,
            "normalized_emg": self.normalized_emg,
            "gait_parameters_all": self.gait_parameters_all,
            "gait_parameters_meanstd": self.gait_parameters_meanstd,
            "gait_parameters_all": self.gait_parameters_all,
            "gait_parameters_meanstd": self.gait_parameters_meanstd,
            "idx_deb_right": self.gait_parameters_all.get("right_leg", {}).get("idx_deb"),
            "idx_end_right": self.gait_parameters_all.get("right_leg", {}).get("idx_end"),
            "idx_deb_left": self.gait_parameters_all.get("left_leg", {}).get("idx_deb"),
            "idx_end_left": self.gait_parameters_all.get("left_leg", {}).get("idx_end"),
        }
