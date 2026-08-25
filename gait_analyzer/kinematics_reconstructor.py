import os
import pickle
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import biorbd
import biobuddy

from gait_analyzer.operator import Operator
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.model_creator import ModelCreator
from gait_analyzer.events.cyclic_events import CyclicEvents
from gait_analyzer.events.unique_events import UniqueEvents


class ReconstructionType(Enum):
    """
    Type of reconstruction to perform
    """

    ONLY_LM = "only_lm"  # Levenberg-Marquardt with 0.0001 initialization
    LM = "lm"  # Levenberg-Marquardt with mid-bounds initialization
    TRF = "trf"  # Trust Region Reflective
    EKF = "ekf"  # Extended Kalman Filter
    LSQ = "lsq"  # BioBuddy's Least Squares


segment_dict = {
    "pelvis": {
        "dof_idx": [0, 1, 2, 3, 4, 5],
        "markers_idx": [0, 1, 2, 3, 49],
        "min_bound": [-3, -3, -3, -np.pi / 4, -np.pi / 4, -np.pi],
        "max_bound": [3, 3, 3, np.pi / 4, np.pi / 4, np.pi],
    },
    "femur_r": {
        "dof_idx": [6, 7, 8],
        "markers_idx": [4, 5, 6, 50, 51],
        "min_bound": [-0.6981317007977318, -1.0471975511965976, -0.5235987755982988],
        "max_bound": [2.0943951023931953, 0.5235987755982988, 0.5235987755982988],
    },
    "tibia_r": {
        "dof_idx": [9],
        "markers_idx": [7, 8, 9, 52, 53],
        "min_bound": [-2.6179938779914944],
        "max_bound": [0],
    },
    "calcn_r": {
        "dof_idx": [10, 11],
        "markers_idx": [10, 11, 12, 54, 55],
        "min_bound": [-0.8726646259971648, -0.2617993877991494],
        "max_bound": [0.5235987755982988, 0.2617993877991494],
    },
    "toes_r": {
        "dof_idx": [12],
        "markers_idx": [13],
        "min_bound": [-0.8726646259971648],
        "max_bound": [1.0471975511965976],
    },
    "femur_l": {
        "dof_idx": [13, 14, 15],
        "markers_idx": [14, 15, 16, 56, 57],
        "min_bound": [-0.6981317007977318, -1.0471975511965976, -0.5235987755982988],
        "max_bound": [2.0943951023931953, 0.5235987755982988, 0.5235987755982988],
    },
    "tibia_l": {
        "dof_idx": [16],
        "markers_idx": [17, 18, 19, 58, 59],
        "min_bound": [-2.6179938779914944],
        "max_bound": [0],
    },
    "calcn_l": {
        "dof_idx": [17, 18],
        "markers_idx": [20, 21, 22, 60, 61],
        "min_bound": [-0.8726646259971648, -0.2617993877991494],
        "max_bound": [0.5235987755982988, 0.2617993877991494],
    },
    "toes_l": {
        "dof_idx": [19],
        "markers_idx": [23],
        "min_bound": [-0.8726646259971648],
        "max_bound": [1.0471975511965976],
    },
    "torso": {
        "dof_idx": [20, 21, 22],
        "markers_idx": [24, 25, 26, 27, 28, 62],
        "min_bound": [-1.5707963267948966, -0.6108652381980153, -0.7853981633974483],
        "max_bound": [0.7853981633974483, 0.6108652381980153, 0.7853981633974483],
    },
    "head": {
        "dof_idx": [23, 24, 25],
        "markers_idx": [29, 30, 31, 32, 33],
        "min_bound": [-0.8726646259971648, -0.59999999999999998, -1.2217],
        "max_bound": [0.7853981633974483, 0.59999999999999998, 1.2217],
    },
    "humerus_r": {
        "dof_idx": [26, 27, 28],
        "markers_idx": [34, 35, 63],
        "min_bound": [-1.5707963300000001, -3.8397000000000001, -1.5707963300000001],
        "max_bound": [3.1415999999999999, 1.5707963300000001, 1.5707963300000001],
    },
    "radius_r": {
        "dof_idx": [29, 30],
        "markers_idx": [36, 37, 64],
        "min_bound": [0.0, -3.1415999999999999],
        "max_bound": [3.1415999999999999, 3.1415999999999999],
    },
    "hand_r": {
        "dof_idx": [31, 32],
        "markers_idx": [38, 39, 65],
        "min_bound": [-1.5708, -0.43633231],
        "max_bound": [1.5708, 0.61086523999999998],
    },
    "fingers_r": {"dof_idx": [33], "markers_idx": [40], "min_bound": [-1.5708], "max_bound": [1.5708]},
    "humerus_l": {
        "dof_idx": [34, 35, 36],
        "markers_idx": [41, 42, 66],
        "min_bound": [-1.5707963300000001, -3.8397000000000001, -1.5707963300000001],
        "max_bound": [3.1415999999999999, 1.5707963300000001, 1.5707963300000001],
    },
    "radius_l": {
        "dof_idx": [37, 38],
        "markers_idx": [43, 44, 67],
        "min_bound": [0.0, -3.1415999999999999],
        "max_bound": [3.1415999999999999, 3.1415999999999999],
    },
    "hand_l": {
        "dof_idx": [39, 40],
        "markers_idx": [45, 46, 68],
        "min_bound": [-1.5708, -0.43633231],
        "max_bound": [1.5708, 0.61086523999999998],
    },
    "fingers_l": {"dof_idx": [41], "markers_idx": [47, 48], "min_bound": [-1.5708], "max_bound": [1.5708]},
}


class KinematicsReconstructor:
    """
    This class reconstruct the kinematics based on the marker position and the model predefined.
    """

    def __init__(
        self,
        experimental_data: ExperimentalData,
        model_creator: ModelCreator,
        events: CyclicEvents,
        cycles_to_analyze: range | None,
        reconstruction_type: ReconstructionType | list[ReconstructionType],
        skip_if_existing: bool,
        animate_kinematics_flag: bool,
        plot_kinematics_flag: bool,
        q_regularization_weight: float | np.ndarray = None,
        qdot_regularization_weight: float | np.ndarray = None,
    ):
        """
        Initialize the KinematicsReconstructor.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        model_creator: ModelCreator
            The biorbd model to use for the kinematics reconstruction
        events: CyclicEvents
            The events to use for the kinematics reconstruction since we exploit the fact that the movement is cyclic
        cycles_to_analyze: range | None
            The range of cycles to analyze
        reconstruction_type: ReconstructionType
            The type of algorithm to use to perform the reconstruction
            If the reconstruction_type is a list, the kinematics will be first reconstructed with the first element of the list, and then withe the other ones as a fallback if the reconstruction os not acceptable (<5cm error on the 75e percentile).
        skip_if_existing: bool
            If True, the kinematics will not be reconstructed if the output file already exists
        animate_kinematics_flag: bool
            If True, the kinematics will be animated through pyorerun
        plot_kinematics_flag: bool
            If True, the kinematics will be plotted and saved in a .png
        q_regularization_weight: float | np.ndarray
            The weight of the regularization term on q for the LSQ reconstruction
        qdot_regularization_weight: float | np.ndarray
            The weight of the regularization term on qdot for the LSQ reconstruction
        """
        # Checks
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError(
                "experimental_data must be an instance of ExperimentalData. You can declare it by running ExperimentalData(file_path)."
            )
        if not isinstance(model_creator, ModelCreator):
            raise ValueError("model_creator must be an instance of ModelCreator.")
        if not (isinstance(events, CyclicEvents) or isinstance(events, UniqueEvents)):
            raise ValueError("events must be an instance of CyclicEvents or UniqueEvents.")
        if isinstance(events, UniqueEvents) and cycles_to_analyze is not None:
            raise NotImplementedError(
                "If events is an instance of UniqueEvents, cycles_to_analyze must be None for now."
            )
        if reconstruction_type is None:
            self.reconstruction_type = [ReconstructionType.ONLY_LM]
        elif isinstance(reconstruction_type, ReconstructionType):
            self.reconstruction_type = [reconstruction_type]
        elif isinstance(reconstruction_type, list):
            if not all(isinstance(i_recons, ReconstructionType) for i_recons in reconstruction_type):
                raise ValueError("reconstruction_type must be a list of ReconstructionType.")
            self.reconstruction_type = reconstruction_type
        else:
            raise ValueError(
                "reconstruction_type must be an instance of ReconstructionType or a list of ReconstructionType."
            )

        # Initial attributes
        self.experimental_data = experimental_data
        self.model_creator = model_creator
        self.events = events
        self.cycles_to_analyze = cycles_to_analyze
        self.q_regularization_weight = q_regularization_weight
        self.qdot_regularization_weight = qdot_regularization_weight

        # Parameters of the reconstruction
        self.acceptance_threshold = 0.1  # 10 cm

        # Extended attributes
        self.frame_range = None
        self.padded_frame_range = None
        self.markers = None
        self.marker_residuals = None
        self.biorbd_model = biorbd.Model(self.model_creator.biorbd_model_full_path)
        self.t = None
        self.q = None
        self.q_filtered = None
        self.qdot = None
        self.qddot = None
        self.is_loaded_kinematics = False

        if skip_if_existing and self.check_if_existing():
            self.is_loaded_kinematics = True
        else:
            # Perform the kinematics reconstruction
            self.check_for_marker_inversion()
            self.perform_kinematics_reconstruction()
            self.save_kinematics_reconstruction()

        if animate_kinematics_flag:
            self.animate_kinematics()

        if plot_kinematics_flag:
            self.plot_kinematics()

    def initialize_regularization_weights(self) -> np.ndarray:
        if self.qdot_regularization_weight is None:
            self.q_regularization_weight = np.zeros((self.biorbd_model.nbQ(),))
            self.q_regularization_weight[3:6] = 1.0
            self.q_regularization_weight[20:23] = 1.0
        if self.qdot_regularization_weight is None:
            self.qdot_regularization_weight = np.ones((self.biorbd_model.nbQ(),)) * 0.1

    def check_if_existing(self) -> bool:
        """
        Check if the kinematics reconstruction already exists.
        If it exists, load the q.
        .
        Returns
        -------
        bool
            If the kinematics reconstruction already exists
        """
        result_file_full_path = self.get_result_file_full_path()
        if os.path.exists(result_file_full_path):
            with open(result_file_full_path, "rb") as file:
                data = pickle.load(file)
                self.frame_range = data["frame_range"]
                self.markers = data["markers"]
                self.cycles_to_analyze = data["cycles_to_analyze_kin"]
                self.t = data["t"]
                self.q = data["q"]
                self.q_filtered = data["q_filtered"]
                self.qdot = data["qdot"]
                self.qddot = data["qddot"]
                self.biorbd_model = biorbd.Model(data["biorbd_model"])
                if isinstance(data["reconstruction_type"], str):
                    self.reconstruction_type = ReconstructionType(data["reconstruction_type"])
                else:
                    self.reconstruction_type = [
                        ReconstructionType(i_recons) for i_recons in data["reconstruction_type"]
                    ]
                self.is_loaded_kinematics = True
            return True
        else:
            return False

    def check_for_marker_inversion(self):
        markers = self.experimental_data.markers_sorted
        nb_markers = markers.shape[1]

        # Create mask for valid (non-NaN) data - shape: (3, nb_markers, nb_frames)
        valid_mask = ~np.isnan(markers)
        marker_valid = np.all(valid_mask, axis=0)  # shape: (nb_markers, nb_frames)

        for i_marker in range(nb_markers):
            marker_name = self.biorbd_model.markerNames()[i_marker].to_string()

            # Get indices of valid frames for this marker
            valid_frames = np.where(marker_valid[i_marker, :])[0]

            if len(valid_frames) < 2:
                raise RuntimeError(f"Marker {marker_name} was only found in two frames.")

            # Extract valid positions for this marker
            valid_positions = markers[:, i_marker, valid_frames]  # shape: (3, n_valid)

            # Compute distances between consecutive valid positions
            position_diffs = np.diff(valid_positions, axis=1)  # shape: (3, n_valid-1)
            distances = np.linalg.norm(position_diffs, axis=0)  # shape: (n_valid-1,)

            # Check for jumps > 20m/s
            jump_indices = np.where(
                distances / np.diff(valid_frames) * self.experimental_data.marker_sampling_frequency > 20
            )[0]

            if len(jump_indices) > 0:
                # Report the first jump found
                jump_idx = jump_indices[0]
                frame_before = valid_frames[jump_idx]
                frame_after = valid_frames[jump_idx + 1]
                jump_distance = distances[jump_idx]

                try:
                    from pyorerun import c3d
                except:
                    raise RuntimeError("To animate the kinematics, you must install Pyorerun.")

                c3d(
                    self.experimental_data.c3d_full_file_path,
                    show_forces=False,
                    show_events=False,
                    marker_trajectories=True,
                    show_marker_labels=False,
                )

                raise RuntimeError(
                    f"Marker {marker_name} seems to be inverted between frames "
                    f"{frame_before} and {frame_after} as the distance is "
                    f"{jump_distance:.3f} (larger than 10m/s) see the animation to make sure."
                )

            print(f"Marker {marker_name} OK: max distance = {np.max(distances):.3f} m")
        return

    def perform_kinematics_reconstruction(self):
        """
        Perform the kinematics reconstruction for all frames, and then only keep the frames in the cycles to analyze.
        This is a waist of computation, but the beginning of the reconstruction is always shitty.
        """

        self.frame_range = range(self.experimental_data.markers_sorted.shape[2])
        markers = self.experimental_data.markers_sorted[:, :, :]

        is_successful_reconstruction = False
        residuals = None
        for recons_method in self.reconstruction_type:
            print(f"Performing inverse kinematics reconstruction using {recons_method.value}")
            if recons_method in [ReconstructionType.ONLY_LM, ReconstructionType.LM, ReconstructionType.TRF]:
                ik = biorbd.InverseKinematics(self.biorbd_model, markers)
                q_recons = ik.solve(method=recons_method.value)
                self.q = q_recons
                residuals = ik.sol()["residuals"]
                if self.q_regularization_weight is not None or self.qdot_regularization_weight is not None:
                    print("Warning: Regularization weights are only used for the LSQ reconstruction method.")
                self.q_filtered, self.qdot, self.qddot = self.filter_kinematics()

            elif recons_method == ReconstructionType.LSQ:
                biobuddy_model = biobuddy.BiomechanicalModelReal().from_biomod(
                    self.model_creator.biorbd_model_full_path
                )
                # TODO: Charbie -> Make this modulable
                q_regularization_weight = np.zeros((self.biorbd_model.nbQ(),))
                q_regularization_weight[3:6] = 1.0
                q_regularization_weight[7:20] = 0.6
                q_regularization_weight[20:23] = 2.0
                q_recons, residuals = biobuddy_model.inverse_kinematics(
                    marker_positions=markers,
                    marker_names=biobuddy_model.marker_names,
                    marker_weights=self.model_creator.marker_weights,
                    method="lm",
                    q_regularization_weight=self.q_regularization_weight,
                    qdot_regularization_weight=self.qdot_regularization_weight,
                    q_target=np.zeros((self.biorbd_model.nbQ(),)),
                    animate_reconstruction=False,
                    compute_residual_distance=True,
                )
                self.q = q_recons
                dt = self.experimental_data.markers_dt
                self.t = np.arange(markers.shape[2]) * dt
                self.q_filtered, self.qdot, self.qddot = self.filter_kinematics()

            elif recons_method == ReconstructionType.EKF:
                if self.q_regularization_weight is not None or self.qdot_regularization_weight is not None:
                    print("Warning: Regularization weights are only used for the LSQ reconstruction method.")

                # Create
                _, q_filtered, qdot, qddot = biorbd.extended_kalman_filter(
                    self.biorbd_model,
                    self.experimental_data.c3d_full_file_path,
                    frames=slice(self.frame_range),
                )
                self.q = q_filtered
                self.q_filtered = q_filtered
                self.qdot = qdot
                self.qddot = qddot

                residuals = np.zeros(
                    (
                        3,
                        q_filtered.shape[1],
                    )
                )
                print("Warning: The EKF acceptance criteria was not implemented yet.")
            else:
                raise NotImplementedError(f"The reconstruction_type {recons_method} is not implemented yet.")

            # Check if this reconstruction was acceptable
            residuals = residuals #[:, index_to_keep]
            print(
                f"75 percentile between : {np.min(np.nanpercentile(residuals, 75, axis=0))} and "
                f"{np.max(np.nanpercentile(residuals, 75, axis=0))}"
            )
            if np.all(np.nanpercentile(residuals, 75, axis=0) < self.acceptance_threshold):
                is_successful_reconstruction = True
                break

        if not is_successful_reconstruction:
            raise RuntimeError(
                "The reconstruction was not successful :( Please consider using a different method or checking the experimental data labeling."
            )

        self.q = q_recons
        self.markers = markers
        self.marker_residuals = residuals

    def filter_kinematics(self):
        """
        Unwrap and filter the joint angles.
        """

        def filter(q):
            filter_type = "savgol"  # "filtfilt"  # "savgol"

            # Filter q
            sampling_rate = 1 / (self.t[1] - self.t[0])
            if filter_type == "savgol":
                q_filtered = Operator.apply_savgol(q, window_length=31, polyorder=3)
            elif filter_type == "filtfilt":
                q_filtered = Operator.apply_filtfilt(q, order=4, sampling_rate=sampling_rate, cutoff_freq=6)
            else:
                raise NotImplementedError(
                    f"filter_type {filter_type} not implemented. It must be 'savgol' or 'filtfilt'."
                )

            # Compute and filter qdot
            qdot = np.zeros_like(q)
            for i_data in range(qdot.shape[0]):
                qdot[i_data, 0] = (q_filtered[i_data, 1] - q_filtered[i_data, 0]) / (
                    self.t[1] - self.t[0]
                )  # Forward finite diff
                qdot[i_data, 1:-1] = (q_filtered[i_data, 2:] - q_filtered[i_data, :-2]) / (
                    self.t[2:] - self.t[:-2]
                )  # Centered finite diff
                qdot[i_data, -1] = (q_filtered[i_data, -1] - q_filtered[i_data, -2]) / (
                    self.t[-1] - self.t[-2]
                )  # Backward finite diff

            # Compute and filter qddot
            qddot = np.zeros_like(q)
            for i_data in range(qddot.shape[0]):
                qddot[i_data, 0] = (qdot[i_data, 1] - qdot[i_data, 0]) / (self.t[1] - self.t[0])
                qddot[i_data, 1:-1] = (qdot[i_data, 2:] - qdot[i_data, :-2]) / (self.t[2:] - self.t[:-2])
                qddot[i_data, -1] = (qdot[i_data, -1] - qdot[i_data, -2]) / (self.t[-1] - self.t[-2])
            for i_data in range(qdot.shape[0]):
                qdot[i_data, :] = Operator.apply_filtfilt(
                    qdot[i_data:i_data + 1, :],
                    order=4,
                    sampling_rate=sampling_rate,
                    cutoff_freq=6
                )
            return q_filtered, qdot, qddot

        q_filtered, qdot, qddot = filter(self.q)

        return q_filtered, qdot, qddot

    def plot_kinematics(self):
        all_in_one = True
        if all_in_one:
            fig = plt.figure(figsize=(10, 10))
            for i_dof in range(self.q.shape[0]):
                if i_dof < 3:
                    plt.plot(
                        self.t, self.q_filtered[i_dof, :], label=f"{self.biorbd_model.nameDof()[i_dof].to_string()} [m]"
                    )
                else:
                    plt.plot(
                        self.t,
                        self.q_filtered[i_dof, :] * 180 / np.pi,
                        label=f"{self.biorbd_model.nameDof()[i_dof].to_string()} [" + r"$^\circ$" + "]",
                    )
            plt.legend()
            fig.tight_layout()
            result_file_full_path = self.get_result_file_full_path(self.experimental_data.result_folder + "/figures")
            fig.savefig(result_file_full_path.replace(".pkl", "_ALL_IN_ONE.png"))
        else:
            fig, axs = plt.subplots(7, 6, figsize=(10, 10))
            axs = axs.ravel()
            for i_dof in range(self.q.shape[0]):
                if i_dof < 3:
                    axs[i_dof].plot(self.t, self.q_filtered[i_dof, :])
                    axs[i_dof].set_title(f"{self.biorbd_model.nameDof()[i_dof].to_string()} [m]")
                else:
                    axs[i_dof].plot(self.t, self.q_filtered[i_dof, :] * 180 / np.pi)
                    axs[i_dof].set_title(f"{self.biorbd_model.nameDof()[i_dof].to_string()} [" + r"$^\circ$" + "]")
            fig.tight_layout()
            result_file_full_path = self.get_result_file_full_path(self.experimental_data.result_folder + "/figures")
            fig.savefig(result_file_full_path.replace(".pkl", ".png"))

    def animate_kinematics(self):
        """
        Animate the kinematics
        """
        try:
            from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers, PyoMuscles
        except:
            raise RuntimeError("To animate the kinematics, you must install Pyorerun.")

        # Model
        model = BiorbdModel.from_biorbd_object(self.biorbd_model)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        model.options.show_marker_labels = False
        model.options.show_center_of_mass_labels = False
        model.options.show_gravity = False

        # Markers
        marker_names = [m.to_string() for m in self.biorbd_model.markerNames()]
        marker_data_with_ones = np.ones((4, self.markers.shape[1], self.markers.shape[2]))
        marker_data_with_ones[:3, :, :] = self.markers

        t_animation = self.t
        frame_range = range(self.experimental_data.markers_sorted.shape[2])
        if self.q.shape[0] == model.nb_q:
            q_animation = self.q_filtered.reshape(model.nb_q, len(list(self.frame_range)))
        else:
            q_animation = self.q_filtered.T

        if q_animation.shape[1] > 500:
            print("To avoid computer crashes, only the first 200 frames will be displayed in the animation. ")
            q_animation = q_animation[:, :500]
            t_animation = t_animation[:500]
            frame_range = frame_range[:500]
            marker_data_with_ones = marker_data_with_ones[:, :, :500]

        # Visualization
        viz = PhaseRerun(t_animation)

        markers = PyoMarkers(data=marker_data_with_ones, marker_names=marker_names, show_labels=False)
        muscle_names = [m.to_string() for m in self.biorbd_model.muscleNames()]

        # Force plates
        analog_idx = Operator.from_marker_frame_to_analog_frame(
            self.experimental_data.analogs_time_vector,
            self.experimental_data.markers_time_vector,
            list(frame_range),
        )

        # EMGs
        emg_data = []
        muscle_name_mapping = self.model_creator.osim_model_type.muscle_name_mapping
        mvc_names = list(self.model_creator.mvc_values.keys())
        for muscle_name in muscle_names:
            if muscle_name in muscle_name_mapping.keys() and muscle_name_mapping[muscle_name] in mvc_names:
                muscle_index = mvc_names.index(muscle_name_mapping[muscle_name])
                emg_data += [self.experimental_data.normalized_emg[muscle_index, :]]
            else:
                nb_frames = self.experimental_data.normalized_emg.shape[1]
                emg_data += [np.zeros((nb_frames,))]
        emg_data = np.array(emg_data)[:, analog_idx]
        emg = PyoMuscles(data=emg_data, muscle_names=muscle_names, colormap="viridis")

        viz.add_force_plate(num=1, corners=self.experimental_data.platform_corners[0])
        viz.add_force_plate(num=2, corners=self.experimental_data.platform_corners[1])
        viz.add_force_data(
            num=1,
            force_origin=self.experimental_data.f_ext_sorted_filtered[0, :3, analog_idx].T,
            force_vector=self.experimental_data.f_ext_sorted_filtered[0, 6:9, analog_idx].T,
        )
        viz.add_force_data(
            num=2,
            force_origin=self.experimental_data.f_ext_sorted_filtered[1, :3, analog_idx].T,
            force_vector=self.experimental_data.f_ext_sorted_filtered[1, 6:9, analog_idx].T,
        )

        viz.add_animated_model(model, q_animation, tracked_markers=markers, muscle_activations_intensity=emg)
        viz.rerun("Kinematics reconstruction")

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        result_file_full_path = f"{result_folder}/inv_kin_{trial_name}.pkl"
        return result_file_full_path

    def save_kinematics_reconstruction(self):
        """
        Save the kinematics reconstruction.
        """
        result_file_full_path = self.get_result_file_full_path()
        with open(result_file_full_path, "wb") as file:
            pickle.dump(self.outputs(), file)

    def inputs(self):
        return {
            "biorbd_model": self.biorbd_model,
            "c3d_full_file_path": self.experimental_data.c3d_full_file_path,
        }

    def outputs(self):
        if isinstance(self.reconstruction_type, list):
            reconstruction_type = [i_recons.value for i_recons in self.reconstruction_type]
        else:
            reconstruction_type = self.reconstruction_type.value
        return {
            "biorbd_model": self.model_creator.biorbd_model_full_path,
            "reconstruction_type": reconstruction_type,
            "cycles_to_analyze_kin": self.cycles_to_analyze,
            "frame_range": self.frame_range,
            "padded_frame_range": self.padded_frame_range,
            "markers": self.markers,
            "marker_residuals": self.marker_residuals,
            "t": self.t,
            "q": self.q,
            "q_filtered": self.q_filtered,
            "qdot": self.qdot,
            "qddot": self.qddot,
            "is_loaded_kinematics": self.is_loaded_kinematics,
        }
