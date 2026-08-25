import pickle
import os
import numpy as np

from gait_analyzer import Operator, KinematicsReconstructor
from gait_analyzer.experimental_data import ExperimentalData


class InverseDynamicsPerformer:
    """
    This class performs the inverse dynamics based on the kinematics and the external forces.
    """

    def __init__(
        self,
        experimental_data: ExperimentalData,
        kinematics_reconstructor: KinematicsReconstructor,
        skip_if_existing: bool,
        reintegrate_flag: bool,
        animate_dynamics_flag: bool,
    ):
        """
        Initialize the InverseDynamicsPerformer.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        kinematics_reconstructor: KinematicsReconstructor
            The kinematics reconstructor
        skip_if_existing: bool
            If True, the inverse dynamics is not performed if the results already exist
        reintegrate_flag: bool
            If True the dynamics is reintegrated to confirm the results
        animate_dynamics_flag: bool
            If True an animation of the dynamics is shown using Pyorerun
        """

        # Checks
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError(
                "experimental_data must be an instance of ExperimentalData. You can declare it by running ExperimentalData(file_path)."
            )
        if not isinstance(kinematics_reconstructor, KinematicsReconstructor):
            raise ValueError(
                "biorbd_model must be an instance of biorbd.Model. You can declare it by running biorbd.Model('path_to_model.bioMod')."
            )
        if not isinstance(skip_if_existing, bool):
            raise ValueError("skip_if_existing must be a boolean")
        if not isinstance(reintegrate_flag, bool):
            raise ValueError("reintegrate_flag must be a boolean")
        if not isinstance(animate_dynamics_flag, bool):
            raise ValueError("animate_dynamics_flag must be a boolean")
        if animate_dynamics_flag and not reintegrate_flag:
            print("When animate_dynamics_flag is True, reintegrate_flag is automatically set to True.")
            print("To avoid crashes, only the first 200 frames will be displayed in the animation.")
            reintegrate_flag = True

        # Initial attributes
        self.experimental_data = experimental_data
        self.biorbd_model = kinematics_reconstructor.biorbd_model
        self.kinematics_reconstructor = kinematics_reconstructor
        self.q_filtered = kinematics_reconstructor.q_filtered
        self.qdot = kinematics_reconstructor.qdot
        self.qddot = kinematics_reconstructor.qddot
        self.t = kinematics_reconstructor.t

        # Extended attributes
        self.tau = None
        self.q_reintegrated = None
        self.is_loaded_inverse_dynamics = False

        # Perform the inverse dynamics
        if skip_if_existing and self.check_if_existing():
            self.is_loaded_inverse_dynamics = True
        else:
            print("Performing inverse dynamics...")
            self.perform_inverse_dynamics()
            self.save_inverse_dynamics()

        # Reintegrate the dynamics to confirm the results (q, tau, f_ext)
        if reintegrate_flag:
            self.reintegrate_dynamics()

        if animate_dynamics_flag:
            self.animate_dynamics()

    def check_if_existing(self) -> bool:
        """
        Check if the events detection already exists.
        If it exists, load the events.
        .
        Returns
        -------
        bool
            If the events detection already exists
        """
        result_file_full_path = self.get_result_file_full_path()
        if os.path.exists(result_file_full_path):
            with open(result_file_full_path, "rb") as file:
                data = pickle.load(file)
                self.tau = data["tau"]
                self.q_reintegrated = data["q_reintegrated"] if data["q_reintegrated"] != 0 else None
            return True
        else:
            return False

    def perform_inverse_dynamics(self):
        tau = np.zeros_like(self.q_filtered)
        for i_node in range(self.q_filtered.shape[1]):
            f_ext = self.get_f_ext_at_frame(i_node)
            tau[:, i_node] = self.biorbd_model.InverseDynamics(
                self.q_filtered[:, i_node], self.qdot[:, i_node], self.qddot[:, i_node], f_ext
            ).to_array()
        self.tau = tau

    def get_f_ext_at_frame(self, i_marker_node: int):
        f_ext_set = self.biorbd_model.externalForceSet()
        i_analog_node = Operator.from_marker_frame_to_analog_frame(
            self.experimental_data.analogs_time_vector, self.experimental_data.markers_time_vector, i_marker_node
        )
        analog_to_marker_ratio = int(
            round(
                self.experimental_data.analogs_time_vector.shape[0]
                / self.experimental_data.markers_time_vector.shape[0]
            )
        )
        frame_range = list(
            range(i_analog_node - (int(analog_to_marker_ratio / 2)), i_analog_node + (int(analog_to_marker_ratio / 2)))
        )

        def safe_nanmean(data):
            result = np.nanmean(data, axis=0)
            result[np.isnan(result)] = 0.0
            return result


        f_ext_set.add(
            "calcn_l",
            safe_nanmean(self.experimental_data.f_ext_sorted_filtered[0, 3:9, frame_range]),
            safe_nanmean(self.experimental_data.f_ext_sorted_filtered[0, :3, frame_range]),
        )
        f_ext_set.add(
            "calcn_r",
            safe_nanmean(self.experimental_data.f_ext_sorted_filtered[1, 3:9, frame_range]),
            safe_nanmean(self.experimental_data.f_ext_sorted_filtered[1, :3, frame_range]),
        )
        return f_ext_set

    def reintegrate_dynamics(self):
        """
        Reintegrate the dynamics to confirm the results using scipy.
        """

        def dynamics(t, x):
            i_node = int(np.argmin(np.abs(self.experimental_data.markers_time_vector - t)))
            q = x[: self.biorbd_model.nbQ()]
            qdot = x[self.biorbd_model.nbQ() :]
            u = self.tau[:, i_node]
            f_ext_biorbd = self.get_f_ext_at_frame(i_node)
            dqdot = self.biorbd_model.ForwardDynamics(q, qdot, u, f_ext_biorbd).to_array()
            return np.hstack((qdot, dqdot))

        # Reintegrate the dynamics -> This version explodes my RAM!
        # t_span = np.array([0, self.experimental_data.markers_time_vector[30]])
        # x_reintegrated = solve_ivp(dynamics,
        #                            y0=np.hstack((self.q_filtered[0, :], np.zeros_like(self.q_filtered)[0, :])),
        #                            t_span=t_span,
        #                            t_eval=self.experimental_data.markers_time_vector[0:30:3],
        #                            method="DOP853")

        # Euler integration for now
        frames_to_reintegrate = range(0, 100)

        nb_frames_to_reintegrate = len(list(frames_to_reintegrate))
        dt = self.experimental_data.markers_dt
        x_reintegrated = np.zeros((2 * self.biorbd_model.nbQ(), nb_frames_to_reintegrate + 1))
        x_reintegrated[:, 0] = np.hstack(
            (self.q_filtered[:, frames_to_reintegrate.start], self.qdot[:, frames_to_reintegrate.start])
        )
        for i, i_node in enumerate(frames_to_reintegrate):
            q = x_reintegrated[: self.biorbd_model.nbQ(), i]
            qdot = x_reintegrated[self.biorbd_model.nbQ() :, i]
            u = self.tau[:, i_node]
            f_ext_biorbd = self.get_f_ext_at_frame(i_node)
            dqdot = self.biorbd_model.ForwardDynamics(q, qdot, u, f_ext_biorbd).to_array()
            dx = np.hstack((qdot, dqdot))
            x_reintegrated[:, i + 1] = x_reintegrated[:, i] + dt * dx

        self.q_reintegrated = np.zeros_like(self.q_filtered)
        self.q_reintegrated[:, list(frames_to_reintegrate)] = x_reintegrated[: self.biorbd_model.nbQ(), :-1]

    def animate_dynamics(self):
        """
        Animate the dynamics reconstruction.
        """
        try:
            from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers
        except:
            raise RuntimeError("To animate the dynamics, you must install Pyorerun.")

        # Add the model
        model = BiorbdModel.from_biorbd_object(self.biorbd_model)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        model.options.show_marker_labels = False
        model.options.show_center_of_mass_labels = False

        viz = PhaseRerun(self.kinematics_reconstructor.t[:200])

        # Add experimental markers
        marker_names = [m.to_string() for m in self.biorbd_model.markerNames()]
        markers = PyoMarkers(data=self.kinematics_reconstructor.markers[:, :, :200], channels=marker_names, show_labels=False)

        # Add force plates to the animation
        force_plate_idx = Operator.from_marker_frame_to_analog_frame(
            self.experimental_data.analogs_time_vector,
            self.experimental_data.markers_time_vector,
            list(self.kinematics_reconstructor.frame_range),
        )[:200]
        viz.add_force_plate(num=0, corners=self.experimental_data.platform_corners[0])
        viz.add_force_plate(num=1, corners=self.experimental_data.platform_corners[1])
        viz.add_force_data(
            num=0,
            force_origin=self.experimental_data.f_ext_sorted_filtered[0, :3, force_plate_idx].T,
            force_vector=self.experimental_data.f_ext_sorted_filtered[0, 6:9, force_plate_idx].T,
        )
        viz.add_force_data(
            num=1,
            force_origin=self.experimental_data.f_ext_sorted_filtered[1, :3, force_plate_idx].T,
            force_vector=self.experimental_data.f_ext_sorted_filtered[1, 6:9, force_plate_idx].T,
        )

        # Add the kinematics
        viz.add_animated_model(model, self.q_filtered[:, :200], tracked_markers=markers)

        # Add the reintegration of the dynamics
        viz.add_animated_model(model, self.q_reintegrated[:, :200])

        # Play
        viz.rerun_by_frame("Dynamics reconstruction")

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        result_file_full_path = f"{result_folder}/inv_dyn_{trial_name}.pkl"
        return result_file_full_path

    def save_inverse_dynamics(self):
        """
        Save the inverse dynamics results.
        """
        if self.q_reintegrated is None:
            self.q_reintegrated = 0
        result_file_full_path = self.get_result_file_full_path()
        with open(result_file_full_path, "wb") as file:
            pickle.dump(self.outputs(), file)

    def inputs(self):
        return {
            "biorbd_model": self.biorbd_model,
            "experimental_data": self.experimental_data,
            "q_filtered": self.q_filtered,
            "qdot": self.qdot,
            "qddot": self.qddot,
        }

    def outputs(self):
        return {
            "tau": self.tau,
            "q_reintegrated": self.q_reintegrated,
            "is_loaded_inverse_dynamics": self.is_loaded_inverse_dynamics,
        }
