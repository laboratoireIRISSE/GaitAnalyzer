import os
import pickle
import numpy as np
import casadi as cas
import biorbd
import biobuddy

try:
    import bioptim
except ImportError:
    print("Skipped Bioptim import as it is not installed")

from gait_analyzer.operator import Operator
from gait_analyzer.model_creator import ModelCreator
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor
from gait_analyzer.inverse_dynamics_performer import InverseDynamicsPerformer
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.events.cyclic_events import CyclicEvents
from gait_analyzer.subject import Subject


class OptimalEstimator:
    """
    This class creates one optimal control problem per cycle and solves each one.
    The goal is to match as closely as possible the experimental data.
    Muscle forces are computed for each cycle individually, then averaged.
    This preserves inter-cycle variability up to the final biomechanical output.
    """

    def __init__(
            self,
            cycles_to_analyze: int | list,
            subject: Subject,
            model_creator: ModelCreator,
            experimental_data: ExperimentalData,
            events: CyclicEvents,
            kinematics_reconstructor: KinematicsReconstructor,
            inverse_dynamic_performer: InverseDynamicsPerformer,
            gait_parameters_all: dict,
            plot_solution_flag: bool,
            animate_solution_flag: bool,
            skip_if_existing: bool,
    ):
        """
        Parameters
        ----------
        cycles_to_analyze: int | list
            The cycle(s) to analyze. One OCP is solved per cycle, then muscle forces are averaged.
        gait_parameters_all: dict
            The gait parameters computed by extract_gait_parameters, used to get consistent cycle indices.
        """

        # Checks
        if not isinstance(cycles_to_analyze, (int, list)):
            raise ValueError("cycles_to_analyze must be an int or a list of int")
        if not isinstance(subject, Subject):
            raise ValueError("subject must be a Subject")
        if not isinstance(model_creator, ModelCreator):
            raise ValueError("model_creator must be a ModelCreator")
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError("experimental_data must be an ExperimentalData")
        if not isinstance(events, CyclicEvents):
            raise ValueError("events must be a CyclicEvents")
        if not isinstance(kinematics_reconstructor, KinematicsReconstructor):
            raise ValueError("kinematics_reconstructor must be a KinematicsReconstructor")
        if not isinstance(inverse_dynamic_performer, InverseDynamicsPerformer):
            raise ValueError("inverse_dynamic_performer must be a InverseDynamicsPerformer")

        # Initial attributes
        self.cycles_to_analyze = [cycles_to_analyze] if isinstance(cycles_to_analyze, int) else cycles_to_analyze
        self.subject = subject
        self.model_creator = model_creator
        self.experimental_data = experimental_data
        self.events = events
        self.kinematics_reconstructor = kinematics_reconstructor
        self.inverse_dynamic_performer = inverse_dynamic_performer
        self.gait_parameters_all = gait_parameters_all

        # Extended attributes — per-cycle results
        self.model_ocp = None
        self.n_shooting = None
        # Individual cycle data (input to each OCP), shape (n_cycles, ..., 101)
        self.q_cycles = None           # (n_cycles, n_dof, 101)
        self.qdot_cycles = None        # (n_cycles, n_dof, 101)
        self.tau_cycles = None         # (n_cycles, n_dof, 101)
        self.f_ext_l_cycles = None     # (n_cycles, 9, 101)
        self.f_ext_r_cycles = None     # (n_cycles, 9, 101)
        self.emg_cycles = None         # (n_cycles, n_muscles, 101)
        self.markers_cycles = None     # (n_cycles, 3, n_markers, 101)
        self.fz_left_cycles = None     # (n_cycles, 101)
        self.fz_right_cycles = None    # (n_cycles, 101)
        self.phase_time_cycles = None  # (n_cycles,)
        # Per-cycle OCP solutions
        self.q_opt_cycles = None           # (n_cycles, n_dof, 101)
        self.qdot_opt_cycles = None        # (n_cycles, n_dof, 101)
        self.tau_opt_cycles = None         # (n_cycles, n_dof, 100)
        self.muscles_opt_cycles = None     # (n_cycles, n_muscles, 100)
        self.f_ext_value_opt_cycles = None     # (n_cycles, 6, 100)
        self.f_ext_position_opt_cycles = None  # (n_cycles, 6, 100)
        self.opt_status_cycles = None      # list of str
        self.muscle_forces_cycles = None   # (n_cycles, n_muscles, 100)
        self.muscle_names = None
        # Final averaged outputs
        self.muscle_forces = None          # (n_muscles, 100) — mean over cycles
        self.q_opt = None                  # (n_dof, 101) — mean over cycles
        self.qdot_opt = None               # (n_dof, 101)
        self.tau_opt = None                # (n_dof, 100)
        self.muscles_opt = None            # (n_muscles, 100)
        self.opt_status = "CVG"
        self.is_loaded_optimal_solution = False

        # Execution
        if skip_if_existing and self.check_if_existing():
            print("Optimal estimation already exists, skipping...")
            self.is_loaded_optimal_solution = True
        else:
            print("Performing optimal estimation...")

            self.generate_no_contacts_model()
            self.prepare_experimental_data_per_cycle()
            self.solve_all_cycles(with_residual_forces=True)
            self.average_muscle_forces()
            self.save_optimal_reconstruction()

            if plot_solution_flag:
                print("Plotting per-cycle solutions is not supported in per-cycle mode.")

        if animate_solution_flag:
            self.animate_solution()

    # ------------------------------------------------------------------ #
    # Model generation                                                     #
    # ------------------------------------------------------------------ #

    def generate_no_contacts_model(self):
        segments_to_remove_dofs_from = [
            "toes_r_rotation_transform",
            "toes_l_rotation_transform",
            "lunate_r_rotation_transform",
            "hand_r_rotation_transform",
            "fingers_r_rotation_transform",
            "lunate_l_rotation_transform",
            "hand_l_rotation_transform",
            "fingers_l_rotation_transform",
        ]

        no_contact_model = biobuddy.BiomechanicalModelReal().from_biomod(self.model_creator.biorbd_model_full_path)
        for segment in no_contact_model.segments:
            if segment.name in segments_to_remove_dofs_from:
                segment.rotations = biobuddy.Rotations.NONE
                segment.translations = biobuddy.Translations.NONE
                segment.dof_names = None
                segment.q_ranges = None
                segment.qdot_ranges = None

        no_contact_model.to_biomod(self.model_creator.biorbd_model_full_path.replace(".bioMod", "_no_contacts.bioMod"))

    # ------------------------------------------------------------------ #
    # Experimental data extraction — one entry per cycle                   #
    # ------------------------------------------------------------------ #

    def prepare_experimental_data_per_cycle(self):
        """
        Extract and interpolate experimental data for each cycle individually.
        No averaging is done here — each cycle's data will feed its own OCP.
        """

        self.model_ocp = self.model_creator.biorbd_model_full_path.replace(".bioMod", "_no_contacts.bioMod")
        model = biorbd.Model(self.model_ocp)

        # ------------------------------------------------------------------ #
        # Cycle indices (right leg only — standard biomechanical convention)  #
        # ------------------------------------------------------------------ #
        idx_deb_right = self.gait_parameters_all["right_leg"]["idx_deb"]
        nb_cycles_available = len(idx_deb_right) - 1

        for c in self.cycles_to_analyze:
            if c >= nb_cycles_available:
                raise RuntimeError(
                    f"cycles_to_analyze={c} is out of range. "
                    f"Only {nb_cycles_available} cycles available (0 to {nb_cycles_available - 1})."
                )

        # ------------------------------------------------------------------ #
        # Helper functions                                                    #
        # ------------------------------------------------------------------ #
        def analog_to_marker(analog_idx):
            return Operator.from_analog_frame_to_marker_frame(
                analogs_time_vector=self.experimental_data.analogs_time_vector,
                markers_time_vector=self.experimental_data.markers_time_vector,
                analog_idx=[analog_idx],
            )[0]

        def marker_to_analog(marker_idx):
            return Operator.from_marker_frame_to_analog_frame(
                analogs_time_vector=self.experimental_data.analogs_time_vector,
                markers_time_vector=self.experimental_data.markers_time_vector,
                marker_idx=int(marker_idx),
            )

        def interp_to_ref(data_2d, n_target):
            n_src = data_2d.shape[1]
            x_src = np.linspace(0, 1, n_src)
            x_tgt = np.linspace(0, 1, n_target)
            return np.array([np.interp(x_tgt, x_src, data_2d[i, :]) for i in range(data_2d.shape[0])])

        # ------------------------------------------------------------------ #
        # DoFs and muscles                                                    #
        # ------------------------------------------------------------------ #
        dof_idx_to_keep = np.array(
            [
                0,   # Pelvis trans X
                1,   # Pelvis trans Y
                2,   # Pelvis trans Z
                3,   # Pelvis rot X
                4,   # Pelvis rot Y
                5,   # Pelvis rot Z
                6,   # femur_r rot X
                7,   # femur_r rot Y
                8,   # femur_r rot Z
                9,   # tibia_r rot X
                10,  # talus_r rot X
                11,  # calc_r rot X
                13,  # femur_l rot X
                14,  # femur_l rot Y
                15,  # femur_l rot Z
                16,  # tibia_l rot X
                17,  # talus_l rot X
                18,  # calc_l rot X
                20,  # thorax rot X
                21,  # thorax rot Y
                22,  # thorax rot Z
                23,  # head_and_neck rot X
                24,  # head_and_neck rot Y
                25,  # head_and_neck rot Z
                26,  # humerus_r rot X
                27,  # humerus_r rot Y
                28,  # humerus_r rot Z
                29,  # ulna_r rot X
                30,  # radius_r rot X
                34,  # humerus_l rot X
                35,  # humerus_l rot Y
                36,  # humerus_l rot Z
                37,  # ulna_l rot X
                38,  # radius_l rot X
            ]
        )

        muscle_names = [m.to_string() for m in model.muscleNames()]
        nb_muscles = len(muscle_names)
        n_analogs = self.experimental_data.f_ext_sorted.shape[2]
        n_frames_ref = 101
        self.n_shooting = n_frames_ref - 1
        self.muscle_names = muscle_names

        print(f"------------------ n_frames_ref = {n_frames_ref}, {len(self.cycles_to_analyze)} cycle(s) to solve ------------------")

        # ------------------------------------------------------------------ #
        # Accumulate per-cycle arrays                                         #
        # ------------------------------------------------------------------ #
        q_cycles        = []
        qdot_cycles     = []
        tau_cycles      = []
        f_ext_l_cycles  = []
        f_ext_r_cycles  = []
        emg_cycles      = []
        markers_cycles  = []
        fz_left_cycles  = []
        fz_right_cycles = []
        phase_time_cycles = []

        for c in self.cycles_to_analyze:
            print(f"  Extracting cycle {c}...")

            start_a = idx_deb_right[c]
            end_a   = idx_deb_right[c + 1]
            start_m = analog_to_marker(start_a)
            end_m   = analog_to_marker(end_a)

            idx_cycle        = np.arange(start_m, end_m)
            index_filtered_q = idx_cycle - self.kinematics_reconstructor.frame_range.start

            q_cycles.append(interp_to_ref(
                self.kinematics_reconstructor.q_filtered[np.ix_(dof_idx_to_keep, index_filtered_q)],
                n_frames_ref,
            ))
            qdot_cycles.append(interp_to_ref(
                self.kinematics_reconstructor.qdot[np.ix_(dof_idx_to_keep, index_filtered_q)],
                n_frames_ref,
            ))
            tau_cycles.append(interp_to_ref(
                self.inverse_dynamic_performer.tau[np.ix_(dof_idx_to_keep, index_filtered_q)],
                n_frames_ref,
            ))

            f_ext_l    = np.zeros((9, n_frames_ref))
            f_ext_r    = np.zeros((9, n_frames_ref))
            emg        = np.zeros((nb_muscles, n_frames_ref))
            idx_frames = np.round(np.linspace(start_m, end_m - 1, n_frames_ref)).astype(int)

            for i_frame, marker_frame in enumerate(idx_frames):
                idx_analogs = marker_to_analog(marker_frame)
                idx_low     = max(0, idx_analogs - 5)
                idx_high    = min(n_analogs, idx_analogs + 5)

                f_ext_l[:, i_frame] = np.mean(
                    self.experimental_data.f_ext_sorted[0, :, idx_low:idx_high], axis=1
                )
                f_ext_r[:, i_frame] = np.mean(
                    self.experimental_data.f_ext_sorted[1, :, idx_low:idx_high], axis=1
                )

                for i_muscle, muscle_name in enumerate(muscle_names):
                    if muscle_name in self.model_creator.osim_model_type.muscle_name_mapping:
                        muscle_pseudo = self.model_creator.osim_model_type.muscle_name_mapping[muscle_name]
                        if muscle_pseudo is not None:
                            muscle_index = self.experimental_data.analog_names.index(muscle_pseudo)
                            emg[i_muscle, i_frame] = np.nanmean(
                                self.experimental_data.normalized_emg[muscle_index, idx_low:idx_high]
                            )

            # EMG left leg: half-cycle shift from right leg
            frame_index_shifted = list(range(n_frames_ref))
            frame_index_shifted[0:int(np.floor(n_frames_ref / 2))] = list(
                range(int(np.ceil(n_frames_ref / 2)), n_frames_ref)
            )
            frame_index_shifted[int(np.floor(n_frames_ref / 2)):] = list(
                range(0, int(np.ceil(n_frames_ref / 2)))
            )
            for i_muscle, muscle_name in enumerate(muscle_names):
                if muscle_name in self.model_creator.osim_model_type.muscle_name_mapping:
                    muscle_pseudo = self.model_creator.osim_model_type.muscle_name_mapping[muscle_name]
                    if muscle_pseudo is not None:
                        i_muscle_l = muscle_names.index(muscle_name.replace("_r", "_l"))
                        emg[i_muscle_l, :] = emg[i_muscle, frame_index_shifted]

            f_ext_l_cycles.append(f_ext_l)
            f_ext_r_cycles.append(f_ext_r)
            emg_cycles.append(emg)

            fz_left_raw  = self.experimental_data.f_ext_sorted_filtered[0, 8, start_a:end_a]
            fz_right_raw = self.experimental_data.f_ext_sorted_filtered[1, 8, start_a:end_a]
            fz_left_cycles.append(np.interp(
                np.linspace(0, 1, n_frames_ref), np.linspace(0, 1, len(fz_left_raw)), fz_left_raw,
            ))
            fz_right_cycles.append(np.interp(
                np.linspace(0, 1, n_frames_ref), np.linspace(0, 1, len(fz_right_raw)), fz_right_raw,
            ))

            markers_cycle  = self.experimental_data.markers_sorted[:, :, idx_cycle]
            n_coords, n_markers, _ = markers_cycle.shape
            markers_interp = np.zeros((n_coords, n_markers, n_frames_ref))
            for i_coord in range(n_coords):
                markers_interp[i_coord] = interp_to_ref(markers_cycle[i_coord], n_frames_ref)
            markers_cycles.append(markers_interp)

            phase_time_cycles.append(
                self.experimental_data.markers_time_vector[end_m - 1]
                - self.experimental_data.markers_time_vector[start_m]
            )

        # Store all cycles
        self.q_cycles         = np.array(q_cycles)           # (n_cycles, n_dof, 101)
        self.qdot_cycles      = np.array(qdot_cycles)        # (n_cycles, n_dof, 101)
        self.tau_cycles       = np.array(tau_cycles)         # (n_cycles, n_dof, 101)
        self.f_ext_l_cycles   = np.array(f_ext_l_cycles)     # (n_cycles, 9, 101)
        self.f_ext_r_cycles   = np.array(f_ext_r_cycles)     # (n_cycles, 9, 101)
        self.emg_cycles       = np.array(emg_cycles)         # (n_cycles, n_muscles, 101)
        self.markers_cycles   = np.array(markers_cycles)     # (n_cycles, 3, n_markers, 101)
        self.fz_left_cycles   = np.array(fz_left_cycles)     # (n_cycles, 101)
        self.fz_right_cycles  = np.array(fz_right_cycles)    # (n_cycles, 101)
        self.phase_time_cycles = np.array(phase_time_cycles) # (n_cycles,)

    # ------------------------------------------------------------------ #
    # Solve one OCP per cycle                                              #
    # ------------------------------------------------------------------ #

    def solve_all_cycles(self, with_residual_forces: bool = False):
        """Solve one OCP per cycle and store per-cycle solutions."""

        n_cycles = len(self.cycles_to_analyze)
        model = biorbd.Model(self.model_ocp)
        nb_q       = self.q_cycles.shape[1]
        nb_muscles = len(self.muscle_names)

        self.q_opt_cycles              = np.zeros((n_cycles, nb_q,       self.n_shooting + 1))
        self.qdot_opt_cycles           = np.zeros((n_cycles, nb_q,       self.n_shooting + 1))
        self.tau_opt_cycles            = np.zeros((n_cycles, nb_q,       self.n_shooting))
        self.muscles_opt_cycles        = np.zeros((n_cycles, nb_muscles, self.n_shooting))
        self.f_ext_value_opt_cycles    = np.zeros((n_cycles, 6,          self.n_shooting))
        self.f_ext_position_opt_cycles = np.zeros((n_cycles, 6,          self.n_shooting))
        self.muscle_forces_cycles      = np.zeros((n_cycles, nb_muscles, self.n_shooting))
        self.opt_status_cycles         = []

        for i_cycle, c in enumerate(self.cycles_to_analyze):
            print(f"\n========== Solving OCP for cycle {c} ({i_cycle + 1}/{n_cycles}) ==========")

            f_ext_exp = {
                "left_leg":  self.f_ext_l_cycles[i_cycle],
                "right_leg": self.f_ext_r_cycles[i_cycle],
            }

            (q_opt, qdot_opt, tau_opt, muscles_opt,
             f_ext_value_opt, f_ext_position_opt, status) = self._solve_single_cycle(
                q_exp=self.q_cycles[i_cycle],
                qdot_exp=self.qdot_cycles[i_cycle],
                tau_exp=self.tau_cycles[i_cycle],
                emg_exp=self.emg_cycles[i_cycle],
                markers_exp=self.markers_cycles[i_cycle],
                f_ext_exp=f_ext_exp,
                phase_time=self.phase_time_cycles[i_cycle],
                with_residual_forces=with_residual_forces,
            )

            self.q_opt_cycles[i_cycle]              = q_opt
            self.qdot_opt_cycles[i_cycle]           = qdot_opt
            self.tau_opt_cycles[i_cycle]            = tau_opt
            self.muscles_opt_cycles[i_cycle]        = muscles_opt
            self.f_ext_value_opt_cycles[i_cycle]    = f_ext_value_opt
            self.f_ext_position_opt_cycles[i_cycle] = f_ext_position_opt
            self.opt_status_cycles.append(status)
            self.muscle_forces_cycles[i_cycle]      = self._compute_muscle_forces(
                q_opt, qdot_opt, muscles_opt
            )

        # Warn if any cycle diverged
        diverged = [self.cycles_to_analyze[i] for i, s in enumerate(self.opt_status_cycles) if s == "DVG"]
        if diverged:
            print(f"WARNING: cycles {diverged} diverged and will still be included in the average.")

        self.opt_status = "CVG" if not diverged else "DVG"

    def _solve_single_cycle(
        self,
        q_exp, qdot_exp, tau_exp, emg_exp, markers_exp, f_ext_exp,
        phase_time, with_residual_forces,
    ):
        """Build and solve one OCP for a single cycle. Returns the solution arrays."""

        try:
            from bioptim import (
                MusclesBiorbdModel,
                DynamicsFunctions,
                DynamicsEvaluation,
                InitialGuessList,
                InterpolationType,
                ObjectiveFcn,
                ObjectiveList,
                OptimalControlProgram,
                PhaseDynamics,
                BoundsList,
                ConstraintList,
                OdeSolver,
                ExternalForceSetTimeSeries,
                Node,
                DynamicsOptionsList,
                SolutionMerge,
                TimeAlignment,
                Solver,
                ConfigureVariables,
            )
        except Exception:
            raise RuntimeError("To reconstruct optimally, you must install Bioptim")

        class CustomMuscleModelNoContacts(MusclesBiorbdModel):
            def __init__(self_, biorbd_model_path, external_force_set=None, with_residual_torque=True):
                super().__init__(
                    biorbd_model_path,
                    external_force_set=external_force_set,
                    with_residual_torque=with_residual_torque,
                )
                if with_residual_forces:
                    self_.control_configuration += [
                        lambda ocp, nlp, as_states, as_controls,
                               as_algebraic_states: ConfigureVariables.configure_translational_forces(
                            ocp, nlp, as_states=False, as_controls=True, as_algebraic_states=False, n_contacts=2
                        )
                    ]

            def dynamics(self_, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
                q    = DynamicsFunctions.get(nlp.states["q"], states)
                qdot = DynamicsFunctions.get(nlp.states["qdot"], states)

                tau_residual    = DynamicsFunctions.get(nlp.controls["tau"], controls)
                mus_activations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
                tau = tau_residual + DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, None)

                if with_residual_forces:
                    f_ext_residual_value    = DynamicsFunctions.get(nlp.controls["contact_forces"], controls)
                    f_ext_residual_position = DynamicsFunctions.get(nlp.controls["contact_positions"], controls)

                external_forces = nlp.get_external_forces(
                    "external_forces", states, controls, algebraic_states, numerical_timeseries
                )
                if with_residual_forces:
                    external_forces[:3]    += f_ext_residual_position[:3]
                    external_forces[6:9]   += f_ext_residual_value[:3]
                    external_forces[9:12]  += f_ext_residual_position[3:6]
                    external_forces[15:18] += f_ext_residual_value[3:6]

                ddq = nlp.model.forward_dynamics()(q, qdot, tau, external_forces, nlp.parameters.cx)
                return DynamicsEvaluation(dxdt=cas.vertcat(qdot, ddq), defects=None)

        external_force_set = ExternalForceSetTimeSeries(nb_frames=self.n_shooting)
        external_force_set.add(
            force_name="calcn_l",
            segment="calcn_l",
            values=f_ext_exp["left_leg"][3:9, :-1],
            point_of_application=f_ext_exp["left_leg"][:3, :-1],
        )
        external_force_set.add(
            force_name="calcn_r",
            segment="calcn_r",
            values=f_ext_exp["right_leg"][3:9, :-1],
            point_of_application=f_ext_exp["right_leg"][:3, :-1],
        )
        numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}

        bio_model = CustomMuscleModelNoContacts(self.model_ocp, external_force_set=external_force_set)
        nb_q       = bio_model.nb_q
        nb_muscles = bio_model.nb_muscles

        r_foot_marker_index = np.array([
            bio_model.marker_index("RCAL"),
            bio_model.marker_index("RMFH1"),
            bio_model.marker_index("RMFH5"),
            bio_model.marker_index("R_foot_up"),
        ])
        l_foot_marker_index = np.array([
            bio_model.marker_index("LCAL"),
            bio_model.marker_index("LMFH1"),
            bio_model.marker_index("LMFH5"),
            bio_model.marker_index("L_foot_up"),
        ])

        objective_functions = ObjectiveList()
        objective_functions.add(objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.001)
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, index=[0, 1, 2, 3, 4, 5]
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="muscles",
            weight=10,
            target=emg_exp[:, :-1],
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=100.0, node=Node.ALL, target=markers_exp
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_MARKERS,
            weight=1000.0,
            node=Node.ALL,
            marker_index=["RCAL", "RMFH1", "RMFH5", "R_foot_up", "LCAL", "LMFH1", "LMFH5", "L_foot_up"],
            target=markers_exp[:, np.hstack((r_foot_marker_index, l_foot_marker_index)), :],
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=1.0, node=Node.ALL, target=q_exp
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_STATE, key="qdot", node=Node.ALL, weight=0.01, target=qdot_exp
        )
        if with_residual_forces:
            objective_functions.add(
                objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                key="contact_forces",
                node=Node.ALL_SHOOTING,
                weight=10,
            )
            objective_functions.add(
                objective=ObjectiveFcn.Lagrange.TRACK_CONTROL,
                key="contact_positions",
                node=Node.ALL_SHOOTING,
                weight=0.01,
                target=np.vstack((f_ext_exp["left_leg"][0:3, :-1], f_ext_exp["right_leg"][0:3, :-1])),
            )

        dynamics = DynamicsOptionsList()
        dynamics.add(
            numerical_data_timeseries=numerical_time_series,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            ode_solver=OdeSolver.RK4(),
        )

        x_bounds = BoundsList()
        min_q = q_exp[:, :] - 0.3
        min_q[:6, :] = q_exp[:6, :] - 0.05
        max_q = q_exp[:, :] + 0.3
        max_q[:6, :] = q_exp[:6, :] + 0.05
        x_bounds.add("q", min_bound=min_q, max_bound=max_q, interpolation=InterpolationType.EACH_FRAME)
        x_bounds.add(
            "qdot",
            min_bound=qdot_exp - 10,
            max_bound=qdot_exp + 10,
            interpolation=InterpolationType.EACH_FRAME,
        )

        x_init = InitialGuessList()
        x_init.add("q",    initial_guess=q_exp,    interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot", initial_guess=qdot_exp, interpolation=InterpolationType.EACH_FRAME)

        u_bounds = BoundsList()
        u_bounds.add("tau",     min_bound=[-800] * nb_q,         max_bound=[800] * nb_q,         interpolation=InterpolationType.CONSTANT)
        u_bounds.add("muscles", min_bound=[0.0001] * nb_muscles, max_bound=[1.0] * nb_muscles,   interpolation=InterpolationType.CONSTANT)
        if with_residual_forces:
            u_bounds.add("contact_forces",    min_bound=[-100] * 6,                  max_bound=[100] * 6,                  interpolation=InterpolationType.CONSTANT)
            u_bounds.add("contact_positions", min_bound=[-2, -2, 0.0, -2, -2, 0.0], max_bound=[2, 2, 0.005, 2, 2, 0.005], interpolation=InterpolationType.CONSTANT)

        u_init = InitialGuessList()
        u_init.add("tau",     initial_guess=tau_exp[:, :-1],  interpolation=InterpolationType.EACH_FRAME)
        u_init.add("muscles", initial_guess=emg_exp[:, :-1],  interpolation=InterpolationType.EACH_FRAME)
        if with_residual_forces:
            u_init.add("contact_forces", initial_guess=[0] * 6, interpolation=InterpolationType.CONSTANT)
            u_init.add(
                "contact_positions",
                initial_guess=np.vstack((f_ext_exp["left_leg"][0:3, :-1], f_ext_exp["right_leg"][0:3, :-1])),
                interpolation=InterpolationType.EACH_FRAME,
            )

        ocp = OptimalControlProgram(
            bio_model=bio_model,
            n_shooting=self.n_shooting,
            phase_time=phase_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            constraints=ConstraintList(),
            use_sx=False,
            n_threads=10,
        )
        ocp.add_plot_penalty()
        ocp.add_plot_ipopt_outputs()

        solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
        solver.set_linear_solver("ma57")
        solver.set_maximum_iterations(1000)
        solver.set_tol(1e-3)
        solution = ocp.solve(solver=solver)

        q_opt       = solution.decision_states(to_merge=SolutionMerge.NODES)["q"]
        qdot_opt    = solution.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
        tau_opt     = solution.decision_controls(to_merge=SolutionMerge.NODES)["tau"]
        muscles_opt = solution.decision_controls(to_merge=SolutionMerge.NODES)["muscles"]

        if with_residual_forces:
            f_ext_value_opt    = solution.decision_controls(to_merge=SolutionMerge.NODES)["contact_forces"]
            f_ext_position_opt = solution.decision_controls(to_merge=SolutionMerge.NODES)["contact_positions"]
        else:
            f_ext_value_opt    = np.zeros((6, self.n_shooting))
            f_ext_position_opt = np.zeros((6, self.n_shooting))

        status = "CVG" if solution.status == 0 else "DVG"
        return q_opt, qdot_opt, tau_opt, muscles_opt, f_ext_value_opt, f_ext_position_opt, status

    # ------------------------------------------------------------------ #
    # Muscle force extraction and averaging                                #
    # ------------------------------------------------------------------ #

    def _compute_muscle_forces(self, q_opt, qdot_opt, muscles_opt):
        """Compute muscle forces for a single cycle solution."""
        model = biorbd.Model(self.model_ocp)
        nb_muscles = model.nbMuscles()
        muscle_forces = np.zeros((nb_muscles, self.n_shooting))
        for i_frame in range(self.n_shooting):
            muscles = model.stateSet()
            for i_muscle, muscle in enumerate(muscles):
                muscle.setActivation(muscles_opt[i_muscle, i_frame])
            muscle_forces[:, i_frame] = model.muscleForces(
                muscles, q_opt[:, i_frame], qdot_opt[:, i_frame]
            ).to_array()
        return muscle_forces

    def average_muscle_forces(self):
        """Average per-cycle results to produce final outputs."""
        print(f"\n  Averaging muscle forces over {len(self.cycles_to_analyze)} cycle(s)...")
        self.muscle_forces = np.mean(self.muscle_forces_cycles, axis=0)  # (n_muscles, 100)
        self.q_opt         = np.mean(self.q_opt_cycles,         axis=0)  # (n_dof, 101)
        self.qdot_opt      = np.mean(self.qdot_opt_cycles,      axis=0)  # (n_dof, 101)
        self.tau_opt       = np.mean(self.tau_opt_cycles,       axis=0)  # (n_dof, 100)
        self.muscles_opt   = np.mean(self.muscles_opt_cycles,   axis=0)  # (n_muscles, 100)

    # ------------------------------------------------------------------ #
    # Animation                                                            #
    # ------------------------------------------------------------------ #

    def animate_solution(self):
        try:
            from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers, PyoMuscles
        except Exception:
            raise RuntimeError("To animate the optimal solution, you must install Pyorerun.")

        # Animate the averaged solution using mean markers and mean f_ext
        mean_markers = np.mean(self.markers_cycles, axis=0)
        mean_f_ext_l = np.mean(self.f_ext_l_cycles, axis=0)
        mean_f_ext_r = np.mean(self.f_ext_r_cycles, axis=0)
        mean_phase_time = float(np.mean(self.phase_time_cycles))

        model_viz = BiorbdModel(self.model_ocp)
        model_viz.options.transparent_mesh = False
        model_viz.options.show_gravity = True

        viz = PhaseRerun(np.linspace(0, mean_phase_time, self.n_shooting + 1))
        markers = PyoMarkers(data=mean_markers, marker_names=list(model_viz.marker_names), show_labels=False)
        nb_muscles = len(model_viz.muscle_names)
        emgs = PyoMuscles(
            data=np.hstack((self.muscles_opt, np.zeros((nb_muscles, 1)))),
            muscle_names=list(model_viz.muscle_names),
            mvc=np.ones((nb_muscles, 1)),
        )

        viz.add_force_plate(num=1, corners=self.experimental_data.platform_corners[0])
        viz.add_force_plate(num=2, corners=self.experimental_data.platform_corners[1])
        viz.add_force_data(num=1, force_origin=mean_f_ext_l[:3, :], force_vector=mean_f_ext_l[6:9, :])
        viz.add_force_data(num=2, force_origin=mean_f_ext_r[:3, :], force_vector=mean_f_ext_r[6:9, :])
        viz.add_animated_model(model_viz, self.q_opt, tracked_markers=markers, muscle_activations_intensity=emgs)
        viz.rerun("OCP averaged solution")

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def check_if_existing(self):
        """Check if the optimal estimation already exists and load it if so."""
        result_file_full_path = self.get_result_file_full_path()
        if os.path.exists(result_file_full_path):
            with open(result_file_full_path, "rb") as file:
                data = pickle.load(file)
                self.model_ocp                  = data["model_ocp"]
                self.n_shooting                 = data["n_shooting"]
                self.muscle_names               = data["muscle_names"]
                # Per-cycle experimental data
                self.q_cycles                   = data["q_cycles"]
                self.qdot_cycles                = data["qdot_cycles"]
                self.tau_cycles                 = data["tau_cycles"]
                self.f_ext_l_cycles             = data["f_ext_l_cycles"]
                self.f_ext_r_cycles             = data["f_ext_r_cycles"]
                self.emg_cycles                 = data["emg_cycles"]
                self.markers_cycles             = data["markers_cycles"]
                self.fz_left_cycles             = data["fz_left_cycles"]
                self.fz_right_cycles            = data["fz_right_cycles"]
                self.phase_time_cycles          = data["phase_time_cycles"]
                # Per-cycle OCP solutions
                self.q_opt_cycles               = data["q_opt_cycles"]
                self.qdot_opt_cycles            = data["qdot_opt_cycles"]
                self.tau_opt_cycles             = data["tau_opt_cycles"]
                self.muscles_opt_cycles         = data["muscles_opt_cycles"]
                self.f_ext_value_opt_cycles     = data["f_ext_value_opt_cycles"]
                self.f_ext_position_opt_cycles  = data["f_ext_position_opt_cycles"]
                self.opt_status_cycles          = data["opt_status_cycles"]
                self.muscle_forces_cycles       = data["muscle_forces_cycles"]
                # Averaged outputs
                self.muscle_forces              = data["muscle_forces"]
                self.q_opt                      = data["q_opt"]
                self.qdot_opt                   = data["qdot_opt"]
                self.tau_opt                    = data["tau_opt"]
                self.muscles_opt                = data["muscles_opt"]
                self.opt_status                 = data["opt_status"]
            return True
        return False

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        return f"{result_folder}/optim_estim_{trial_name}_{self.opt_status}.pkl"

    def save_optimal_reconstruction(self):
        result_file_full_path = self.get_result_file_full_path()
        with open(result_file_full_path, "wb") as file:
            pickle.dump(self.outputs(), file)

    # ------------------------------------------------------------------ #
    # Inputs / Outputs                                                     #
    # ------------------------------------------------------------------ #

    def inputs(self):
        return {
            "cycles_to_analyze":        self.cycles_to_analyze,
            "biorbd_model_path":        self.model_creator.biorbd_model_full_path,
            "experimental_data":        self.experimental_data,
            "events":                   self.events,
            "kinematics_reconstructor": self.kinematics_reconstructor,
        }

    def outputs(self):
        return {
            # ---- Model ----
            "model_ocp":                    self.model_ocp,
            "n_shooting":                   self.n_shooting,
            "muscle_names":                 self.muscle_names,
            # ---- Per-cycle experimental data ----
            "q_cycles":                     self.q_cycles,           # (n_cycles, n_dof, 101)
            "qdot_cycles":                  self.qdot_cycles,        # (n_cycles, n_dof, 101)
            "tau_cycles":                   self.tau_cycles,         # (n_cycles, n_dof, 101)
            "f_ext_l_cycles":               self.f_ext_l_cycles,     # (n_cycles, 9, 101)
            "f_ext_r_cycles":               self.f_ext_r_cycles,     # (n_cycles, 9, 101)
            "emg_cycles":                   self.emg_cycles,         # (n_cycles, n_muscles, 101)
            "markers_cycles":               self.markers_cycles,     # (n_cycles, 3, n_markers, 101)
            "fz_left_cycles":               self.fz_left_cycles,     # (n_cycles, 101)
            "fz_right_cycles":              self.fz_right_cycles,    # (n_cycles, 101)
            "phase_time_cycles":            self.phase_time_cycles,  # (n_cycles,)
            # ---- Per-cycle OCP solutions ----
            "q_opt_cycles":                 self.q_opt_cycles,               # (n_cycles, n_dof, 101)
            "qdot_opt_cycles":              self.qdot_opt_cycles,            # (n_cycles, n_dof, 101)
            "tau_opt_cycles":               self.tau_opt_cycles,             # (n_cycles, n_dof, 100)
            "muscles_opt_cycles":           self.muscles_opt_cycles,         # (n_cycles, n_muscles, 100)
            "f_ext_value_opt_cycles":       self.f_ext_value_opt_cycles,     # (n_cycles, 6, 100)
            "f_ext_position_opt_cycles":    self.f_ext_position_opt_cycles,  # (n_cycles, 6, 100)
            "opt_status_cycles":            self.opt_status_cycles,          # list of str
            "muscle_forces_cycles":         self.muscle_forces_cycles,       # (n_cycles, n_muscles, 100)
            # ---- Averaged outputs ----
            "muscle_forces":                self.muscle_forces,   # (n_muscles, 100)
            "q_opt":                        self.q_opt,           # (n_dof, 101)
            "qdot_opt":                     self.qdot_opt,        # (n_dof, 101)
            "tau_opt":                      self.tau_opt,         # (n_dof, 100)
            "muscles_opt":                  self.muscles_opt,     # (n_muscles, 100)
            "opt_status":                   self.opt_status,
        }
