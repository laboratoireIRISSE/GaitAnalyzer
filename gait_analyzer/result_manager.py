import numpy as np

from gait_analyzer.biomechanics_quantities.angular_momentum_calculator import AngularMomentumCalculator
from gait_analyzer.model_creator import ModelCreator
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.inverse_dynamics_performer import InverseDynamicsPerformer
from gait_analyzer.events.cyclic_events import CyclicEvents
from gait_analyzer.events.unique_events import UniqueEvents
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor
from gait_analyzer.optimal_estimator import OptimalEstimator
from gait_analyzer.subject import Subject, Side
from gait_analyzer.biomechanics_quantities.marginofstability_calculator import MarginofStabilityCalculation
from gait_analyzer.biomechanics_quantities.mechanical_energy_calculator import MechanicalEnergyCalculator
from gait_analyzer.biomechanics_quantities.com_mma_distance_calculator import ComMmaDistanceCalculator
from gait_analyzer.biomechanics_quantities.probability_of_instability_calculator import ProbabilityOfInstability

class ResultManager:
    """
    This class contains all the results from the gait analysis and is the main class handling all types of analysis to perform on the experimental data.
    """

    def __init__(self, subject: Subject, cycles_to_analyze: range, static_trial: str, result_folder: str, trial_full_file_path, static_trial_full_file_path):
        """
        Initialize the ResultManager.
        .
        Parameters
        ----------
        subject: Subject
            The subject to analyze
        cycles_to_analyze: range
            The range of cycles to analyze
        static_trial: str
            The full file path of the static trial ([...]_static.c3d)
        result_folder: str
            The folder where the results will be saved. It will look like result_folder/subject_name.
        """
        # Checks:
        if not isinstance(subject, Subject):
            raise ValueError("subject must be a Subject")
        if not (isinstance(cycles_to_analyze, range) or cycles_to_analyze is None):
            raise ValueError(
                "cycles_to_analyze must be a range of cycles to analyze or None if all frames should be analyzed."
            )
        if not isinstance(static_trial, str):
            raise ValueError("static_trial must be a string")
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")

        # Initial attributes
        self.subject = subject
        self.cycles_to_analyze = cycles_to_analyze
        self.result_folder = result_folder
        self.static_trial = static_trial
        self.trial_full_file_path = trial_full_file_path
        self.result_folder = result_folder
        self.static_trial_full_file_path = static_trial_full_file_path


        # Extended attributes
        self.experimental_data = None
        self.model_creator = None
        self.gait_parameters = None
        self.events = None
        self.kinematics_reconstructor = None
        self.inverse_dynamics_performer = None
        self.optimal_estimator = None
        self.angular_momentum_calculator = None
        self.marginofstability_calculator = None
        self.percentage_of_instability_calculator = None
        self.mechanical_energy_calculator = None
        self.com_mma_distance_calculator = None
        self.mechanical_energy = None
        self.gait_parameters_all = None
        self.E_pot = None
        self.E_kin = None

    def create_model(
        self,
        osim_model_type,
        skip_if_existing: bool,
        functional_trials_path: str = None,
        mvc_trials_path: str = None,
        q_regularization_weight: float | np.ndarray = 0.01,
        qdot_regularization_weight: float | np.ndarray = 0.0,
        animate_model_flag: bool = False,
        vtp_geometry_path: str = "../../Geometry_cleaned",
    ):
        """
        Create and add the biorbd model to the ResultManager
        """

        # Checks
        if self.model_creator is not None:
            raise Exception("Biorbd model already added")

        # Add ModelCreator
        self.model_creator = ModelCreator(
            subject=self.subject,
            static_trial=self.static_trial,
            functional_trials_path=functional_trials_path,
            mvc_trials_path=mvc_trials_path,
            models_result_folder=f"{self.result_folder}/models",
            osim_model_type=osim_model_type,
            q_regularization_weight=q_regularization_weight,
            qdot_regularization_weight=qdot_regularization_weight,
            skip_if_existing=skip_if_existing,
            animate_model_flag=animate_model_flag,
            vtp_geometry_path=vtp_geometry_path,
        )

    def add_experimental_data(
        self,
        c3d_file_name: str,
        markers_to_ignore: list[str] = [],
        analogs_to_ignore: list[str] = [],
        animate_c3d_flag: bool = False,
    ):

        # Checks
        if self.experimental_data is not None:
            raise Exception("Experimental data already added")
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_biorbd_model()")

        # Add experimental data
        self.experimental_data = ExperimentalData(
            c3d_file_name=c3d_file_name,
            markers_to_ignore=markers_to_ignore,
            analogs_to_ignore=analogs_to_ignore,
            result_folder=self.result_folder,
            model_creator=self.model_creator,
            animate_c3d_flag=animate_c3d_flag,
        )

    def add_cyclic_events(self, force_plate_sides: list[Side], skip_if_existing: bool, plot_phases_flag: bool = False):

        # Checks
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is not None:
            raise Exception("CyclicEvents or UniqueEvents were already added to the ResultManager")

        # Add events
        self.events = CyclicEvents(
            experimental_data=self.experimental_data,
            force_plate_sides=force_plate_sides,
            skip_if_existing=skip_if_existing,
            plot_phases_flag=plot_phases_flag,
        )

    def add_unique_events(self, skip_if_existing: bool, plot_phases_flag: bool = False):

        # Checks
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is not None:
            raise Exception("CyclicEvents or UniqueEvents were already added to the ResultManager")

        # Add events
        self.events = UniqueEvents(
            experimental_data=self.experimental_data,
            skip_if_existing=skip_if_existing,
        )

    def reconstruct_kinematics(
        self,
        reconstruction_type=None,
        skip_if_existing: bool = False,
        animate_kinematics_flag: bool = False,
        plot_kinematics_flag: bool = False,
        q_regularization_weight: float | np.ndarray = None,
        qdot_regularization_weight: float | np.ndarray = None,
    ):
        # Checks
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is None:
            raise Exception(
                "Please run the events detection first by running ResultManager.add_cyclic_events() or ResultManager.add_unique_events()"
            )
        if self.kinematics_reconstructor is not None:
            raise Exception("kinematics_reconstructor already added")

        # Reconstruct kinematics
        self.kinematics_reconstructor = KinematicsReconstructor(
            self.experimental_data,
            self.model_creator,
            self.events,
            self.cycles_to_analyze,
            reconstruction_type=reconstruction_type,
            skip_if_existing=skip_if_existing,
            animate_kinematics_flag=animate_kinematics_flag,
            plot_kinematics_flag=plot_kinematics_flag,
            q_regularization_weight=q_regularization_weight,
            qdot_regularization_weight=qdot_regularization_weight,
        )

    def perform_inverse_dynamics(
        self, skip_if_existing: bool, reintegrate_flag: bool = True, animate_dynamics_flag: bool = False
    ):
        # Checks
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.kinematics_reconstructor is None:
            raise Exception(
                "Please add the kinematics reconstructor first by running ResultManager.reconstruct_kinematics()"
            )
        if self.inverse_dynamics_performer is not None:
            raise Exception("inverse_dynamics_performer already added")

        # Perform inverse dynamics
        self.inverse_dynamics_performer = InverseDynamicsPerformer(
            self.experimental_data,
            self.kinematics_reconstructor,
            skip_if_existing=skip_if_existing,
            reintegrate_flag=reintegrate_flag,
            animate_dynamics_flag=animate_dynamics_flag,
        )

    def compute_angular_momentum(self, skip_if_existing: bool = False):
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.kinematics_reconstructor is None:
            raise Exception(
                "Please add the kinematics reconstructor first by running ResultManager.reconstruct_kinematics()"
            )
        if self.angular_momentum_calculator is not None:
            raise Exception("Angular momentum has already been calculated")

        self.angular_momentum_calculator = AngularMomentumCalculator(
            self.model_creator.biorbd_model,
            self.experimental_data,
            self.kinematics_reconstructor,
            self.subject,
            skip_if_existing=skip_if_existing,
        )

    def compute_com_mma_distance(self, skip_if_existing: bool = False):
        if self.angular_momentum_calculator is None:
            raise Exception("Compute angular momentum first")
        if self.kinematics_reconstructor is None:
            raise Exception("Compute kinematics first")

        self.com_mma_distance_calculator = ComMmaDistanceCalculator(
            angular_momentum_calculator=self.angular_momentum_calculator,
            experimental_data=self.experimental_data,
            subject=self.subject,
            q=self.kinematics_reconstructor.q_filtered,
            skip_if_existing=skip_if_existing,
        )

        self.dCoM_MMA = self.com_mma_distance_calculator.dCoM_MMA_norm

    def compute_mechanical_energy(self, skip_if_existing: bool = False, plot: bool = False):

        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.kinematics_reconstructor is None:
            raise Exception(
                "Please add the kinematics reconstructor first by running ResultManager.reconstruct_kinematics()"
            )
        if self.angular_momentum_calculator is None:
            raise Exception(
                "Please compute angular momentum first (needed for segment COM and COMdot data)"
            )

        if self.mechanical_energy_calculator is None:
            self.mechanical_energy_calculator = MechanicalEnergyCalculator(
                self.model_creator.biorbd_model,
                self.experimental_data,
                self.kinematics_reconstructor,
                self.subject,
                self.angular_momentum_calculator.segments_data,
            )

        mechanical_energy = self.mechanical_energy_calculator.compute_mechanical_energy(
            skip_if_existing=skip_if_existing
        )

        if plot:
            self.mechanical_energy_calculator.plot_energy()

        return mechanical_energy

    def compute_marginofstability(self, skip_if_existing: bool = False):
        if self.marginofstability_calculator is not None:
            return

        self.marginofstability_calculator = MarginofStabilityCalculation(
            model=self.model_creator.biorbd_model,
            markers_sorted=self.experimental_data.markers_sorted,
            model_marker_names=self.experimental_data.model_marker_names,
            q=self.kinematics_reconstructor.q_filtered,
            qdot=self.kinematics_reconstructor.qdot,
            experimental_data=self.experimental_data,
            skip_if_existing=skip_if_existing,
        )

    def compute_probalityofstability(self, skip_if_existing: bool = False):
        if self.percentage_of_instability_calculator is not None:
            return

        self.percentage_of_instability_calculator = ProbabilityOfInstability(
            marginofstability_calculator=self.marginofstability_calculator,
            experimental_data=self.experimental_data,
            subject=self.subject,
            skip_if_existing=skip_if_existing,
        )

    def estimate_optimally(
            self,
            cycles_to_analyze: int | list,  # <- int | list
            plot_solution_flag: bool = False,
            animate_solution_flag: bool = False,
            skip_if_existing: bool = False,
    ):

        # Checks
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is None:
            raise Exception(
                "Please run the events detection first by running ResultManager.add_cyclic_events() or ResultManager.add_unique_events()"
            )
        if self.kinematics_reconstructor is None:
            raise Exception(
                "Please run the kinematics reconstruction first by running ResultManager.reconstruct_kinematics()"
            )
        if self.inverse_dynamics_performer is None:
            raise Exception("Please run the inverse dynamics first by running ResultManager.perform_inverse_dynamics()")
        if self.experimental_data.gait_parameters_all is None:
            raise Exception(
                "Please compute gait parameters first by running ResultManager.extract_gait_parameters()"
            )

        # Perform the optimal estimation optimization
        self.optimal_estimator = OptimalEstimator(
            cycles_to_analyze=cycles_to_analyze,
            subject=self.subject,
            model_creator=self.model_creator,
            experimental_data=self.experimental_data,
            events=self.events,
            kinematics_reconstructor=self.kinematics_reconstructor,
            inverse_dynamic_performer=self.inverse_dynamics_performer,
            gait_parameters_all=self.experimental_data.gait_parameters_all,
            plot_solution_flag=plot_solution_flag,
            animate_solution_flag=animate_solution_flag,
            skip_if_existing=skip_if_existing,
        )
