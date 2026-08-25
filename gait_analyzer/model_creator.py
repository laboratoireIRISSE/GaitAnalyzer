import os
import pickle
from copy import deepcopy
import numpy as np
import biorbd
import ezc3d
from pyomeca import Analogs

from biobuddy import (
    BiomechanicalModelReal,
    MuscleType,
    MuscleStateType,
    ScaleTool,
    RangeOfMotion,
    Ranges,
    C3dData,
    MarkerReal,
    JointCenterTool,
    Score,
    Sara,
    Translations,
    Rotations,
    MarkerWeight,
    AxisWiseScaling,
    RotoTransMatrix,
    Axis,
)
from gait_analyzer.subject import Subject


class OsimModels:

    @property
    def osim_model_name(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].osim_model_name."
        )

    @property
    def original_osim_model_full_path(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].original_osim_model_full_path."
        )

    @property
    def xml_setup_file(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].xml_setup_file."
        )

    @property
    def muscles_to_ignore(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].muscles_to_ignore."
        )

    @property
    def markers_to_ignore(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].markers_to_ignore."
        )

    @property
    def ranges_to_adjust(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].ranges_to_adjust."
        )

    @property
    def segments_to_fix(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].segments_to_fix."
        )

    @property
    def markers_to_add(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].markers_to_add."
        )

    def perform_modifications(self, model, static_trial):
        """
        1. Make the gravity vector point downwards (necessary because OpenSim is Y-up and biorbd is Z-up)
        2. Remove the markers that are not needed (markers_to_ignore)
        3. Remove the degrees of freedom that are not needed (segments_to_fix)
        4. Change the ranges of motion for the segments (ranges_to_adjust)
        5. Remove the muscles/via_points/muscle_groups that are not needed (muscles_to_ignore)
        6. Add the marker clusters (markers_to_add)
        7. Fix the conditional and moving via points
        """
        # Modify gravity
        model.gravity = np.array([0, 0, -9.81])

        # Modify segments
        for segment in model.segments:
            markers = deepcopy(segment.markers)
            for marker in markers:
                if marker in self.markers_to_ignore:
                    segment.remove_marker(marker)
            if segment.name in self.ranges_to_adjust.keys():
                min_bounds = [r[0] for r in self.ranges_to_adjust[segment.name]]
                max_bounds = [r[1] for r in self.ranges_to_adjust[segment.name]]
                segment.q_ranges = RangeOfMotion(Ranges.Q, min_bounds, max_bounds)
            if segment in self.segments_to_fix:
                segment.translations = Translations.NONE
                segment.rotations = Rotations.NONE
                segment.q_ranges = None
                segment.qdot_ranges = None

        # Modify muscles
        # Remove muscles
        muscles_to_ignore = [m for m in self.muscles_to_ignore if m in model.muscle_names]
        for muscle_group in model.muscle_groups:
            muscles = deepcopy(model.muscle_groups[muscle_group.name].muscles)
            for muscle in muscles:
                if muscle.name in muscles_to_ignore:
                    model.muscle_groups[muscle_group.name].remove_muscle(muscle.name)
        # Remove muscle groups that are now empty
        muscle_groups = deepcopy(model.muscle_groups)
        for muscle_group in muscle_groups:
            if muscle_group.nb_muscles == 0:
                model.remove_muscle_group(muscle_group.name)

        # Add the marker clusters
        jcs_in_global = model.forward_kinematics()
        c3d_data = C3dData(static_trial, first_frame=100, last_frame=200)
        for segment_name in self.markers_to_add.keys():
            for marker in self.markers_to_add[segment_name]:
                position_in_global = c3d_data.mean_marker_position(marker)
                rt = jcs_in_global[segment_name][0]
                position_in_local = rt.inverse @ position_in_global
                model.segments[segment_name].add_marker(
                    MarkerReal(
                        name=marker,
                        parent_name=segment_name,
                        position=position_in_local,
                        is_anatomical=False,
                        is_technical=True,
                    )
                )

        return model

    # Child classes acting as an enum
    class WholeBody:
        """This is a hole body model that consists of 23 bodies, 42 degrees of freedom and 30 muscles.
        The whole-body geometric model and the lower limbs, pelvis and upper limbs anthropometry are based on the running model of Hammer et al. 2010 which consists of 12 segments and 29 degrees of freedom.
        Extra segments and degrees of freedom were later added based on Dumas et al. 2007.
        Each lower extremity had seven degrees-of-freedom; the hip was modeled as a ball-and-socket joint (3 DoFs), the knee was modeled as a revolute joint with 1 dof, the ankle was modeled as 2 revolute joints and feet toes with one revolute joint.
        The pelvis joint was model as a free flyer joint (6 DoFs) to allow the model to translate and rotate in the 3D space, the lumbar motion was modeled as a ball-and-socket joint (3 DoFs) (Anderson and Pandy, 1999) and the neck joint was modeled as a ball-and-socket joint (3 DoFs).
        Mass properties of the torso and head (including the neck) segments were estimated from Dumas et al., 2007. Each arm consisted of 8 degrees-of-freedom; the shoulder was modeled as a ball-and-socket joint (3 DoFs), the elbow and forearm rotation were each modeled with revolute joints (1 dof) (Holzbaur et al., 2005), the wrist flexion and deviation were each modeled with revolute joints and the hand fingers were modeled with 1 revolute joint.
        Mass properties for the arms were estimated from 1 and de Leva, 1996. The model also include 30 superficial muscles of the whole body.
        [Charbie -> Link ?]
        """

        def __init__(self):
            # Generic
            self.osim_model_name = "wholebody"
            parent_path = os.path.dirname(os.path.abspath(__file__))
            self._original_osim_model_full_path = parent_path + "/../models/OpenSim_models/wholebody_Flo.osim"
            self._xml_setup_file = parent_path + "/../models/OpenSim_models/wholebody_Flo.xml"

            # Specific
            self._muscles_to_ignore = [
                "ant_delt_r",
                "ant_delt_l",
                "medial_delt_l",
                "post_delt_r",
                "post_delt_l",
                "medial_delt_r",
                "ercspn_r",
                "ercspn_l",
                "rect_abd_r",
                "rect_abd_l",
                "r_stern_mast",
                "l_stern_mast",
                "r_trap_acr",
                "l_trap_acr",
                "TRIlong",
                "TRIlong_l",
                "TRIlat",
                "TRIlat_l",
                "BIClong",
                "BIClong_l",
                "BRD",
                "BRD_l",
                "FCR",
                "FCR_l",
                "ECRL",
                "ECRL_l",
                "PT",
                "PT_l",
                "LAT2",
                "LAT2_l",
                "PECM2",
                "PECM2_l",
                "glut_med1_r",
                "glut_med1_l",
            ]
            self._markers_to_ignore = []
            self._ranges_to_adjust = {
                "pelvis_translation": [
                    [-3, 3],
                    [-3, 3],
                    [-3, 3],
                ],
                "pelvis_rotation_transform": [
                    [-np.pi / 4, np.pi / 4],
                    [-np.pi / 4, np.pi / 4],
                    [-np.pi, np.pi],
                ],
                "femur_r_rotation_transform": [
                    [-40 * np.pi / 180, 120 * np.pi / 180],
                    [-60 * np.pi / 180, 30 * np.pi / 180],
                    [-30 * np.pi / 180, 30 * np.pi / 180],
                ],
                "tibia_r_rotation_transform": [
                    [-150 * np.pi / 180, 0],
                ],
                "talus_r_ankle_angle_r": [
                    [-50 * np.pi / 180, 30 * np.pi / 180],  # Ankle Flexion
                ],
                "calcn_r_subtalar_angle_r": [
                    [-15 * np.pi / 180, 15 * np.pi / 180],  # Ankle Inversion
                ],
                "toes_r_rotation_transform": [
                    [-25 * np.pi / 180, 25 * np.pi / 180],  # Toes Flexion
                ],
                "femur_l_rotation_transform": [
                    [-40 * np.pi / 180, 120 * np.pi / 180],
                    [-60 * np.pi / 180, 30 * np.pi / 180],
                    [-30 * np.pi / 180, 30 * np.pi / 180],
                ],
                "tibia_l_rotation_transform": [
                    [-150 * np.pi / 180, 0],
                ],
                "talus_l_ankle_angle_l": [
                    [-50 * np.pi / 180, 30 * np.pi / 180],  # Ankle Flexion
                ],
                "calcn_l_subtalar_angle_l": [
                    [-15 * np.pi / 180, 15 * np.pi / 180],  # Ankle Inversion
                ],
                "toes_l_rotation_transform": [
                    [-25 * np.pi / 180, 25 * np.pi / 180],  # Toes Flexion
                ],
                "torso_rotation_transform": [
                    [-90 * np.pi / 180, 45 * np.pi / 180],
                    [-35 * np.pi / 180, 35 * np.pi / 180],
                    [-45 * np.pi / 180, 45 * np.pi / 180],
                ],
                "head_neck_rotation_transform": [[-50 * np.pi / 180, 45 * np.pi / 180], [-0.6, 0.6], [-1.2217, 1.2217]],
                "humerus_r_rotation_transform": [
                    [-np.pi / 2, np.pi],
                    [-3.8397, np.pi / 2],
                    [-np.pi / 2, np.pi / 2],
                ],
                "ulna_r_elbow_flex_r": [
                    [0.0, np.pi],
                ],
                "radius_r_pro_sup_r": [
                    [-np.pi, np.pi],
                ],
                "lunate_r_rotation_transform": [
                    [-np.pi / 2, np.pi / 2],
                ],
                "hand_r_rotation_transform": [
                    [-0.43633231, 0.61086524],
                ],
                "fingers_r_rotation_transform": [
                    [-np.pi / 2, np.pi / 2],
                ],
                "humerus_l_rotation_transform": [
                    [-np.pi / 2, np.pi],
                    [-3.8397, np.pi / 2],
                    [-np.pi / 2, np.pi / 2],
                ],
                "ulna_l_elbow_flex_l": [
                    [0.0, np.pi],
                ],
                "radius_l_pro_sup_l": [
                    [-np.pi, np.pi],
                ],
                "lunate_l_rotation_transform": [
                    [-np.pi / 2, np.pi / 2],
                ],
                "hand_l_rotation_transform": [
                    [-0.43633231, 0.61086524],
                ],
                "fingers_l_rotation_transform": [
                    [-np.pi / 2, np.pi / 2],
                ],
            }
            self._segments_to_fix = [
                "toes_r_rotation_transform",
                "hand_r_rotation_transform",
                "fingers_r_rotation_transform",
                "toes_l_rotation_transform",
                "hand_l_rotation_transform",
                "fingers_l_rotation_transform",
            ]
            self._markers_to_add = {
                "femur_r": ["R_fem_up", "R_fem_downF", "R_fem_downB"],
                "femur_l": ["L_fem_up", "L_fem_downF", "L_fem_downB"],
                "tibia_r": ["R_tib_up", "R_tib_downF", "R_tib_downB"],
                "tibia_l": ["L_tib_up", "L_tib_downF", "L_tib_downB"],
                "calcn_r": ["R_foot_up"],
                "calcn_l": ["L_foot_up"],
                "humerus_r": ["R_arm_up", "R_arm_downF", "R_arm_downB"],
                "radius_r": ["R_fore_up", "R_fore_downF", "R_fore_downB"],
                "humerus_l": ["L_arm_up", "L_arm_downF", "L_arm_downB"],
                "radius_l": ["L_fore_up", "L_fore_downF", "L_fore_downB"],
            }
            self._muscle_name_mapping = {
                "semiten_r": "SEMITENDINOUS",
                "bifemlh_r": "BICEPS_FEM",
                "sar_r": "RECTUS_FEM",
                "tfl_r": None,
                "vas_med_r": "VASTM",
                "vas_lat_r": "VASTM",
                "soleus_r": "SOL",
                "tib_post_r": "SOL",
                "tib_ant_r": "TIB",
                "per_long_r": "GM",
                "med_gas_r": "GM",
                "lat_gas_r": "GM",
            }

        @property
        def original_osim_model_full_path(self) -> str:
            return self._original_osim_model_full_path

        @original_osim_model_full_path.setter
        def original_osim_model_full_path(self, value: str) -> None:
            self._original_osim_model_full_path = value

        @property
        def xml_setup_file(self) -> str:
            return self._xml_setup_file

        @xml_setup_file.setter
        def xml_setup_file(self, value: str) -> None:
            self._xml_setup_file = value

        @property
        def muscles_to_ignore(self) -> list[str]:
            return self._muscles_to_ignore

        @muscles_to_ignore.setter
        def muscles_to_ignore(self, value: list[str]) -> None:
            self._muscles_to_ignore = value

        @property
        def markers_to_ignore(self) -> list[str]:
            return self._markers_to_ignore

        @markers_to_ignore.setter
        def markers_to_ignore(self, value: list[str]) -> None:
            self._markers_to_ignore = value

        @property
        def ranges_to_adjust(self) -> dict[str, list[list[float]]]:
            return self._ranges_to_adjust

        @ranges_to_adjust.setter
        def ranges_to_adjust(self, value: dict[str, list[list[float]]]) -> None:
            self._ranges_to_adjust = value

        @property
        def segments_to_fix(self) -> list[str]:
            return self._segments_to_fix

        @segments_to_fix.setter
        def segments_to_fix(self, value: list[str]) -> None:
            self._segments_to_fix = value

        @property
        def markers_to_add(self) -> dict[str, list[str]]:
            return self._markers_to_add

        @markers_to_add.setter
        def markers_to_add(self, value: dict[str, list[str]]) -> None:
            self._markers_to_add = value

        @property
        def muscle_name_mapping(self) -> dict[str, str]:
            """
            This method returns a dictionary that maps the muscle names from the original model to the experimental EMG names.
            This is useful as multiple muscles might be associated with the same EMG signal.
            The keys are the name of the muscles in the model, and the values are the name of the analog (EMG) input in the c3d (motion capture system).
            """
            return self._muscle_name_mapping

        @muscle_name_mapping.setter
        def muscle_name_mapping(self, value: dict[str, str]) -> None:
            self._muscle_name_mapping = value

        def perform_modifications(self, model, static_trial):
            OsimModels.perform_modifications(self, model, static_trial)


class ModelCreator:
    def __init__(
        self,
        subject: Subject,
        static_trial: str,
        functional_trials_path: str,
        mvc_trials_path: str,
        models_result_folder: str,
        osim_model_type,
        q_regularization_weight: float | np.ndarray,
        qdot_regularization_weight: float | np.ndarray,
        skip_if_existing: bool,
        animate_model_flag: bool,
        vtp_geometry_path: str,
    ):
        """
        Initialize the ModelCreator.
        .
        Parameters
        ----------
        subject: Subject
            The subject to create the model for.
        static_trial: str
            The path to the static trial c3d file to use to create the model.
        models_result_folder: str
            The folder where the models will be saved.
        osim_model_type: OsimModels
            The type of model to create.
        q_regularization_weight: float | np.ndarray
            The weight to use for the regularization term on the joint angles during the inverse kinematic step of the scaling procedure.
        qdot_regularization_weight: float | np.ndarray
            The weight to use for the regularization term on the joint velocities during the inverse kinematic step of the scaling procedure.
        skip_if_existing: bool
            If the model already exists, skip the creation.
        animate_model_flag: bool
            If True, animate the model after creating it.
        """

        # Checks
        if not isinstance(subject, Subject):
            raise ValueError("subject must be a Subject.")
        if not isinstance(static_trial, str):
            raise ValueError("static_trial must be a string.")
        if functional_trials_path is not None:
            if not isinstance(functional_trials_path, str):
                raise ValueError("functional_trials_path must be a string.")
            if not os.path.exists(functional_trials_path):
                raise RuntimeError(f"Functional trials path {functional_trials_path} does not exist.")
        # if not os.path.exists(mvc_trials_path):
        #     raise RuntimeError(f"MVC trials path {mvc_trials_path} does not exist.")
        if mvc_trials_path is not None and not os.path.exists(mvc_trials_path):
            raise RuntimeError(f"MVC trials path {mvc_trials_path} does not exist.")
        if not isinstance(models_result_folder, str):
            raise ValueError("models_result_folder must be a string.")
        if not isinstance(q_regularization_weight, (float, int, np.ndarray)):
            raise ValueError("q_regularization_weight must be a float.")
        if not isinstance(qdot_regularization_weight, (float, int, np.ndarray)):
            raise ValueError("qdot_regularization_weight must be a float.")
        if not isinstance(skip_if_existing, bool):
            raise ValueError("skip_if_existing must be a boolean.")
        if not isinstance(animate_model_flag, bool):
            raise ValueError("animate_model_flag must be a boolean.")
        if not isinstance(vtp_geometry_path, str):
            raise ValueError("vtp_geometry_path must be a string.")

        # Initial attributes
        self.subject = subject
        self.osim_model_type = osim_model_type
        self.static_trial = static_trial
        self.functional_trials_path = functional_trials_path
        self.mvc_trials_path = mvc_trials_path
        self.models_result_folder = models_result_folder
        self.q_regularization_weight = q_regularization_weight
        self.qdot_regularization_weight = qdot_regularization_weight

        # Extended attributes
        self.trc_file_path = None
        self.vtp_geometry_path = vtp_geometry_path
        self.biorbd_model_full_path = (
            self.models_result_folder
            + "/"
            + osim_model_type.osim_model_name
            + "_"
            + self.subject.subject_name
            + ".bioMod"
        )
        self.model = None  # This is the object that will be modified to be personalized to the subject
        self.marker_weights = None  # This will be set later by the scale tool
        self.new_model_created = False
        self.mvc_values = None  # This will be set later by the get_mvc_values method

        # Create the models
        if skip_if_existing and self.check_if_existing():
            print(f"The model {self.biorbd_model_full_path} already exists, so it is being used.")
            self.biorbd_model = biorbd.Model(self.biorbd_model_full_path)
        else:
            print(f"The model {self.biorbd_model_full_path} is being created...")
            self.read_osim_model()
            self.scale_model()
            self.osim_model_type.perform_modifications(self.model, self.static_trial)
            if self.functional_trials_path is None:
                print("Skipping relocation of joint centers based on functional trials.")
            else:
                self.relocate_joint_centers_functionally(animate_model_flag)
            self.create_biorbd_model()
            self.biorbd_model = biorbd.Model(self.biorbd_model_full_path)
            self.get_mvc_values(plot_emg_flag=False)
            self.save_model()

        if animate_model_flag:
            self.animate_model()

    def check_if_existing(self) -> bool:
        """
        Check if the model already exists.
        If it exists, load the model and the mvc_values.
        .
        Returns
        -------
        bool
            If the model already exists
        """
        result_file_full_path = (
            self.models_result_folder
            + "/"
            + self.osim_model_type.osim_model_name
            + "_"
            + self.subject.subject_name
            + ".pkl"
        )
        if os.path.exists(result_file_full_path):
            with open(result_file_full_path, "rb") as file:
                data = pickle.load(file)
                self.new_model_created = False
                self.mvc_values = data["mvc_values"]
                self.marker_weights = data["marker_weights"]
            return True
        else:
            return False

    def read_osim_model(self):
        self.model = BiomechanicalModelReal().from_osim(
            filepath=self.osim_model_type.original_osim_model_full_path,
            muscle_type=MuscleType.HILL_DE_GROOTE,
            muscle_state_type=MuscleStateType.DEGROOTE,
            mesh_dir=self.vtp_geometry_path,
        )
        self.model.fix_via_points()

    def scale_model(self):

        # Modify the model's ground orientation
        rt_matrix = RotoTransMatrix()
        rt_matrix.from_euler_angles_and_translation(
            angle_sequence="xy", angles=np.array([np.pi / 2, np.pi]), translation=np.array([0, 0, 0])
        )
        self.model.segments["ground"].segment_coordinate_system.scs = rt_matrix

        scale_tool = ScaleTool(original_model=self.model).from_xml(filepath=self.osim_model_type.xml_setup_file)
        scale_tool.scaling_segments["torso"].scaling_type = AxisWiseScaling(
            marker_pairs={
                Translations.X: [["STR", "T10"], ["SUP", "C7"]],
                Translations.Y: [["RASIS", "RA"], ["LASIS", "LA"], ["RPSIS", "RA"], ["LPSIS", "LA"]],
                Translations.Z: [["LA", "RA"],["LASIS", "RA"], ["RASIS", "LA"], ["LASIS", "RASIS"]],
            },
        )
        self.model = scale_tool.scale(
            static_trial=C3dData(self.static_trial, first_frame=100, last_frame=200),
            mass=self.subject.subject_mass,
            q_regularization_weight=self.q_regularization_weight,
            qdot_regularization_weight=self.qdot_regularization_weight,
            make_static_pose_the_models_zero=True,
            visualize_optimal_static_pose=True,
            method="lm",
        )
        self.marker_weights = scale_tool.marker_weights

    def relocate_joint_centers_functionally(self, animate_model_flag: bool = True):

        animate_reconstruction = animate_model_flag

        # Move the model's joint centers
        joint_center_tool = JointCenterTool(self.model, animate_reconstruction=animate_reconstruction)

        trials_list = {
            "right_hip": None,
            "right_knee": None,
            "right_ankle": None,
            "left_hip": None,
            "left_knee": None,
            "left_ankle": None,
        }
        # Find the functional trials
        for trial_name in trials_list.keys():
            found = False
            for file in os.listdir(self.functional_trials_path):
                if file.endswith(f"{trial_name}.c3d"):
                    trials_list[trial_name] = os.path.join(self.functional_trials_path, file)
                    print(f"Found functional trial: {file}.")
                    found = True
                    break
            if not found:
                abs_path = os.path.abspath(self.functional_trials_path)
                raise RuntimeError(f"The functional trial for {trial_name} was not found in the directory {abs_path}.")

        # Hip Right
        joint_center_tool.add(
            Score(
                functional_trial=C3dData(trials_list["right_hip"]),
                parent_name="pelvis",
                child_name="femur_r",
                parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
                child_marker_names=["RLFE", "RMFE"] + self.osim_model_type.markers_to_add["femur_r"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=animate_reconstruction,
            )
        )
        # Knee right
        joint_center_tool.add(
            Sara(
                functional_trial=C3dData(trials_list["right_knee"]),
                parent_name="femur_r",
                child_name="tibia_r",
                parent_marker_names=["RGT"] + self.osim_model_type.markers_to_add["femur_r"],
                child_marker_names=["RATT", "RLM", "RSPH"] + self.osim_model_type.markers_to_add["tibia_r"],
                joint_center_markers=["RLFE", "RMFE"],
                distal_markers=["RLM", "RSPH"],
                is_longitudinal_axis_from_jcs_to_distal_markers=False,
                expected_rotation_axis_orientation=Axis("right_knee_sara", "RMFE", "RLFE"),
                initialize_whole_trial_reconstruction=False,
                animate_rt=animate_reconstruction,
            )
        )
        # Ankle right
        joint_center_tool.add(
            Score(
                functional_trial=C3dData(trials_list["right_ankle"]),
                parent_name="tibia_r",
                child_name="calcn_r",
                parent_marker_names=["RATT", "RLM", "RSPH"] + self.osim_model_type.markers_to_add["tibia_r"],
                child_marker_names=["RCAL", "RMFH1", "RMFH5"] + self.osim_model_type.markers_to_add["calcn_r"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=animate_reconstruction,
            )
        )
        # Hip Left
        joint_center_tool.add(
            Score(
                functional_trial=C3dData(trials_list["left_hip"]),
                parent_name="pelvis",
                child_name="femur_l",
                parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
                child_marker_names=["LGT", "LLFE", "LMFE"] + self.osim_model_type.markers_to_add["femur_l"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=animate_reconstruction,
            )
        )
        # Knee Left
        joint_center_tool.add(
            Sara(
                functional_trial=C3dData(trials_list["left_knee"]),
                parent_name="femur_l",
                child_name="tibia_l",
                parent_marker_names=["LGT"] + self.osim_model_type.markers_to_add["femur_l"],
                child_marker_names=["LATT", "LLM", "LSPH"] + self.osim_model_type.markers_to_add["tibia_l"],
                joint_center_markers=["LLFE", "LMFE"],
                distal_markers=["LLM", "LSPH"],
                is_longitudinal_axis_from_jcs_to_distal_markers=False,
                expected_rotation_axis_orientation=Axis("left_knee_sara", "LLFE", "LMFE"),
                initialize_whole_trial_reconstruction=False,
                animate_rt=animate_reconstruction,
            )
        )
        # Ankle Left
        joint_center_tool.add(
            Score(
                functional_trial=C3dData(trials_list["left_ankle"]),
                parent_name="tibia_l",
                child_name="calcn_l",
                parent_marker_names=["LATT", "LLM", "LSPH"] + self.osim_model_type.markers_to_add["tibia_l"],
                child_marker_names=["LCAL", "LMFH1", "LMFH5"] + self.osim_model_type.markers_to_add["calcn_l"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=animate_reconstruction,
            )
        )

        original_marker_weights = deepcopy(self.marker_weights)
        for key in self.osim_model_type.markers_to_add.keys():
            for marker in self.osim_model_type.markers_to_add[key]:
                if marker not in original_marker_weights:
                    self.marker_weights._append(MarkerWeight(name=marker, weight=5.0))

        self.model = joint_center_tool.replace_joint_centers(self.marker_weights)

    def create_biorbd_model(self):
        self.model.to_biomod(self.biorbd_model_full_path, with_mesh=True)
        self.new_model_created = True

    def animate_model(self):
        """
        Animate the model
        """
        try:
            from pyorerun import LiveModelAnimation
        except:
            raise RuntimeError("To animate the model, you must install Pyorerun.")

        animation = LiveModelAnimation(self.biorbd_model_full_path, with_q_charts=True)
        animation.rerun()

    def get_mvc_values(self, plot_emg_flag: bool = False):
        """
        Extract the maximal EMG signal as the max of the filtered EMG signal for each muscle during the MVC trial.
        """
        # if self.mvc_trials_path is None:
        #     raise NotImplementedError("This should eb allowed but I did not take the time to implement it.")
        if self.mvc_trials_path is None:
            print("No MVC trials path provided, skipping MVC extraction.")
            self.mvc_values = {}
            return
        mvc_values = {}
        emg_values = {}
        for mvc in os.listdir(self.mvc_trials_path):
            if mvc.endswith(".c3d"):
                mvc_trial = ezc3d.c3d(os.path.join(self.mvc_trials_path, mvc))
                analog_names = [name for name in mvc_trial["parameters"]["ANALOG"]["LABELS"]["value"]]
                emg_units = 1
                if mvc_trial["parameters"]["ANALOG"]["UNITS"]["value"][0] == "V":
                    emg_units = 1_000_000  # Convert to microV

                for name in analog_names:
                    if mvc.endswith(name + ".c3d"):
                        emg = Analogs.from_c3d(
                            os.path.join(self.mvc_trials_path, mvc), suffix_delimiter=".", usecols=[name]
                        )
                        emg = emg.interpolate_na(dim="time", method="linear")
                        emg_processed = (
                            emg.meca.band_pass(order=2, cutoff=[10, 425])
                            .meca.center()
                            .meca.abs()
                            .meca.low_pass(order=4, cutoff=5, freq=emg.rate)
                        ) * emg_units
                        emg_values[name] = np.array(emg_processed)
                        mvc_values[name] = float(np.nanmax(emg_processed))
        self.mvc_values = mvc_values

        if plot_emg_flag:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(len(self.mvc_values.keys()), 1, figsize=(10, 19))
            for i_ax, emg_name in enumerate(self.mvc_values.keys()):
                axs[i_ax].plot(emg_values[emg_name], "-r")
                axs[i_ax].plot(
                    np.array([0, len(emg_values[emg_name])]),
                    np.array([self.mvc_values[emg_name], self.mvc_values[emg_name]]),
                    "k--",
                )
                axs[i_ax].set_ylabel(emg_name)
            plt.savefig("mvc_emg.png")
            # plt.show()

    def save_model(self):
        """
        Save the model building conditions.
        """
        result_file_full_path = (
            self.models_result_folder
            + "/"
            + self.osim_model_type.osim_model_name
            + "_"
            + self.subject.subject_name
            + ".pkl"
        )
        with open(result_file_full_path, "wb") as file:
            outputs = self.outputs()
            outputs["biorbd_model"] = None  # Remove the biorbd model from the outputs because it is not picklable
            pickle.dump(outputs, file)

    def inputs(self):
        return {
            "subject_name": self.subject.subject_name,
            "subject_mass": self.subject.subject_mass,
            "osim_model_type": self.osim_model_type,
            "static_trial": self.static_trial,
            "functional_trials_path": self.functional_trials_path,
            "mvc_trials_path": self.mvc_trials_path,
        }

    def outputs(self):
        return {
            "biorbd_model_full_path": self.biorbd_model_full_path,
            "biorbd_model": self.biorbd_model,
            "new_model_created": self.new_model_created,
            "functional_trials_path": self.functional_trials_path,
            "mvc_trials_path": self.mvc_trials_path,
            "mvc_values": self.mvc_values,
            "marker_weights": self.marker_weights,
        }
