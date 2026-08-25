import os
import pickle
from scipy.io import savemat
from datetime import date
import subprocess
import json
import shutil
import ezc3d
import numpy as np

from gait_analyzer.subject import Subject


class AnalysisPerformer:
    def __init__(
        self,
        analysis_to_perform: callable,
        subjects_to_analyze: list[Subject],
        cycles_to_analyze: range | dict[str, dict[str, range]] | None = None,
        result_folder: str = os.path.dirname(os.path.abspath(__file__)) + "/results/",
        trails_to_analyze: list[str] = None,
        skip_if_existing: bool = False,
        **kwargs,
    ):
        """
        Initialize the AnalysisPerformer.
        .
        Parameters
        ----------
        analysis_to_perform: callable(subject_name: str, subject_mass: float, c3d_file_name: str)
            The analysis to perform
        subjects_to_analyze: list[Subject]
            The list of subjects to analyze
        cycles_to_analyze: range | cycles_to_analyze
            The range of cycles to analyze
        result_folder: str
            The folder where the results will be saved. It will look like result_folder/subject_name.
        trails_to_analyze: list[str]
            The list of trails to analyze. If None, all the trails will be analyzed.
        skip_if_existing: bool
            If True, the analysis will not be performed if the results already exist.
        **kwargs: Any
            Any additional arguments to pass to the analysis_to_perform function
        """

        # Checks:
        if not callable(analysis_to_perform):
            raise ValueError("analysis_to_perform must be a callable")
        if not isinstance(subjects_to_analyze, list):
            raise ValueError("subjects_to_analyze must be a list of Subject")
        for subject in subjects_to_analyze:
            if not isinstance(subject, Subject):
                raise ValueError("All elements of subjects_to_analyze must be Subject")
        if cycles_to_analyze is None:
            cycles_to_analyze = {}
            for subject in subjects_to_analyze:
                cycles_to_analyze[subject.subject_name] = None
        elif isinstance(cycles_to_analyze, range):
            temporary_cycles_to_analyze = cycles_to_analyze
            cycles_to_analyze = {}
            for subject in subjects_to_analyze:
                cycles_to_analyze[subject.subject_name] = temporary_cycles_to_analyze
        elif isinstance(cycles_to_analyze, dict):
            for subject_name, cycles in cycles_to_analyze.items():
                if not isinstance(subject_name, str):
                    raise ValueError("Keys of cycles_to_analyze must be strings (subject names)")
                if not (isinstance(cycles, (range, dict)) or cycles is None):
                    raise ValueError(
                        "Values of cycles_to_analyze must be ranges of cycles to analyze or None if all cycles should be analyzed."
                    )
        else:
            raise ValueError(
                "cycles_to_analyze must be a range or a dictionary of ranges ({'subject_name': {'trial_name': range}})"
            )
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")
        if not isinstance(trails_to_analyze, list) and trails_to_analyze is not None:
            raise ValueError("trails_to_analyze must be a list of strings")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
            print(f"Result folder did not exist, I have created it here {os.path.abspath(result_folder)}")

        # Initial attributes
        self.analysis_to_perform = analysis_to_perform
        self.subjects_to_analyze = subjects_to_analyze
        self.cycles_to_analyze = cycles_to_analyze
        self.result_folder = result_folder
        self.trails_to_analyze = trails_to_analyze
        self.skip_if_existing = skip_if_existing
        self.kwargs = kwargs

        # Extended attributes
        self.figures_result_folder = None
        self.models_result_folder = None

        # Run the analysis
        self.check_for_geometry_files()
        self.run_analysis()

    @staticmethod
    def get_version():
        """
        Save the version of the code and the date of the analysis for future reference
        """

        # Packages installed in the env
        # Running 'conda list' command and parse it as JSON
        result = subprocess.run(["conda", "list", "--json"], capture_output=True, text=True)
        packages = json.loads(result.stdout)
        packages_versions = {elt["name"]: elt["version"] for elt in packages}

        version_dic = {
            "date_of_the_analysis": date.today().strftime("%b-%d-%Y-%H-%M-%S"),
            "biorbd_version": packages_versions["biorbd"],
            "pyomeca_version": packages_versions["pyomeca"] if "pyomeca" in packages_versions else "Not installed",
            "ezc3d_version": packages_versions["ezc3d"],
            "bioptim_version": (
                packages_versions["bioptim"] if "bioptim" in packages_versions else "Not installed through conda-forge"
            ),
        }
        return version_dic

    def save_subject_results(self, results, result_file_name: str, cycles_to_analyze: range | None):
        """
        Save the results of the analysis in a pickle file and a matlab file.
        .
        Parameters
        ----------
        results: ResultManager
            The ResultManager containing the results of the analysis performed by analysis_to_perform
        result_file_name: str
            The name of the file where the results will be saved. The file will be saved as result_file_name.pkl and result_file_name.mat
        cycles_to_analyze: range | None
            The range of cycles to analyze. If None, all cycles will be analyzed.
        """

        result_dict = self.get_version()
        result_dict["cycles_to_analyze"] = cycles_to_analyze if cycles_to_analyze is not None else 0
        for attr_name in dir(results):
            attr = getattr(results, attr_name)
            if not callable(attr) and not attr_name.startswith("__"):
                if hasattr(attr, "outputs") and callable(getattr(attr, "outputs")):
                    this_output_dict = attr.outputs()
                    for key, value in this_output_dict.items():
                        if key in result_dict:
                            raise ValueError(
                                f"Key {key} from class {attr_name} already exists in the result dictionary, please change the key to differentiate them."
                            )
                        elif key == "biorbd_model":
                            pass  # biorbd models are not picklable
                        elif value is None:
                            pass  # Nones are not picklable
                        else:
                            result_dict[key] = value

        # Save the results
        # For python analysis
        with open(result_file_name + ".pkl", "wb") as f:
            pickle.dump(result_dict, f)
        # For matlab analysis
        # --- Fix for MATLAB field name length limit (31 chars) ---
        def shorten_keys(d, max_len=31):
            """Recursively truncates keys that are too long for savemat()"""
            new_dict = {}
            for k, v in d.items():
                new_key = k[:max_len] if isinstance(k, str) else k
                if isinstance(v, dict):
                    new_dict[new_key] = shorten_keys(v, max_len)
                else:
                    new_dict[new_key] = v
            return new_dict

        result_dict = shorten_keys(result_dict)
        savemat(result_file_name + ".mat", result_dict)

    def check_for_geometry_files(self):
        """
        This function is necessary since it is not possible to exclude the examples/results/ folder from the git repository while tracking the examples/results/Geometry/ folder.
        So, it was chosen to track the vtps from the models/OpenSim_models/Geometry/ folder and copy them to the examples/results/Geometry/ folder.
        This is not a bad solution since the vtp files are needed if a user wants to open the osim model in OpenSim.
        """
        # If the folder does not exist, create it and fill it with all the geometry files
        parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.exists(parent_path + "/examples/results/Geometry/"):
            print("Copying the Geometry .vtp files to the folder examples/results/Geometry/")
            os.makedirs(parent_path + "/examples/results/Geometry/")
            for file in os.listdir(parent_path + "/models/OpenSim_models/Geometry/"):
                shutil.copyfile(
                    parent_path + f"/models/OpenSim_models/Geometry/{file}",
                    parent_path + f"/examples/results/Geometry/{file}",
                )
        else:
            # If the folder exists, check if the files are the same size (if not replace the file)
            for file in os.listdir(parent_path + "/models/OpenSim_models/Geometry/"):
                if os.path.exists(parent_path + f"/examples/results/Geometry/{file}"):
                    if os.path.getsize(parent_path + f"/models/OpenSim_models/Geometry/{file}") != os.path.getsize(
                        parent_path + f"/examples/results/Geometry/{file}"
                    ):
                        print(f"Copying {file}.vtp to the folder examples/results/Geometry/")
                        shutil.copyfile(
                            parent_path + f"/models/OpenSim_models/Geometry/{file}",
                            parent_path + f"/examples/results/Geometry/{file}",
                        )
                else:
                    shutil.copyfile(
                        parent_path + f"/models/OpenSim_models/Geometry/{file}",
                        parent_path + f"/examples/results/Geometry/{file}",
                    )

    def get_cycles_to_analyze_for_this_trial(self, subject_name: str, data_file: str):
        if self.cycles_to_analyze is None:
            cycles_to_analyze = None
        else:
            if subject_name not in self.cycles_to_analyze.keys():
                raise ValueError("Please provide a cycles_to_analyze for each subject in the subjects_to_analyze list.")
            temporary_cycles_to_analyze = self.cycles_to_analyze[subject_name]
            if temporary_cycles_to_analyze is None:
                cycles_to_analyze = None
            elif isinstance(temporary_cycles_to_analyze, range):
                cycles_to_analyze = temporary_cycles_to_analyze
            elif isinstance(temporary_cycles_to_analyze, dict):
                file_found = False
                for key in temporary_cycles_to_analyze.keys():
                    if data_file[:-4].endswith(key):
                        cycles_to_analyze = temporary_cycles_to_analyze[key]
                        break
                if not file_found:
                    ValueError(
                        f"Please provide a cycles_to_analyze for each trial in the data file {data_file} for subject {subject_name}."
                    )
            else:
                ValueError(
                    f"cycles_to_analyze for subject {subject_name} must be a range or a dictionary of ranges, not {type(temporary_cycles_to_analyze)}"
                )
        return cycles_to_analyze

    def run_analysis(self):
        """
        Loops over the data files and perform the analysis specified by the user (on the subjects specified by the user).
        """
        parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Loop over all subjects
        for subject in self.subjects_to_analyze:

            subject_name = subject.subject_name
            subject_data_folder = parent_path + f"/data/{subject_name}"

            # Checks
            if not os.path.exists(subject_data_folder):
                os.makedirs(subject_data_folder)
                tempo_subject_path = os.path.abspath(subject_data_folder)
                raise RuntimeError(
                    f"Data folder for subject {subject_name} does not exist. I have created it here {tempo_subject_path}, please put the data files in here."
                )

            # Loop over files to find the static trial
            static_trial_full_file_path = None
            for data_file in os.listdir(subject_data_folder):
                if data_file.endswith("static.c3d"):
                    static_trial_full_file_path = parent_path + f"/data/{subject_name}/{data_file}"
                    break
            if not static_trial_full_file_path:
                raise FileNotFoundError(
                    f"Please put the static trial file here {os.path.abspath(subject_data_folder)} and name it [...]_static.c3d"
                )

            # Define mass with static trial
            if subject.subject_mass is None:
                static_c3d = ezc3d.c3d(static_trial_full_file_path, extract_forceplat_data=True)
                summed_force = 0
                for i_platform in range(len(static_c3d["data"]["platform"])):
                    summed_force += static_c3d["data"]["platform"][i_platform]["force"]
                # Réunion Island : ~9.782
                subject.subject_mass = np.nanmedian(np.linalg.norm(summed_force, axis=0)) / 9.8

            # Define subject specific paths
            result_folder = f"{self.result_folder}/{subject_name}"
            self.figures_result_folder = f"{result_folder}/figures"
            self.models_result_folder = f"{result_folder}/models"
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
                os.makedirs(self.figures_result_folder)
                os.makedirs(self.models_result_folder)
                print("The results folder was created here: ", os.path.abspath(result_folder))
            if not os.path.exists(self.figures_result_folder):
                os.makedirs(self.figures_result_folder)
            if not os.path.exists(self.models_result_folder):
                os.makedirs(self.models_result_folder)

            # Loop over all data files
            for data_file in os.listdir(subject_data_folder):
                # Files that we should not analyze
                if data_file.endswith("static.c3d") or not data_file.endswith(".c3d"):
                    continue
                if self.trails_to_analyze is not None and not any(
                    trail in data_file for trail in self.trails_to_analyze
                ):
                    continue

                c3d_file_name = parent_path + f"/data/{subject_name}/{data_file}"
                result_file_name = f"{result_folder}/{data_file.replace('.c3d', '_results')}"

                # Skip if already exists
                if self.skip_if_existing and os.path.exists(result_file_name + ".pkl"):
                    print(f"Skipping {subject_name} - {data_file} because it already exists.")
                    continue

                cycles_to_analyze = self.get_cycles_to_analyze_for_this_trial(subject_name, data_file)

                # Actually perform the analysis
                print("\n\n\nAnalyzing ", subject_name, " : ***** ", data_file, " *****")
                results = self.analysis_to_perform(
                    subject,
                    cycles_to_analyze,
                    static_trial_full_file_path,
                    c3d_file_name,
                    result_folder,
                    **self.kwargs,

                )
                self.save_subject_results(results, result_file_name, cycles_to_analyze)
