import os
import pickle
import numpy as np
from scipy.stats import norm
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.subject import Subject
from gait_analyzer.biomechanics_quantities.margin_of_stability_calculator import (
    MarginOfStabilityCalculator,
)


class ProbabilityOfInstability:
    """
    Computes the Probability of Instability (PoI) based on ML and AP Margin of Stability (MoS)

    Steps:
    1. Detect heel strikes from GRF
    2. Compute one MoS value per step
    3. Estimate distribution (mu, sigma)
    4. Compute PoI = P(MoS < 0)

    Reference:
    Render AC, Cusumano JP, Dingwell JB. Probability of lateral instability while walking on
    winding paths. Journal of Biomechanics. 2024;176:112361. doi:10.1016/j.jbiomech.2024.112361.
    (defined there for the mediolateral direction; applied analogously to AP here)
    """

    def __init__(
        self,
        marginofstability_calculator: MarginOfStabilityCalculator,
        experimental_data: ExperimentalData,
        subject: Subject,
        skip_if_existing: bool,
        heel_strike_threshold: float = 20,
    ):
        self.marginofstability_calculator = marginofstability_calculator
        self.experimental_data = experimental_data
        self.subject_mass = subject.subject_mass
        self.subject_height = subject.subject_height
        self.f_foot1 = self.experimental_data.f_ext_sorted_filtered[0, 6:9, :]
        self.f_foot2 = self.experimental_data.f_ext_sorted_filtered[1, 6:9, :]
        self.ML_MoS = self.marginofstability_calculator.ML_MoS
        self.AP_MoS = self.marginofstability_calculator.AP_MoS
        self.heel_strike_threshold = heel_strike_threshold

        self.heel_strikes_all = None
        self.mos_steps_AP = None
        self.mos_steps_ML = None
        self.mu_ML = None
        self.mu_AP = None
        self.sigma_ML = None
        self.sigma_AP = None
        self.PoI_ML = None
        self.PoI_AP = None
        self.PoI_empirical_ML = None
        self.PoI_empirical_AP = None
        self.is_loaded_mos = False

        if skip_if_existing and self.check_if_existing():
            self.is_loaded_mos = True
        else:
            self.detect_heel_strikes()
            self.compute_mos_per_step()
            self.compute_poi()
            self.save()

    def detect_heel_strikes(self):
        """
        Detect heel strikes for both feet using vertical GRF threshold crossing.
        The threshold (self.heel_strike_threshold, in Newtons) can be set via the constructor.
        """
        fv_foot1 = self.f_foot1[2, :]
        fv_foot2 = self.f_foot2[2, :]

        threshold = self.heel_strike_threshold

        contact1 = fv_foot1 > threshold
        contact2 = fv_foot2 > threshold

        hs_foot1 = []
        hs_foot2 = []

        for i in range(1, len(contact1) - 1):
            if contact1[i] and not contact1[i - 1]:
                if fv_foot1[i + 1] > threshold:
                    hs_foot1.append(i)

        for i in range(1, len(contact2) - 1):
            if contact2[i] and not contact2[i - 1]:
                if fv_foot2[i + 1] > threshold:
                    hs_foot2.append(i)

        hs_foot1 = np.array(hs_foot1)
        hs_foot2 = np.array(hs_foot2)

        hs_all = np.concatenate(
            [
                np.vstack((hs_foot1, np.ones(len(hs_foot1)))).T,
                np.vstack((hs_foot2, np.zeros(len(hs_foot2)))).T,
            ]
        )

        hs_all = hs_all[np.argsort(hs_all[:, 0])]

        self.heel_strikes_all = hs_all  # [frame, foot_id]

    def compute_mos_per_step(self):
        """
        Average ML and AP MoS per step, handling different sampling rates
        between GRF (heel strikes) and MoS.
        """
        mos_steps_ML = []
        mos_steps_AP = []

        fs_grf = self.experimental_data.analogs_sampling_frequency
        fs_mos = self.experimental_data.marker_sampling_frequency
        factor = fs_grf / fs_mos

        heel_strikes_mos = (self.heel_strikes_all[:, 0] / factor).astype(int)

        for i in range(len(heel_strikes_mos) - 1):
            start = heel_strikes_mos[i]
            end = heel_strikes_mos[i + 1]

            if end <= start:
                continue

            mos_segment_ML = self.ML_MoS[start:end]
            mos_segment_AP = self.AP_MoS[start:end]

            # Compute the meanwhile ignoring NaNs
            if not np.all(np.isnan(mos_segment_ML)):
                mos_steps_ML.append(np.nanmean(mos_segment_ML))
            if not np.all(np.isnan(mos_segment_AP)):
                mos_steps_AP.append(np.nanmean(mos_segment_AP))

        self.mos_steps_ML = np.array(mos_steps_ML)
        self.mos_steps_AP = np.array(mos_steps_AP)

    def compute_poi(self):
        """
        Compute PoI from MoS distribution
        """
        # ML
        if len(self.mos_steps_ML) < 2:
            self.mu_ML = np.nan
            self.sigma_ML = np.nan
            self.PoI_ML = np.nan
            return

        self.mu_ML = np.mean(self.mos_steps_ML)
        self.sigma_ML = np.std(self.mos_steps_ML, ddof=1)

        if self.sigma_ML > 0:
            z = (0 - self.mu_ML) / self.sigma_ML
            self.PoI_ML = norm.cdf(z)
        else:
            self.PoI_ML = np.nan

        self.PoI_ML = (np.sum(self.mos_steps_ML < 0) / len(self.mos_steps_ML)) * 100

        # AP
        if len(self.mos_steps_AP) < 2:
            self.mu_AP = np.nan
            self.sigma_AP = np.nan
            self.PoI_AP = np.nan
            return

        self.mu_AP = np.mean(self.mos_steps_AP)
        self.sigma_AP = np.std(self.mos_steps_AP, ddof=1)

        if self.sigma_AP > 0:
            z = (0 - self.mu_AP) / self.sigma_AP
            self.PoI_AP = norm.cdf(z)
        else:
            self.PoI_AP = np.nan

        self.PoI_AP = (np.sum(self.mos_steps_AP < 0) / len(self.mos_steps_AP)) * 100

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        return f"{result_folder}/probability_of_instability_{trial_name}.pkl"

    def save(self):
        with open(self.get_result_file_full_path(), "wb") as file:
            pickle.dump(
                {
                    "heel_strikes_all": self.heel_strikes_all,
                    "mos_steps_ML": self.mos_steps_ML,
                    "mos_steps_AP": self.mos_steps_AP,
                    "mu_ML": self.mu_ML,
                    "mu_AP": self.mu_AP,
                    "sigma_ML": self.sigma_ML,
                    "sigma_AP": self.sigma_AP,
                    "PoI_ML": self.PoI_ML,
                    "PoI_AP": self.PoI_AP,
                },
                file,
            )

    def check_if_existing(self) -> bool:
        path = self.get_result_file_full_path()
        if os.path.exists(path):
            with open(path, "rb") as file:
                data = pickle.load(file)
                self.heel_strikes_all = data["heel_strikes_all"]
                self.mos_steps_ML = data["mos_steps_ML"]
                self.mos_steps_AP = data["mos_steps_AP"]
                self.mu_ML = data["mu_ML"]
                self.mu_AP = data["mu_AP"]
                self.sigma_ML = data["sigma_ML"]
                self.sigma_AP = data["sigma_AP"]
                self.PoI_ML = data["PoI_ML"]
                self.PoI_AP = data["PoI_AP"]
            return True
        return False

    def outputs(self):
        return {
            "PoI_ML": self.PoI_ML,
            "PoI_AP": self.PoI_AP,
        }
