from gait_analyzer import (
    helper,
    ResultManager,
    OsimModels,
    AnalysisPerformer,
    PlotLegData,
    LegToPlot,
    PlotType,
    Subject,
    Side,
    ReconstructionType,
    MarkerLabelingHandler,
    OrganizedResult,
)


def analysis_to_perform(
    subject: Subject,
    cycles_to_analyze: range | None,
    static_trial: str,
    c3d_file_name: str,
    result_folder: str,
):

    # --- Example of analysis that must be performed in order --- #
    results = ResultManager(
        subject=subject,
        cycles_to_analyze=cycles_to_analyze,
        static_trial=static_trial,
        result_folder=result_folder,
    )

    results.create_model(
        osim_model_type=OsimModels.WholeBody(),
        functional_trials_path=f"../data/{subject.subject_name}/functional_trials/",
        mvc_trials_path=f"../data/{subject.subject_name}/maximal_voluntary_contractions/",
        q_regularization_weight=0.01,
        skip_if_existing=True,
        animate_model_flag=False,
    )

    markers_to_ignore = []
    analogs_to_ignore = [
        "Channel_01",
        "Channel_02",
        "Channel_03",
        "Channel_04",
        "Channel_05",
        "Channel_06",
        "Channel_07",
        "Channel_08",
        "Channel_09",
        "Channel_10",
        "Channel_11",
        "Channel_12",
        "Bertec_treadmill_speed",
    ]
    results.add_experimental_data(
        c3d_file_name=c3d_file_name, markers_to_ignore=markers_to_ignore, analogs_to_ignore=analogs_to_ignore
    )

    results.add_cyclic_events(force_plate_sides=[Side.RIGHT, Side.LEFT], skip_if_existing=True, plot_phases_flag=False)

    results.reconstruct_kinematics(
        reconstruction_type=[ReconstructionType.LSQ, ReconstructionType.ONLY_LM],
        animate_kinematics_flag=False,
        plot_kinematics_flag=False,
        skip_if_existing=True,
    )

    results.perform_inverse_dynamics(skip_if_existing=True, reintegrate_flag=False, animate_dynamics_flag=False)

    # --- Example of analysis that can be performed in any order --- #
    results.estimate_optimally(
        cycle_to_analyze=5,
        plot_solution_flag=True,
        animate_solution_flag=True,
        skip_if_existing=True,
    )

    return results


def parameters_to_extract_for_statistical_analysis():
    # TODO: Add the parameters you want to extract for statistical analysis
    pass


if __name__ == "__main__":

    # --- Example of how to get help on a GaitAnalyzer class --- #
    # helper(Operator)

    # --- Create the list of participants --- #
    subjects_to_analyze = []
    subjects_to_analyze.append(Subject(subject_name="LEM_PRE_chev"))
    # ... add other participants here

    # --- Example of how to run the analysis --- #
    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=subjects_to_analyze,
        cycles_to_analyze=range(5, -5),
        result_folder="results",
        skip_if_existing=False,
    )

    # --- Example of how to create a OrganizedResult object --- #
    organized_result = OrganizedResult(
        result_folder="results",
        plot_type=PlotType.MUSCLE_FORCES,
        nb_frames_interp=101,
        conditions_to_compare=["_1p25"],
    )
    organized_result.save("results/OptimEstim_organized.pkl")

    # --- Example of how to plot the joint angles--- #
    plot = PlotLegData(
        organized_result=organized_result,
    )
    plot.draw_plot()
    plot.save("results/OptimEstim_MUSCLE_FORCES_plot.png")
    plot.show()
