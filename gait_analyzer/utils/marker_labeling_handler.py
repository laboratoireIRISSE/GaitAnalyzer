import numpy as np
import plotly.graph_objects as go
import ezc3d


class MarkerLabelingHandler:
    def __init__(self, c3d_path: str):
        self.c3d_path = c3d_path
        self.c3d = ezc3d.c3d(c3d_path)
        self.markers = self.c3d["data"]["points"]
        self.marker_names = self.c3d["parameters"]["POINT"]["LABELS"]["value"]

    def show_marker_labeling_plot(self, marker_to_show: list[str] = None):

        if marker_to_show is None:
            marker_to_show = self.marker_names

        fig = go.Figure()
        x_vector = np.arange(self.markers.shape[2])
        for i_marker, marker_name in enumerate(self.marker_names):
            if marker_name in marker_to_show:
                fig.add_trace(
                    go.Scatter(
                        x=x_vector,
                        y=np.linalg.norm(self.markers[:3, i_marker, :], axis=0),
                        mode="lines",
                        name=marker_name,
                        line=dict(width=2),
                    )
                )
        fig.update_layout(xaxis_title="Frames", yaxis_title="Markers", template="plotly_white")

        fig.write_html("markers.html")
        fig.show(renderer="browser")

    def invert_marker_labeling(self, marker_names: list, frame_start: int, frame_end: int):
        """
        Invert the marker labeling by swapping two markers.
        """

        if not isinstance(marker_names, list):
            raise TypeError("marker_names should be a list of two marker names to invert.")
        if len(marker_names) != 2:
            raise ValueError("marker_names should contain exactly two marker names to invert.")
        if not isinstance(frame_start, int) or not isinstance(frame_end, int):
            raise TypeError("frame_start and frame_end should be integers.")
        if frame_start < 0 or frame_end >= self.markers.shape[2] or frame_end < frame_start:
            raise ValueError(f"Invalid frame range specified [{frame_start}, {frame_end}].")

        marker_indices = [self.marker_names.index(name) for name in marker_names]

        # Keep a copy of the old marker data
        old_first_marker_data = self.markers[:, marker_indices[0], frame_start : frame_end + 1].copy()

        # Make the modifications
        self.markers[:, marker_indices[0], frame_start : frame_end + 1] = self.markers[
            :, marker_indices[1], frame_start : frame_end + 1
        ].copy()
        self.markers[:, marker_indices[1], frame_start : frame_end + 1] = old_first_marker_data
        self.c3d["data"]["points"] = self.markers

        return

    def remove_label(self, marker_name: str, frame_start: int, frame_end: int):
        """
        Remove the label of one marker between frame_start and frame_end.
        """

        if not isinstance(marker_name, str):
            raise TypeError("marker_name should be the name of the marker to remove labeling from.")
        if not isinstance(frame_start, int) or not isinstance(frame_end, int):
            raise TypeError("frame_start and frame_end should be integers.")
        if frame_start < 0 or frame_end >= self.markers.shape[2] or frame_end < frame_start:
            raise ValueError(f"Invalid frame range specified [{frame_start}, {frame_end}].")

        marker_index = self.marker_names.index(marker_name)

        # Make the modifications
        self.markers[:3, marker_index, frame_start : frame_end + 1] = np.nan
        self.c3d["data"]["points"] = self.markers

        return

    def animate_c3d(self):
        try:
            import pyorerun

            pyorerun.c3d(
                self.c3d_path, show_forces=False, show_events=False, marker_trajectories=False, show_marker_labels=True
            )
        except ImportError:
            raise ImportError("pyorerun is not installed. Please install it to animate the C3D data.")

    def save_c3d(self, output_c3d_path: str):
        """
        Save the modified c3d file with the new marker labeling.
        """
        self.c3d.write(output_c3d_path)
        print(f"C3D file saved to {output_c3d_path}")
