import ipywidgets as w
import bqplot.pyplot as plt


class DiagnosticPlots(w.Box):
    """Dashboard containing the diagnostic plots"""

    def __init__(self, *args, **kwargs):
        self.widgets_layout = w.Box()
        self.build_widgets()
        kwargs["children"] = [self.widgets_layout]
        super(DiagnosticPlots, self).__init__(*args, **kwargs)

    def update(self, *args, **kwargs):
        """
        uses the trained model to predict on test data and updates the plots
        """
        self.y_true = kwargs["y_true"]
        self.model = kwargs["model"]
        self.X = kwargs["X"]

    def build_widgets(self, *args, **kwargs):
        """returns the list of widgets/plots"""
        pass

    def set_layout(self, layout):
        """set the layout of the plots"""
        self.widgets_layout.layout = layout

    def reset(self):
        """empties all the figures"""
        pass


class RegressionDiagnosticPlots(DiagnosticPlots):
    """
    Dashboard containing the basic diagnostic plot for
    regression problems: scatter of residuals vs fitted values
    """

    def update(self, *args, **kwargs):
        super(RegressionDiagnosticPlots, self).update(*args, **kwargs)
        self.y_pred = self.model.predict(self.X).squeeze()
        self.residuals = self.y_true - self.y_pred

        # update the residual plot
        with self.residuals_plot.hold_sync():
            self.residuals_plot.x = self.y_pred[:2000]
            self.residuals_plot.y = self.residuals[:2000]

    def build_widgets(self, *args, **kwargs):
        # residuals plot
        self.residuals_fig = plt.figure(
            title="Residuals vs Predicted Values",
            layout=w.Layout(
                width="960px", height="600px",
                overflow_x="hidden", overflow_y="hidden"
            ),
        )

        axes_options = {
            "y": {"label": "Residual", "tick_format": "0.1f"},
            "x": {"label": "Predicted Value"},
        }

        self.residuals_plot = plt.scatter(
            [],
            [],
            colors=["yellow"],
            default_size=16,
            stroke="black",
            axes_options=axes_options,
        )
        # zero line
        plt.hline(level=0, colors=["limegreen"], stroke_width=3)

        self.widgets_layout = w.HBox([self.residuals_fig])

    def reset(self):
        """empties all the figures"""
        self.residuals_plot.x = []
