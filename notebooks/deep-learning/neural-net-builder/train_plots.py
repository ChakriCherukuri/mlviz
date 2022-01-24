import numpy as np
import tensorflow
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers

import ipywidgets as w

import bqplot as bq
import bqplot.pyplot as plt


class TrainingPlotsDashboard(w.Box):
    """
    dashboard for training plots (loss/accuracy curves)
    """

    def __init__(self, *args, **kwargs):
        self.epochs = kwargs.get("epochs", 100)
        self.width = kwargs.get("width", 960)
        self.height = kwargs.get("height", 500)
        self.tab_layout = w.Layout(
            width=str(self.width) + "px", height=str(self.height) + "px"
        )

        self.widgets_layout = w.Box()
        self.build_widgets()
        kwargs["children"] = [self.widgets_layout]
        super(TrainingPlotsDashboard, self).__init__(*args, **kwargs)

    def build_widgets(self):
        # loss curve
        self.loss_fig = plt.figure(
            title="Loss Curve", layout={"width": "600px"}
        )
        axes_options = {
            "y": {
                "label": "Loss",
                "tick_format": ".1f",
                "label_offset": "-1em",
                "label_location": "end",
            },
            "x": {"label": "Epochs"},
        }
        self.loss_plot = plt.plot(
            [],
            [],
            colors=["orangered", "limegreen"],
            axes_options=axes_options,
            display_legend=True,
            labels=["Train", "Test"],
        )

        # accuracy curve
        self.accuracy_fig = plt.figure(
            title="Accuracy Curve", layout={"width": "600px"}
        )
        plt.scales(scales={"y": bq.LinearScale(min=0, max=1)})
        axes_options = {
            "y": {
                "label": "R Square",
                "tick_format": ".1%",
                "label_offset": "-1em",
                "label_location": "end",
            },
            "x": {"label": "Epochs"},
        }
        self.accuracy_plot = plt.plot(
            [],
            [],
            colors=["orangered", "limegreen"],
            axes_options=axes_options,
            display_legend=True,
            labels=["Train", "Test"],
        )

        self.progress_bar = w.IntProgress(
            description="Training Progress",
            min=0,
            max=(self.epochs - 1),
            style={"description_width": "initial"},
        )

        # first tab components: loss/accuracy curves
        self.plots_layout = w.VBox(
            [self.progress_bar, w.HBox([self.loss_fig, self.accuracy_fig])],
            layout=self.tab_layout,
        )

        axes_options = {
            "x": {"grid_lines": "none", "tick_format": ".1f", "num_ticks": 5},
            "y": {"grid_lines": "none", "num_ticks": 6},
        }

        # weights hist
        self.weights_fig = plt.figure(title="Weights")
        hist_args = dict(
            sample=np.array([]),
            colors=["salmon"],
            axes_options=axes_options,
            stroke="white",
        )
        self.weights_hist = plt.bin(**hist_args)

        # biases hist
        self.biases_fig = plt.figure(title="Biases")
        self.biases_hist = plt.bin(**hist_args)

        # activations hist
        self.activations_fig = plt.figure(title="Activations")
        self.activations_hist = plt.bin(**hist_args)

        for fig in [self.weights_fig, self.biases_fig, self.activations_fig]:
            fig.layout.width = "400px"
            fig.layout.height = "350px"

        self.layers_dd = w.Dropdown(description="Layers")
        self.epoch_slider = w.IntSlider(description="Epoch", min=1, step=1)

        # second tab components: distributions of weights/biases/activations
        self.distributions_layout = w.VBox(
            [
                self.layers_dd,
                self.epoch_slider,
                w.HBox(
                    [self.weights_fig, self.biases_fig, self.activations_fig]
                ),
            ],
            layout=self.tab_layout,
        )

        self.tab = w.Tab(
            [self.plots_layout, self.distributions_layout],
            _titles={0: "Loss/Accuracy Plots", 1: "Distributions"},
        )

        self.widgets_layout = self.tab

    def clear_plots(self):
        self.loss_plot.x = []
        self.accuracy_plot.x = []

        for hist in [
            self.weights_hist, self.biases_hist, self.activations_hist
        ]:
            hist.sample = np.array([])


class TrainingCallback(Callback):
    def __init__(self, *args, **kwargs):
        self.dashboard = kwargs["dashboard"]
        self.X_train = kwargs["X_train"]
        self.y_train = kwargs["y_train"]

    def on_train_begin(self, epoch, logs={}):
        self.activation_layers = tensorflow.keras.Model(
            inputs=self.model.inputs,
            outputs=[
                layer.output
                for layer in self.model.layers
                if isinstance(layer, layers.Activation)
            ],
        )

        self.epochs = []
        self.train_loss = []
        self.test_loss = []

        self.train_acc = []
        self.test_acc = []

        self.epoch_weights = []
        self.epoch_biases = []
        self.epoch_activations = []

    def on_epoch_end(self, epoch, logs={}):
        self.dashboard.progress_bar.value = epoch
        self.epochs.append(epoch + 1)

        # weights of all dense layers except the output layer
        weights_biases = [
            layer.get_weights()
            for layer in self.model.layers
            if isinstance(layer, layers.Dense)
        ]
        weights = [w[0].ravel() for w in weights_biases]
        biases = [w[1].ravel() for w in weights_biases]
        self.epoch_weights.append(weights)
        self.epoch_biases.append(biases)

        # activations
        activation_vals = [
            a.numpy().ravel() for a in self.activation_layers(self.X_train)
        ]
        self.epoch_activations.append(activation_vals)

        # loss and accuracy values
        self.train_loss.append(logs["loss"])

        if "acc" in logs:
            self.train_acc.append(logs["acc"])
        elif "r_square" in logs:
            self.train_acc.append(logs["r_square"])

        if "val_loss" in logs:
            self.test_loss.append(logs["val_loss"])

        if "val_acc" in logs:
            self.test_acc.append(logs["val_acc"])
        elif "val_r_square" in logs:
            self.test_acc.append(logs["val_r_square"])

        # update dashboard plots

        # loss plot
        with self.dashboard.loss_plot.hold_sync():
            self.dashboard.loss_plot.x = self.epochs
            if len(self.test_loss) > 0:
                self.dashboard.loss_plot.y = [self.train_loss, self.test_loss]
            else:
                self.dashboard.loss_plot.y = self.train_loss

        # accuracy plot
        with self.dashboard.accuracy_plot.hold_sync():
            self.dashboard.accuracy_plot.x = self.epochs

            if len(self.test_acc) > 0:
                self.dashboard.accuracy_plot.y = [
                    self.train_acc, self.test_acc
                ]
            else:
                self.dashboard.accuracy_plot.y = self.train_acc

    def on_train_end(self, logs={}):
        # get the count of hidden layers
        hiddel_layers = [
            layer for layer in self.model.layers
            if isinstance(layer, layers.Dense)
        ]
        num_layers = len(hiddel_layers)
        epochs = len(self.epoch_weights)
        self.dashboard.layers_dd.options = [
            "Layer " + str(i + 1) for i in range(num_layers - 1)
        ] + ["Output Layer"]
        self.dashboard.epoch_slider.max = epochs

        self.dashboard.layers_dd.observe(self.update_histograms, "value")
        self.dashboard.epoch_slider.observe(self.update_histograms, "value")
        self.update_histograms(None)

    def update_histograms(self, *args):
        num_layers = len(self.dashboard.layers_dd.options)
        selected_layer = self.dashboard.layers_dd.value
        layer_idx = self.dashboard.layers_dd.options.index(selected_layer)
        epoch_idx = self.dashboard.epoch_slider.value - 1

        with self.dashboard.weights_hist.hold_sync():
            self.dashboard.weights_hist.sample = self.epoch_weights[epoch_idx][
                layer_idx
            ]
            self.dashboard.weights_hist.bins = 25

        with self.dashboard.biases_hist.hold_sync():
            self.dashboard.biases_hist.sample = \
                self.epoch_biases[epoch_idx][layer_idx]
            self.dashboard.biases_hist.bins = 25

        # update histograms except for the last layer which is output layer
        if layer_idx != num_layers - 1:
            with self.dashboard.activations_hist.hold_sync():
                self.dashboard.activations_hist.sample = \
                    self.epoch_activations[epoch_idx][layer_idx]
                self.dashboard.activations_hist.bins = 25
        else:
            with self.dashboard.activations_hist.hold_sync():
                self.dashboard.activations_hist.sample = np.array([0])
                self.dashboard.activations_hist.bins = 1
