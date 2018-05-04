from bqplot import *

class ConfusionMatrix(Figure):
    def __init__(self, *args, **kwargs):
        self.conf_mat = kwargs['matrix']
        self.title = kwargs.get('title', '')
        n = len(self.conf_mat)
        self.labels = kwargs.get('labels', np.arange(n))
        row_scale = OrdinalScale(reverse=True)
        col_scale = OrdinalScale()
        color_scale = ColorScale(scheme='Greens')
        row_axis = Axis(scale=row_scale, orientation='vertical', label='Actual Label')
        col_axis = Axis(scale=col_scale, label='Predicted Label')
        
        self.conf_mat_grid = GridHeatMap(
            column=self.labels,
            row=self.labels,
            color=(self.conf_mat ** .3),
            scales={'row': row_scale, 'column': col_scale, 'color': color_scale},
            interactions={'click': 'select'},
            anchor_style={'stroke': 'red', 'stroke-width': 3},
            selected_style={'stroke': 'red'})

        y, x, text = zip(*[(self.labels[i],
                            self.labels[j],
                            str(self.conf_mat[i, j])) for i in range(n) for j in range(n)])

        self.grid_labels = Label(x=x, y=y, text=text,
                                 scales={'x': col_scale, 
                                         'y': row_scale},
                                 font_size=16,
                                 align='middle',
                                 colors=['black'])

        self.title = 'Confusion Matrix'
        self.marks = [self.conf_mat_grid, self.grid_labels]
        self.padding_y = 0.0
        self.axes = [row_axis, col_axis]
        self.fig_margin = dict(left=50, top=40, bottom=40, right=20)
        self.layout.width = '460px'
        self.layout.height = '400px'
        
        super(ConfusionMatrix, self).__init__(*args, **kwargs)