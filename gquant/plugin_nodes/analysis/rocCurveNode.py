from gquant.dataframe_flow import Node
from bqplot import Axis, LinearScale,  Figure, Lines, PanZoom
import dask_cudf
import cudf
from gquant.dataframe_flow.portsSpecSchema import ConfSchema
from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from sklearn import metrics


class RocCurveNode(Node, _PortTypesMixin):

    def init(self):
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'roc_curve'

    def columns_setup(self):
        cols_required = {}
        if 'label' in self.conf:
            cols_required[self.conf['label']] = None
        if 'prediction' in self.conf:
            cols_required[self.conf['prediction']] = None
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }
        return {self.OUTPUT_PORT_NAME: {}}

    def ports_setup(self):
        return _PortTypesMixin.ports_setup_different_output_type(self,
                                                                 Figure)

    def conf_schema(self):
        json = {
            "title": "ROC Curve Configuration",
            "type": "object",
            "description": """Plot the ROC Curve for binary classification problem.
            """,
            "properties": {
                "points":  {
                    "type": "number",
                    "description": "number of data points for the chart"
                },
                "label":  {
                    "type": "string",
                    "description": "Ground truth label column name"
                },
                "prediction":  {
                    "type": "string",
                    "description": "prediction probablity column"
                },

            },
            "required": ["label", "prediction"],
        }
        ui = {
        }
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['label']['enum'] = enums
            json['properties']['prediction']['enum'] = enums
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        Plot the ROC curve 

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        Figure

        """
        input_df = inputs[self.INPUT_PORT_NAME]
        if isinstance(input_df,  dask_cudf.DataFrame):
            input_df = input_df.compute()  # get the computed value

        if 'points' in self.conf:
            num_points = self.conf['points']
            stride = max(len(input_df) // num_points, 1)
        else:
            stride = 1
        linear_x = LinearScale()
        linear_y = LinearScale()
        yax = Axis(label='True Positive Rate', scale=linear_x,
                   orientation='vertical')
        xax = Axis(label='False Positive Rate', scale=linear_y,
                   orientation='horizontal')
        panzoom_main = PanZoom(scales={'x': [linear_x]})

        label_col = input_df[self.conf['label']].values
        pred_col = input_df[self.conf['prediction']].values
        if isinstance(input_df, cudf.DataFrame):
           fpr, tpr, _ = metrics.roc_curve(label_col.get(),
                                           pred_col.get())
        else:
           fpr, tpr, _ = metrics.roc_curve(label_col,
                                           pred_col)
        curve_label = 'ROC (area = {:.2f})'.format(metrics.auc(fpr, tpr))
        line = Lines(x=fpr[::stride],
                     y=tpr[::stride],
                     scales={'x': linear_x, 'y': linear_y},
                     colors=['blue'], labels=[curve_label],
                     display_legend=True)
        new_fig = Figure(marks=[line], axes=[yax, xax], title='ROC Curve',
                         interaction=panzoom_main)
        return {self.OUTPUT_PORT_NAME: new_fig}
