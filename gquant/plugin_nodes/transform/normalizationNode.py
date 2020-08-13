from gquant.dataframe_flow import Node
from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from gquant.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                   PortsSpecSchema, NodePorts)
from .data_obj import NormalizationData


class NormalizationNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'df_in'
        self.OUTPUT_PORT_NAME = 'df_out'
        self.INPUT_NORM_MODEL_NAME = 'norm_data_in'
        self.OUTPUT_NORM_MODEL_NAME = 'norm_data_out'
        cols_required = {}
        self.required = {
            self.INPUT_PORT_NAME: cols_required,
            self.INPUT_NORM_MODEL_NAME: cols_required
        }

    def ports_setup_from_types(self, types):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: types
            },
            self.INPUT_NORM_MODEL_NAME: {
                port_type: NormalizationData
            }
        }

        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: types
            },
            self.OUTPUT_NORM_MODEL_NAME: {
                port_type: NormalizationData
            }
        }

        input_connections = self.get_connected_inports()
        if self.INPUT_PORT_NAME in input_connections:
            determined_type = input_connections[self.INPUT_PORT_NAME]
            input_ports.update({self.INPUT_PORT_NAME:
                                {port_type: determined_type}})
            output_ports.update({self.OUTPUT_PORT_NAME: {
                                 port_type: determined_type}})
            # connected
            return NodePorts(inports=input_ports,
                             outports=output_ports)
        else:
            return NodePorts(inports=input_ports, outports=output_ports)

    def columns_setup(self):
        if 'columns' in self.conf and self.conf.get('include', True):
            cols_required = {}
            for col in self.conf['columns']:
                cols_required[col] = None
            self.required = {
                self.INPUT_PORT_NAME: cols_required,
                self.INPUT_NORM_MODEL_NAME: cols_required
            }
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            if 'columns' in self.conf:
                if self.conf.get('include', True):
                    included_colums = self.conf['columns']
                else:
                    included_colums = set(enums) - set(self.conf['columns'])
                cols_required = {}
                for col in included_colums:
                    if col in col_from_inport:
                        cols_required[col] = col_from_inport[col]
                self.required = {
                    self.INPUT_PORT_NAME: cols_required,
                    self.INPUT_NORM_MODEL_NAME: cols_required
                }
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport,
                self.OUTPUT_NORM_MODEL_NAME: col_from_inport
            }
            return output_cols
        else:
            col_from_inport = self.required[self.INPUT_PORT_NAME]
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport,
                self.OUTPUT_NORM_MODEL_NAME: col_from_inport
            }
            return output_cols

    def ports_setup(self):
        return _PortTypesMixin.ports_setup(self)

    def conf_schema(self):
        json = {
            "title": "Normalization Node configure",
            "type": "object",
            "description": "Normalize the columns to have zero mean and std 1",
            "properties": {
                "columns":  {
                    "type": "array",
                    "description": """an array of columns that need to
                     be normalized""",
                    "items": {
                        "type": "string"
                    }
                },
                "include":  {
                    "type": "boolean",
                    "description": """if set true, the `columns` need to be 
                    normalized. if false, all dataframe columns except the 
                    `columns` need to be normalized""",
                    "default": True
                },
            },
            "required": ["columns"],
        }
        ui = {}
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
        return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        select the asset based on asset id, which is defined in `asset` in the
        nodes' conf

        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        input_df = inputs[self.INPUT_PORT_NAME]
        if self.conf.get('include', True):
            cols = self.conf['columns']
        else:
            cols = input_df.columns.difference(self.conf['columns'])
        if self.INPUT_NORM_MODEL_NAME in inputs:
            norm_data = inputs[self.INPUT_NORM_MODEL_NAME].data
            means = norm_data['mean']
            stds = norm_data['std']
        else:
            # need to compute the mean and std
            means = input_df[cols].mean()
            stds = input_df[cols].std()
        norm = (input_df[cols] - means) / stds
        col_dict = {i: norm[i] for i in cols}
        norm_df = input_df.assign(**col_dict)
        output = {}
        if self.outport_connected(self.OUTPUT_PORT_NAME):
            output.update({self.OUTPUT_PORT_NAME: norm_df})
        if self.outport_connected(self.OUTPUT_NORM_MODEL_NAME):
            norm_data = {"mean": means, "std": stds}
            payload = NormalizationData(norm_data)
            output.update({self.OUTPUT_NORM_MODEL_NAME: payload})
        return output
