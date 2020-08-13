from gquant.dataframe_flow import Node
import cudf
import dask_cudf
import xgboost as xgb
import dask
from gquant.dataframe_flow._port_type_node import _PortTypesMixin
from gquant.dataframe_flow.portsSpecSchema import (ConfSchema,
                                                   PortsSpecSchema, NodePorts)
from xgboost import Booster


__all__ = ['TrainXGBoostNode']


class TrainXGBoostNode(Node, _PortTypesMixin):

    def init(self):
        _PortTypesMixin.init(self)
        self.INPUT_PORT_NAME = 'in'
        self.OUTPUT_PORT_NAME = 'model_out'
        cols_required = {}
        self.required = {
            self.INPUT_PORT_NAME: cols_required
        }

    def ports_setup_from_types(self, types):
        port_type = PortsSpecSchema.port_type
        input_ports = {
            self.INPUT_PORT_NAME: {
                port_type: types
            }
        }
        output_ports = {
            self.OUTPUT_PORT_NAME: {
                port_type: Booster
            }
        }
        input_connections = self.get_connected_inports()
        if self.INPUT_PORT_NAME in input_connections:
            determined_type = input_connections[self.INPUT_PORT_NAME]
            input_ports.update({self.INPUT_PORT_NAME:
                                {port_type: determined_type}})
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
                self.INPUT_PORT_NAME: cols_required
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
                if ('target' in self.conf and
                        self.conf['target'] in col_from_inport):
                    cols_required[self.conf['target']
                                  ] = col_from_inport[self.conf['target']]
                self.required = {
                    self.INPUT_PORT_NAME: cols_required,
                }
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport,
            }
            return output_cols
        else:
            col_from_inport = {}
            output_cols = {
                self.OUTPUT_PORT_NAME: col_from_inport,
            }
            return output_cols

    def ports_setup(self):
        types = [cudf.DataFrame,
                 dask_cudf.DataFrame]
        return self.ports_setup_from_types(types)

    def conf_schema(self):
        json = {
            "title": "XGBoost Node configure",
            "type": "object",
            "description": """Split the data into training and testing based on
             'train_data', train a XGBoost model based on the training data, 
             make predictions for all the data points, compute the trading.
            """,
            "properties": {
                "num_of_rounds": {
                    "type": "number",
                    "description": """The number of rounds for boosting""",
                    "default": 100
                },
                "target":  {
                    "type": "string",
                    "description": "the column used as dependent variable"
                },
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": """columns in the input dataframe that
        are considered as training features or not depending on `include` flag."""
                },
                "include":  {
                    "type": "boolean",
                    "description": """if set true, the `columns` are treated as independent variables.
                     if false, all dataframe columns are independent variables except the `columns`""",
                    "default": True
                },
                "xgboost_parameters": {
                    "type": "object",
                    "description": "xgoobst parameters",
                    "properties": {
                        'eta': {
                            "type": "number",
                            "description": "Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.",
                            "default": 0.3
                        },
                        'min_child_weight': {
                            "type": "number",
                            "description": "Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.",
                            "default": 1
                        },
                        'subsample': {
                            "type": "number",
                            "description": "Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.",
                            "default": 1
                        },
                        'sampling_method': {
                            "type": "string",
                            "description": "The method to use to sample the training instances.",
                            "enum": ["uniform", "gradient_based"],
                            "default": "uniform",
                        },
                        'colsample_bytree': {
                            "type": "number",
                            "description": "is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.",
                            "default": 1
                        },
                        'colsample_bylevel': {
                            "type": "number",
                            "description": "is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree",
                            "default": 1
                        },
                        'colsample_bynode': {
                            "type": "number",
                            "description": " is the subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level.",
                            "default": 1
                        },
                        'max_depth': {
                            "type": "number",
                            "description": "Maximum depth of a tree.",
                            "default": 8
                        },
                        "max_leaves": {
                            "type": "number",
                            "description": "maximum number of tree leaves",
                            "default": 2**8
                        },
                        "grow_policy": {
                            "type": "string",
                            "enum": ["depthwise", "lossguide"],
                            "description": "Controls a way new nodes are added to the tree. Currently supported only if tree_method is set to hist.",
                            "default": "depthwise"
                        },
                        "gamma": {
                            "type": "number",
                            "description": """Minimum loss reduction required
                            to make a further partition on a leaf node of the
                            tree.""",
                            "default": 0
                        },
                        "lambda": {
                            "type": "number",
                            "description": """L2 regularization term on weights. Increasing this value will make model more conservative.""",
                            "default": 1
                        },
                        "alpha": {
                            "type": "number",
                            "description": """L1 regularization term on weights. Increasing this value will make model more conservative.""",
                            "default": 0
                        },
                        "tree_method": {
                            "type": "string",
                            "description": """The tree construction algorithm used in XGBoost""",
                            "enum": ["auto", "exact", "approx", 'hist', 'gpu_hist'],
                            "default": "auto"
                        },
                        "single_precision_histogram": {
                            "type": "boolean",
                            "description": "for hist and `gpu_hist tree method, Use single precision to build histograms instead of double precision.",
                            "default": False
                        },
                        "deterministic_histogram": {
                            "type": "boolean",
                            "description": "for gpu_hist tree method, Build histogram on GPU deterministically. Histogram building is not deterministic due to the non-associative aspect of floating point summation. We employ a pre-rounding routine to mitigate the issue, which may lead to slightly lower accuracy. Set to false to disable it.",
                            "default": False
                        },
                        "objective": {
                            "type": "string",
                            "enum": ["reg:squarederror", "reg:squaredlogerror",
                                     "reg:logistic", "reg:pseudohubererror",
                                     "binary:logistic", "binary:logitraw",
                                     "binary:hinge", "count:poisson", 
                                     "survival:cox", "survival:aft",
                                     "aft_loss_distribution", "multi:softmax",
                                     "multi:softprob", "rank:pairwise", "rank:ndcg",
                                     "rank:map", "reg:gamma", "reg:tweedie"
                                     ],
                            "description": """Specify the learning task and
                            the corresponding learning objective.""",
                            "default": "reg:squarederror"
                        }
                    }
                }
            },
            "required": ["target", "num_of_rounds", "columns"],
        }
        ui = {}
        input_columns = self.get_input_columns()
        if self.INPUT_PORT_NAME in input_columns:
            col_from_inport = input_columns[self.INPUT_PORT_NAME]
            enums = [col for col in col_from_inport.keys()]
            json['properties']['columns']['items']['enum'] = enums
            json['properties']['target']['enum'] = enums
            return ConfSchema(json=json, ui=ui)
        else:
            return ConfSchema(json=json, ui=ui)

    def process(self, inputs):
        """
        The process is doing following things:
            1. split the data into training and testing based on provided
               conf['train_date']. If it is not provided, all the data is
               treated as training data.
            2. train a XGBoost model based on the training data
            3. Make predictions for all the data points including training and
               testing.
            4. From the prediction of returns, compute the trading signals that
               can be used in the backtesting.
        Arguments
        -------
         inputs: list
            list of input dataframes.
        Returns
        -------
        dataframe
        """
        dxgb_params = {
                'max_depth':         8,
                'max_leaves':        2 ** 8,
                'tree_method':       'gpu_hist',
                'objective':         'reg:squarederror',
        }
        # num_of_rounds = 100
        if 'xgboost_parameters' in self.conf:
            dxgb_params.update(self.conf['xgboost_parameters'])
        input_df = inputs[self.INPUT_PORT_NAME]
        if self.conf.get('include', True):
            included_colums = self.conf['columns']
        else:
            included_colums = set(input_df.columns) - set(self.conf['columns'])
        train_cols = list(set(included_colums) - set([self.conf['target']]))

        if isinstance(input_df, dask_cudf.DataFrame):
            # get the client
            client = dask.distributed.client.default_client()
            train = input_df[train_cols]
            target = input_df[self.conf['target']]
            dmatrix = xgb.dask.DaskDMatrix(client, train, label=target)
            bst = xgb.dask.train(client, dxgb_params, dmatrix,
                                 num_boost_round=self.conf["num_of_rounds"])
            bst = bst['booster']
        elif isinstance(input_df, cudf.DataFrame):
            train = input_df[train_cols]
            target = input_df[self.conf['target']]
            dmatrix = xgb.DMatrix(train, label=target)
            bst = xgb.train(dxgb_params, dmatrix,
                            num_boost_round=self.conf["num_of_rounds"])
        return {self.OUTPUT_PORT_NAME: bst}
