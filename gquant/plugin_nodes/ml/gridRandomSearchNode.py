from ..util.contextCompositeNode import ContextCompositeNode
from gquant.dataframe_flow.portsSpecSchema import ConfSchema

__all__ = ["GridRandomSearchNode"]


class GridRandomSearchNode(ContextCompositeNode):

    def conf_schema(self):
        # get's the input when it gets the conf
        input_columns = self.get_input_columns()
        json = {}
        if self.INPUT_CONFIG in input_columns:
            conf = input_columns[self.INPUT_CONFIG]
            if 'context' in conf:
                json = {
                    "definitions": {
                        "number": {
                            "type": "object",
                            "oneOf": [
                                  {
                                      "title": 'randn',
                                      "description": """Wraps tune.sample_from around np.random.randn.
                                       tune.randn(10) is equivalent to np.random.randn(10)""",
                                      "properties": {
                                          "function": {
                                              "type": "string",
                                              "enum": ['randn'],
                                              "default": 'randn'
                                          },
                                          "args": {
                                              "type": "array",
                                              "items": [
                                                      {
                                                          "type": "number",
                                                          "default": 1.0
                                                      }
                                              ]
                                          }
                                      }
                                  },
                                {
                                    "title": "uniform",
                                      "description": """Wraps tune.sample_from around np.random.uniform""",
                                      "properties": {
                                          "function": {
                                              "type": "string",
                                              "enum": ['uniform'],
                                              "default": 'uniform'
                                          },
                                          "args": {
                                              "type": "array",
                                              "items": [
                                                      {
                                                          "type": "number",
                                                          "default": 0.0
                                                      },
                                                  {
                                                          "type": "number",
                                                          "default": 10.0
                                                  }
                                              ]
                                          }

                                      }
                                  },
                                {
                                    "title": "loguniform",
                                      "description": """Sugar for sampling in different orders of magnitude.,
                                      parameters, min_bound – Lower boundary of the output interval,
                                      max_bound (float) – Upper boundary of the output interval (1e-2), 
                                      base – Base of the log. Defaults to 10.""",
                                      "properties": {
                                          "function": {
                                              "type": "string",
                                              "enum": ['loguniform'],
                                              "default": 'loguniform'
                                          },
                                          "args": {
                                              "type": "array",
                                              "items": [
                                                      {
                                                          "type": "number",
                                                          "default": 1e-4
                                                      },
                                                  {
                                                          "type": "number",
                                                          "default": 1e-2
                                                  },
                                                  {
                                                          "type": "number",
                                                          "default": 10
                                                  }
                                              ]
                                          }
                                      }
                                  },
                                {
                                    "title": "choice",
                                   "description": """Wraps tune.sample_from around random.choice.""",
                                      "properties": {
                                          "function": {
                                              "type": "string",
                                              "enum": ['choice'],
                                              "default": 'choice'
                                          },
                                          "args": {
                                              "type": "array",
                                              "items": {
                                                      "type": "number"
                                              }
                                          }
                                      }
                                  },
                                {
                                    "title": "grid_search",
                                   "description": """Convenience method for specifying grid search over a value.""",
                                      "properties": {
                                          "function": {
                                              "type": "string",
                                              "enum": ['grid_search'],
                                              "default": 'grid_search'
                                          },
                                          "args": {
                                              "type": "array",
                                              "items": {
                                                      "type": "number"
                                              }
                                          }
                                      }
                                  }
                            ]
                        },
                        "string": {
                            "type": "object",
                            "oneOf": [
                                  {
                                    "title": "choice",
                                   "description": """Wraps tune.sample_from around random.choice.""",
                                      "properties": {
                                          "function": {
                                              "type": "string",
                                              "enum": ['choice'],
                                              "default": 'choice'
                                          },
                                          "args": {
                                              "type": "array",
                                              "items": {
                                                      "type": "string"
                                              }
                                          }

                                      }
                                  },
                                {
                                    "title": "grid_search",
                                   "description": """Convenience method for specifying grid search over a value.""",
                                      "properties": {
                                          "function": {
                                              "type": "string",
                                              "enum": ['grid_search'],
                                              "default": 'grid_search'
                                          },
                                          "args": {
                                              "type": "array",
                                              "items": {
                                                      "type": "string"
                                              }
                                          }

                                      }
                                  }
                            ]
                        }
                    },
                    "description": """
                    Use Tune to specify a grid search or random search for a context composite node.
                    """,
                    "type": "object",
                    "properties": {
                        "parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                    }
                                },
                                "dependencies": {
                                    "name": {
                                        "oneOf": [],
                                    }
                                }
                            },
                        },
                        "tune": {
                            "type": "object",
                            "properties": {
                                "local_dir": {
                                    "type": "string",
                                    "description": """
                                     Local dir to save training results to.
                                    """,
                                    "default": "./ray"
                                },
                                "name": {
                                    "type": "string",
                                    "description": """Name of experiment""",
                                    "default": "exp"
                                },
                                "num_samples": {
                                    "type": "number",
                                    "description": """
                                     Number of times to sample from the hyperparameter 
                                     space. Defaults to 1. If grid_search is provided 
                                     as an argument, the grid will be repeated
                                      num_samples of times.
                                    """,
                                    "default": 1
                                },
                                "resources_per_trial": {
                                    "type": "object",
                                    "description": """
                                    Machine resources to allocate per trial, e.g.
                                     {"cpu": 64, "gpu": 8}. Note that GPUs will 
                                     not be assigned unless you specify them here.
                                      Defaults to 1 CPU and 0 GPUs 
                                    """,
                                    "properties": {
                                        "cpu": {
                                            "type": 'number',
                                            "default": 1
                                        },
                                        "gpu": {
                                            "type": 'number',
                                            "default": 1
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
                context = conf['context']
                json['properties']['parameters'][
                    'items']['properties']['name']['enum'] = list(context.keys())
                options = json['properties']['parameters'][
                    'items']['dependencies']['name']['oneOf']
                for var in context.keys():
                    if (context[var]['type'] == 'number' or
                            context[var]['type'] == 'string'):
                        obj = {
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": [var]
                                },
                                "search": {
                                    "$ref": "#/definitions/{}".format(
                                        context[var]['type'])
                                }
                            }
                        }
                        options.append(obj)
        ui = {
            "tune": {
                "local_dir": {"ui:widget": "PathSelector"}
            }
        }
        out_schema = ConfSchema(json=json, ui=ui)
        return out_schema
