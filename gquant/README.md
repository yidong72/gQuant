# gQuant - Graph Computation Toolkit

## What is gQuant?

gQuant is a tool that helps you to organize the workflows. 

1. It define a TaskGraph file format `.gq.yaml` that describes the workflow. It can be edited easily by `gquantlab` JupyterLab plugin.
2. Dynamically compute the input-output ports compatibility, dataframe columns names and types, ports types to prevent connection errors. 
3. Nodes can have multiple output ports that can be used to generate different output types. E.g. some data loader Node provides both `cudf` and `dask_cudf` output ports. The multiple GPUs distributed computation computation is automatically enabled by switching to the `dask_cudf` output port. 
4. Provides the standard API to extend your computation Nodes.
5. The composite node can encapsulate the TaskGraph into a single node for easy reuse. The composite node can be exported as a regular gQuant node without any coding.
6. gQuant can be extended by writing a plugin with a set of nodes for a particular domain. Check `plugins` for examples.

These examples can be used as-is or, as they are open source, can be extended to suit your environments.

## Binary pip installation

To install the gQuant graph computation library, first install the dependence libraries:
```bash
conda install dask networkx python-graphviz ruamel.yaml pandas
```

Then install `gquant`:
```bash
pip install gquant
```
Or install `gquant` at the root directory:
```bash
pip install .
```

gQuant node plugins can be registered in two ways: 

  1. Register the plugin in `gquantrc` file. Check the `System environment` for details
  2. Write a external plugin using 'entry point' to register it. Check the `external` directory for details

## System environment 

There are a few system environment that the user can overwrite. 

The custom module files are specified in the `gquantrc` file. `GQUANT_CONFIG` enviroment variable points to the location of this file. By default, it points to 
`$CWD\gquantrc`. 

In the example `gquantrc`, system environment variable `MODULEPATH` is used to point to the paths of the module files.
To start the jupyterlab, please make sure `MODULEPATH` is set properly. 
