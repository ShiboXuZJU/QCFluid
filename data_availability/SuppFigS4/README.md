# Here we provide device proformance details for both fluid examples.

Due to gate parameters may drift with time, we calibrated and benchmarked device performance for both fluid examples (diverging flow and 2D vortex).

## Contents
- [diverging_flow_sq_property.xlsx](./diverging_flow_sq_property.xlsx):
Here we provide single-qubit properties for the diverging_flow case.

- [diverging_flow_algorithm_graph_error.xlsx](./diverging_flow_algorithm_graph_error.xlsx):
Here we provide two-qubit gate properties for the diverging flow case. In the experiment, we bias idle qubits to avoid frequency collisions while applying parallel CZ gates (see our [previous work](https://doi.org/10.1038/s41567-024-02529-6) for detail). And each parallel CZ gate pattern (named "algorithm_graph *") will be calibrated individually. There are two sheets in diverging_flow_algorithm_graph_error.xlsx named "CZ" and "algorithm_z" which records the two-qubit CZ gate and single-qubit idle gate cycle errors (errors from reference single-qubit gates are included) extracted from cross-entropy benchmarking (XEB).

- [vortex_sq_property.xlsx](./vortex_sq_property.xlsx):
Here we provide single-qubit properties for the vortex case.

- [vortex_algorithm_graph_error.xlsx](./vortex_algorithm_graph_error.xlsx):
Here we provide two-qubit gate properties for the vortex case.

## NOTE
In FigS2, we only show the single-qubit properties for the diverging flow case. But the CZ errors shown in (c) are alreadly taken into account both examples. And single-qubit idle gate (named "algorithm_z") errors are not shown.