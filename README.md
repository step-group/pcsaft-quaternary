# pcsaft-quaternary

The core scripts to make pseudoternary phase diagrams with PC(P)-SAFT.

## Usage

The main script is `main.py`. It will generate a pseudoternary phase diagram for the quaternary system specified in `input.json`. The output will be saved as `output.png`.

## Input

The input file `input.json` should contain the following fields:

- `components`: A list of the four components in the quaternary system. Each component should be specified as a string, e.g. "water", "ethanol", "acetone", "benzene".
- `temperature`: The temperature at which to calculate the phase diagram, in Kelvin.
- `pressure`: The pressure at which to calculate the phase diagram, in bar.
