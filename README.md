# pcsaft-quaternary

LLE phase diagrams for ternary and quaternary systems via PC-SAFT.

## Installation

```bash
pip install -e .
```

## Quick start

```python
import si_units as si
from pcsaft_quaternary import ternary_lle, pseudoternary_lle

# True ternary (3 components)
ternary_lle(
    pure_json="params.json",
    T=298.15 * si.KELVIN, P=1.0 * si.BAR,
    solute="2-phenylethanol", solvent="carvone", diluent="water",
)

# Pseudoternary (4 components)
pseudoternary_lle(
    pure_json="params.json",
    T=298.15 * si.KELVIN, P=1.0 * si.BAR,
    solute="2-phenylethanol", solvent1="thymol", solvent2="carvone",
    diluent="water", solvent_ratio=1.0,
)
```

See [`examples/`](examples/) for complete scripts.

## License

Licensed under either of

* [MIT](https://opensource.org/licenses/MIT)
* [Apache-2.0](https://opensource.org/licenses/Apache-2.0)
* [BSD-2-Clause](https://opensource.org/licenses/BSD-2-Clause)

at your option.
