# ChemRevOscillator
## Code accompanying Insights into Chemically Fueled Supramolecular Polymers
## https://doi.org/10.1021/acs.chemrev.1c00958

This repository contains both code necessary to reproduce the data presented in the manuscript, as well as an interactive viewer for any curious reader to immediately explore the system. 

To simply view results of simulations, a docker container is provided. 

``` docker run -p 8050:8050 dlivitz/chemrev ```

The source script for the above docker container is available inside the "interactive_viewer" folder. 

The code relies on the python package pyodesys
>Dahlgren, (2018). pyodesys: Straightforward numerical integration of ODE systems from Python. Journal of Open Source Software, 3(21), 490, https://doi.org/10.21105/joss.00490


Latin hypercube sampling (LHS) over a section of the search space is configured, using 500,000 data points. This script may be configured to explore any desired parameter range, but the output size will become large.  
