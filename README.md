# forschungsarbeit-appendix
The digital appendix contains the Python scripts which where used to perform the experiments:

- **larsim_chaospy.py** contains the code used for the Monte-Carlo simulation.
- **larsim_adaptive.py** uses sparseSpACE to perform an Adaptive Grid Integration.
- **larsim_combi.py** uses sparseSpACE to perform a Standard-Combination Sparse Grid Integration.

To execute the scripts one has to install the `Larsim_Utility_Set` as well as `sparseSpACE`. Also the Data required for LARSIM and its exectuable have to be present.  
The file `parallel_integrator.py` contains both versions of our new parallel integrator. Our changes are also implemented in the `jonas-treplin-experimental` branch of the sparseSpACE Github project.  
The configuration file `configuration_larsim_updated_lai.json` describes the configuration we used for our experiments.  
The processed data we used in our plots is also provided:
- **\*_results.json** are json serialized results of our test. They include the calculated expectation and variance as well as the number of points used.
- **ada_workpacketlengths.json** is a list of lengths of work packets. We distinguish between overall length and uncached length.
- **combi_workpacketlengths.json** is also a list of work packet lengths. They are divided by level.
- **evaluation_waits.json** is a list of waiting times for each process. This data comes from a run using the adaptive approach.

