#!/bin/bash

# Variant of gen.sh that runs the example search workflow on a SLURM
# cluster via Pegasus' glite/batch style. The example executables.ini pins
# some jobs to specific condorpool sites to test them; the config-overrides
# below move everything to the slurm site instead.

set -e

pycbc_make_offline_search_workflow \
--workflow-name gw \
--output-dir output \
--config-files analysis.ini plotting.ini executables.ini injections_minimal.ini slurm.ini \
--config-overrides results_page:output-path:$(pwd)/html \
                   "pegasus_profile:pycbc|primary_site:slurm" \
                   "pegasus_profile-coinc:pycbc|site:slurm" \
                   "pegasus_profile-inspiral:pycbc|site:slurm" \
                   "pegasus_profile-results_page:pycbc|site:slurm" \
--plan-now
