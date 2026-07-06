.. _workflowslurm:

##################################################
Running PyCBC workflows on SLURM clusters
##################################################

============
Introduction
============

PyCBC workflows are planned and executed by `Pegasus WMS
<https://pegasus.isi.edu/>`_, which uses HTCondor DAGMan as its workflow
engine. On clusters that use SLURM as the batch scheduler, jobs can still be
managed through Pegasus: the workflow's jobs are declared on the built-in
``slurm`` site, which uses Pegasus' *glite* (batch) submission style. In this
mode HTCondor's grid universe and the *blahp* translate every job into an
``sbatch`` submission on the local cluster.

The chain of tools is::

    pycbc_make_*_workflow -> pegasus-plan -> HTCondor DAGMan
        -> grid universe / blahp -> sbatch -> SLURM compute node

Both the offline search workflow (``pycbc_make_offline_search_workflow``) and
the inference workflow (``pycbc_make_inference_workflow``) can be run this
way; the ``slurm`` site is part of the standard site catalog that PyCBC
generates for every workflow.

============
Requirements
============

* A **shared filesystem** between the workflow submit host (usually the
  cluster login node) and the SLURM compute nodes. The ``slurm`` site runs in
  Pegasus' ``sharedfs`` data configuration; input/output data, the PyCBC
  installation and the workflow scratch directory must all be visible on the
  compute nodes at the same paths.
* **HTCondor** installed and running on the submit host. Only a minimal
  personal-condor / mini-condor setup is needed (a schedd for DAGMan); jobs
  are forwarded to SLURM by the blahp, which is shipped with HTCondor.
* **Pegasus** installed on the submit host.
* The Pegasus glite attribute scripts installed into the blahp. Pegasus
  provides these and they can be installed with::

      pegasus-configure-glite

  This copies (amongst others) ``slurm_local_submit_attributes.sh`` into the
  blahp's glite directory, which is what turns job requirements (walltime,
  memory, extra arguments, ...) into ``#SBATCH`` directives. Without this
  step jobs will still run, but resource requests will be silently ignored.
* Generate **and start** the workflow from a shell in which your PyCBC
  environment is active. Hierarchical workflows re-run ``pegasus-plan`` at
  runtime in the environment inherited from the submitting shell; with a
  system-wide Pegasus install this invokes ``pegasus-db-admin`` with
  whatever ``python3`` is first in the PATH, which must be able to import
  ``sqlalchemy``. Starting the workflow from a clean shell without the
  virtual environment typically fails the deferred planning stage with
  ``ModuleNotFoundError: No module named 'sqlalchemy'``.

=============
Configuration
=============

To send the workflow's jobs to SLURM, set the primary site to ``slurm`` in
your configuration files (or on the command line with
``--config-overrides``)::

    [pegasus_profile]
    pycbc|primary_site = slurm

    [pegasus_profile-slurm]
    ; sbatch --partition; omit to use the cluster's default partition
    pycbc|partition = mypartition
    ; sbatch --account; omit if your cluster does not use accounting
    pycbc|account = myproject
    ; directory on the shared filesystem to use as workflow scratch space
    pycbc|site-scratch = /path/on/shared/filesystem
    pycbc|unique-scratch =
    ; extra sbatch directives applied to every job on this site
    ; pegasus|glite.arguments = --nodes=1 --exclusive

Individual executables can be kept off the cluster (for example, quick
plotting jobs) by pinning them to the local site in their own section::

    [pegasus_profile-results_page]
    pycbc|site = local

Executables that generate sub-workflow dax files at runtime (in the
offline search these are the minifollowup generators) **must** be pinned
to the local site: Pegasus' deferred planning expects the generated dax
files in the local site's scratch directory, but for a sharedfs site
that stages to itself no transfer puts them there and the sub-workflow
planning fails with ``Expected local file does not exist``::

    [pegasus_profile-foreground_minifollowup]
    pycbc|site = local

    [pegasus_profile-singles_minifollowup]
    pycbc|site = local

    [pegasus_profile-injection_minifollowup]
    pycbc|site = local

--------------------------
Per-job resource requests
--------------------------

With the glite style the HTCondor ``request_cpus``/``request_memory``
profiles are **not** translated into sbatch directives. Instead, use the
``pegasus`` namespace profiles in the ``[pegasus_profile-<executable>]``
section of your configuration file:

* ``pegasus|runtime`` -- expected runtime in seconds, becomes
  ``#SBATCH --time``
* ``pegasus|cores`` -- number of tasks, becomes ``#SBATCH --ntasks``
* ``pegasus|memory`` -- memory per process in MB, becomes
  ``#SBATCH --mem-per-cpu``
* ``pegasus|glite.arguments`` -- arbitrary additional ``#SBATCH`` arguments

For example, to give the inference jobs 16 cores and 40 GB on a single
node::

    [pegasus_profile-inference]
    pegasus|glite.arguments = --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=40G

========
Examples
========

An example of running the offline search workflow example on a SLURM
cluster:

.. literalinclude:: ../../examples/search/gen_slurm.sh
   :language: bash

with the additional configuration file:

.. literalinclude:: ../../examples/search/slurm.ini
   :language: ini

An add-on configuration file for the inference workflow examples, appended
to the end of the ``--config-files`` list of
``pycbc_make_inference_workflow``:

.. literalinclude:: ../../examples/workflow/inference/slurm.ini
   :language: ini
