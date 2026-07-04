# Copyright (C) 2021 The PyCBC development team

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
""" This module provides default site catalogs, which should be suitable for
most use cases. You can override individual details here. It should also be
possible to implement a new site, but not sure how that would work in practice.
"""

import logging
import os.path
import tempfile
import urllib.parse
from shutil import which
from urllib.parse import urljoin
from urllib.request import pathname2url

from Pegasus.api import Directory, FileServer, Site, Operation, Namespace
from Pegasus.api import Arch, OS, SiteCatalog
from Pegasus.api import Grid, Scheduler, SupportedJobs

from pycbc.version import last_release, version, release  # noqa


logger = logging.getLogger('pycbc.workflow.pegasus_sites')

if release:
    sing_version = version
else:
    sing_version = last_release

# NOTE urllib is weird. For some reason it only allows known schemes and will
# give *wrong* results, rather then failing, if you use something like gsiftp
# We can add schemes explicitly, as below, but be careful with this!
urllib.parse.uses_relative.append('gsiftp')
urllib.parse.uses_netloc.append('gsiftp')

KNOWN_SITES = ['local', 'condorpool_symlink',
               'condorpool_copy', 'condorpool_shared', 'osg', 'slurm']


def add_site_pegasus_profile(site, cp):
    """Add options from [pegasus_profile] in configparser to site"""
    # Add global profile information
    if cp.has_section('pegasus_profile'):
        add_ini_site_profile(site, cp, 'pegasus_profile')
    # Add site-specific profile information
    if cp.has_section('pegasus_profile-{}'.format(site.name)):
        add_ini_site_profile(site, cp, 'pegasus_profile-{}'.format(site.name))


def add_ini_site_profile(site, cp, sec):
    """Add options from sec in configparser to site"""
    for opt in cp.options(sec):
        namespace = opt.split('|')[0]
        if namespace in ('pycbc', 'container'):
            continue

        value = cp.get(sec, opt).strip()
        key = opt.split('|')[1]
        site.add_profiles(Namespace(namespace), key=key, value=value)


def add_local_site(sitecat, cp, local_path, local_url):
    """Add the local site to site catalog"""
    # local_url must end with a '/'
    if not local_url.endswith('/'):
        local_url = local_url + '/'

    local = Site("local", arch=Arch.X86_64, os_type=OS.LINUX)
    add_site_pegasus_profile(local, cp)

    local_dir = Directory(Directory.SHARED_SCRATCH,
                          path=os.path.join(local_path, 'local-site-scratch'))
    local_file_serv = FileServer(urljoin(local_url, 'local-site-scratch'),
                                 Operation.ALL)
    local_dir.add_file_servers(local_file_serv)
    local.add_directories(local_dir)

    local.add_profiles(Namespace.PEGASUS, key="style", value="condor")
    sitecat.add_sites(local)


def add_condorpool_symlink_site(sitecat, cp):
    """Add condorpool_symlink site to site catalog"""
    site = Site("condorpool_symlink", arch=Arch.X86_64, os_type=OS.LINUX)
    add_site_pegasus_profile(site, cp)

    site.add_profiles(Namespace.PEGASUS, key="style", value="condor")
    site.add_profiles(Namespace.PEGASUS, key="data.configuration",
                      value="nonsharedfs")
    site.add_profiles(Namespace.PEGASUS, key='transfer.bypass.input.staging',
                      value="true")
    site.add_profiles(Namespace.PEGASUS, key='auxillary.local',
                      value="true")
    site.add_profiles(Namespace.CONDOR, key="My.OpenScienceGrid",
                      value="False")
    site.add_profiles(Namespace.CONDOR, key="should_transfer_files",
                      value="Yes")
    site.add_profiles(Namespace.CONDOR, key="when_to_transfer_output",
                      value="ON_EXIT_OR_EVICT")
    site.add_profiles(Namespace.CONDOR, key="My.DESIRED_Sites",
                      value='"nogrid"')
    site.add_profiles(Namespace.CONDOR, key="My.IS_GLIDEIN",
                      value='"False"')
    site.add_profiles(Namespace.CONDOR, key="My.flock_local",
                      value="True")
    site.add_profiles(Namespace.DAGMAN, key="retry", value="2")
    sitecat.add_sites(site)


def add_condorpool_copy_site(sitecat, cp):
    """Add condorpool_copy site to site catalog"""
    site = Site("condorpool_copy", arch=Arch.X86_64, os_type=OS.LINUX)
    add_site_pegasus_profile(site, cp)

    site.add_profiles(Namespace.PEGASUS, key="style", value="condor")
    site.add_profiles(Namespace.PEGASUS, key="data.configuration",
                      value="condorio")
    site.add_profiles(Namespace.PEGASUS, key='transfer.bypass.input.staging',
                      value="true")
    # This explicitly disables symlinking
    site.add_profiles(Namespace.PEGASUS, key='nosymlink',
                      value=True)
    site.add_profiles(Namespace.PEGASUS, key='auxillary.local',
                      value="true")
    site.add_profiles(Namespace.CONDOR, key="My.OpenScienceGrid",
                      value="False")
    site.add_profiles(Namespace.CONDOR, key="should_transfer_files",
                      value="Yes")
    site.add_profiles(Namespace.CONDOR, key="when_to_transfer_output",
                      value="ON_EXIT_OR_EVICT")
    site.add_profiles(Namespace.CONDOR, key="My.DESIRED_Sites",
                      value='"nogrid"')
    site.add_profiles(Namespace.CONDOR, key="My.IS_GLIDEIN",
                      value='"False"')
    site.add_profiles(Namespace.CONDOR, key="My.flock_local",
                      value="True")
    site.add_profiles(Namespace.DAGMAN, key="retry", value="2")
    sitecat.add_sites(site)


def add_condorpool_shared_site(sitecat, cp, local_path, local_url):
    """Add condorpool_shared site to site catalog"""
    # local_url must end with a '/'
    if not local_url.endswith('/'):
        local_url = local_url + '/'

    site = Site("condorpool_shared", arch=Arch.X86_64, os_type=OS.LINUX)
    add_site_pegasus_profile(site, cp)

    # It's annoying that this is needed!
    local_dir = Directory(Directory.SHARED_SCRATCH,
                          path=os.path.join(local_path, 'cpool-site-scratch'))
    local_file_serv = FileServer(urljoin(local_url, 'cpool-site-scratch'),
                                 Operation.ALL)
    local_dir.add_file_servers(local_file_serv)
    site.add_directories(local_dir)

    site.add_profiles(Namespace.PEGASUS, key="style", value="condor")
    site.add_profiles(Namespace.PEGASUS, key="data.configuration",
                      value="sharedfs")
    site.add_profiles(Namespace.PEGASUS, key='transfer.bypass.input.staging',
                      value="true")
    site.add_profiles(Namespace.PEGASUS, key='auxillary.local',
                      value="true")
    site.add_profiles(Namespace.CONDOR, key="My.OpenScienceGrid",
                      value="False")
    site.add_profiles(Namespace.CONDOR, key="should_transfer_files",
                      value="Yes")
    site.add_profiles(Namespace.CONDOR, key="when_to_transfer_output",
                      value="ON_EXIT_OR_EVICT")
    site.add_profiles(Namespace.CONDOR, key="My.DESIRED_Sites",
                      value='"nogrid"')
    site.add_profiles(Namespace.CONDOR, key="My.IS_GLIDEIN",
                      value='"False"')
    site.add_profiles(Namespace.CONDOR, key="My.flock_local",
                      value="True")
    site.add_profiles(Namespace.DAGMAN, key="retry", value="2")
    # Need to set PEGASUS_HOME
    site.add_profiles(Namespace.ENV, key="PEGASUS_HOME",
                      value=get_pegasus_home())
    sitecat.add_sites(site)


def get_pegasus_home():
    """Locate the Pegasus installation prefix from the pegasus-plan command"""
    peg_home = which('pegasus-plan')
    if peg_home is None:
        raise RuntimeError(
            'pegasus-plan command not found. '
            'Make sure Pegasus is correctly installed.'
        )
    if not peg_home.endswith('bin/pegasus-plan'):
        raise RuntimeError(
            f'path to pegasus-plan is weird: {peg_home}. '
            'Make sure Pegasus is correctly installed.'
        )
    return peg_home.replace('bin/pegasus-plan', '')


def add_slurm_site(sitecat, cp, local_path, local_url):
    """Add a SLURM cluster site to the site catalog.

    Jobs on this site are submitted to the local SLURM batch system
    through HTCondor's grid universe and the blahp (the "glite" style
    in Pegasus). This requires an HTCondor install (which provides the
    blahp) on the workflow submit host, and assumes the submit host
    shares a filesystem with the SLURM compute nodes.

    The SLURM partition can be chosen with ``pycbc|partition`` and the
    accounting project with ``pycbc|account`` in the
    ``[pegasus_profile-slurm]`` section of the configuration file. Any
    further sbatch directives can be supplied through the
    ``pegasus|glite.arguments`` profile, and per-job resources through
    the ``pegasus|cores``, ``pegasus|memory`` and ``pegasus|runtime``
    profiles of each executable (the condor ``request_*`` profiles are
    not translated into sbatch directives).
    """
    # local_url must end with a '/'
    if not local_url.endswith('/'):
        local_url = local_url + '/'

    site = Site("slurm", arch=Arch.X86_64, os_type=OS.LINUX)
    add_site_pegasus_profile(site, cp)

    sec = 'pegasus_profile-slurm'

    # The compute nodes are assumed to see the same filesystem as the
    # submit host, so run in a shared scratch directory as for
    # condorpool_shared.
    site_dir = Directory(Directory.SHARED_SCRATCH,
                         path=os.path.join(local_path,
                                           'slurm-site-scratch'),
                         shared_file_system=True)
    site_file_serv = FileServer(urljoin(local_url, 'slurm-site-scratch'),
                                Operation.ALL)
    site_dir.add_file_servers(site_file_serv)
    site.add_directories(site_dir)

    # The contact string is not used for local batch submission via the
    # blahp, but the site catalog schema requires one.
    if cp.has_option(sec, 'pycbc|host'):
        contact = cp.get(sec, 'pycbc|host')
    else:
        contact = 'localhost'
    # Declare a gateway for every job type: even with auxillary.local
    # set, Pegasus maps some jobs (e.g. the remote worker-package
    # staging job) to this site, and the glite style refuses any job
    # whose type has no gateway to derive grid_resource from
    site.add_grids(
        *[Grid(Grid.BATCH, contact, Scheduler.SLURM, job_type=job_type)
          for job_type in (SupportedJobs.COMPUTE,
                           SupportedJobs.AUXILLARY,
                           SupportedJobs.TRANSFER,
                           SupportedJobs.REGISTER,
                           SupportedJobs.CLEANUP)]
    )

    site.add_profiles(Namespace.PEGASUS, key="style", value="glite")
    # Pegasus <= 5.1.2 does not derive grid_resource from the grid
    # gateways above, so set it explicitly on every job of this site
    site.add_profiles(Namespace.CONDOR, key="grid_resource",
                      value="batch slurm")
    site.add_profiles(Namespace.PEGASUS, key="data.configuration",
                      value="sharedfs")
    # Run transfer/create-dir/cleanup jobs on the submit host rather
    # than through the queue
    site.add_profiles(Namespace.PEGASUS, key='auxillary.local',
                      value="true")
    site.add_profiles(Namespace.DAGMAN, key="retry", value="2")

    # Convenience options for common sbatch settings. The queue profile
    # becomes 'batch_queue' (sbatch --partition) and the project profile
    # becomes 'batch_project' (sbatch --account) in the generated grid
    # universe submit file.
    if cp.has_option(sec, 'pycbc|partition'):
        site.add_profiles(Namespace.PEGASUS, key="queue",
                          value=cp.get(sec, 'pycbc|partition'))
    if cp.has_option(sec, 'pycbc|account'):
        site.add_profiles(Namespace.PEGASUS, key="project",
                          value=cp.get(sec, 'pycbc|account'))

    # The blahp-generated batch script does not source the user's
    # environment, so kickstart must be locatable through PEGASUS_HOME
    site.add_profiles(Namespace.ENV, key="PEGASUS_HOME",
                      value=get_pegasus_home())
    sitecat.add_sites(site)


# NOTE: We should now be able to add a nonfs site. I'll leave this for a
#       future patch/as demanded feature though. The setup would largely be
#       the same as the OSG site, except without the OSG specific things.

# def add_condorpool_nonfs_site(sitecat, cp):


def add_osg_site(sitecat, cp):
    """Add osg site to site catalog"""
    site = Site("osg", arch=Arch.X86_64, os_type=OS.LINUX)
    add_site_pegasus_profile(site, cp)
    site.add_profiles(Namespace.PEGASUS, key="style", value="condor")
    site.add_profiles(Namespace.PEGASUS, key="data.configuration",
                      value="condorio")
    site.add_profiles(Namespace.PEGASUS, key='transfer.bypass.input.staging',
                      value="true")
    site.add_profiles(Namespace.CONDOR, key="should_transfer_files",
                      value="Yes")
    site.add_profiles(Namespace.CONDOR, key="when_to_transfer_output",
                      value="ON_SUCCESS")
    site.add_profiles(Namespace.CONDOR, key="success_exit_code",
                      value="0")
    site.add_profiles(Namespace.CONDOR, key="My.OpenScienceGrid",
                      value="True")
    site.add_profiles(Namespace.CONDOR, key="ulog_execute_attrs",
                      value="GLIDEIN_Site")
    site.add_profiles(Namespace.CONDOR, key="My.InitializeModulesEnv",
                      value="False")
    site.add_profiles(Namespace.CONDOR, key="My.SingularityCleanEnv",
                      value="True")
    # These numbers below correspond to the codes in table B.2 here:
    # https://htcondor.readthedocs.io/en/24.0/codes-other-values/job-event-log-codes.html
    # Values recommended by a condor expert
    site.add_profiles(Namespace.CONDOR, key="My.DAGManNodesMask",
                      value=r"\"0,1,2,4,5,7,8,9,10,11,12,13,16,17,24,27,35,36,40\"")
    site.add_profiles(Namespace.CONDOR, key="Requirements",
                      value="(HAS_SINGULARITY =?= TRUE) && "
                            "(IS_GLIDEIN =?= True) && "
                            "(HAS_CVMFS_singularity_opensciencegrid_org =?= True)")
    cvmfs_loc = '"/cvmfs/singularity.opensciencegrid.org/pycbc/pycbc-el8:v'
    cvmfs_loc += sing_version + '"'
    site.add_profiles(Namespace.CONDOR, key="My.SingularityImage",
                      value=cvmfs_loc)
    # On OSG failure rate is high
    site.add_profiles(Namespace.DAGMAN, key="retry", value="4")
    site.add_profiles(Namespace.ENV, key="LAL_DATA_PATH",
                      value="/cvmfs/software.igwn.org/pycbc/lalsuite-extra/current/share/lalsimulation")
    # Add MKL location to LD_LIBRARY_PATH for OSG
    site.add_profiles(Namespace.ENV, key="LD_LIBRARY_PATH",
                      value="/usr/local/lib:/.singularity.d/libs")
    sitecat.add_sites(site)


def add_site(sitecat, sitename, cp, out_dir=None):
    """Add site sitename to site catalog"""
    # Allow local site scratch to be overriden for any site which uses it
    sec = 'pegasus_profile-{}'.format(sitename)
    opt = 'pycbc|site-scratch'
    if cp.has_option(sec, opt):
        out_dir = os.path.abspath(cp.get(sec, opt))
        if cp.has_option(sec, 'pycbc|unique-scratch'):
            scratchdir = tempfile.mkdtemp(prefix='pycbc-tmp_', dir=out_dir)
            os.chmod(scratchdir, 0o755)
            try:
                os.symlink(scratchdir, '{}-site-scratch'.format(sitename))
            except OSError:
                pass
            out_dir = scratchdir
    elif out_dir is None:
        out_dir = os.getcwd()
    local_url = urljoin('file://', pathname2url(out_dir))
    if sitename == 'local':
        add_local_site(sitecat, cp, out_dir, local_url)
    elif sitename == 'condorpool_symlink':
        add_condorpool_symlink_site(sitecat, cp)
    elif sitename == 'condorpool_copy':
        add_condorpool_copy_site(sitecat, cp)
    elif sitename == 'condorpool_shared':
        add_condorpool_shared_site(sitecat, cp, out_dir, local_url)
    elif sitename == 'osg':
        add_osg_site(sitecat, cp)
    elif sitename == 'slurm':
        add_slurm_site(sitecat, cp, out_dir, local_url)
    else:
        raise ValueError("Do not recognize site {}".format(sitename))


def make_catalog(cp, out_dir):
    """Make combined catalog of built-in known sites"""
    catalog = SiteCatalog()
    for site in KNOWN_SITES:
        add_site(catalog, site, cp, out_dir=out_dir)
    return catalog
