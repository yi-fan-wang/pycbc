# Copyright (C) 2026 The PyCBC development team
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
These are the unittests for the pycbc.workflow.pegasus_sites module
"""
import os
import tempfile
import unittest
from unittest import mock

import yaml
from Pegasus.api import SiteCatalog

from pycbc.workflow.configuration import WorkflowConfigParser
from pycbc.workflow import pegasus_sites

from utils import simple_exit


def make_config(extra=None):
    """Return a WorkflowConfigParser with optional extra sections"""
    cp = WorkflowConfigParser()
    if extra is not None:
        for sec, opts in extra.items():
            cp.add_section(sec)
            for opt, val in opts.items():
                cp.set(sec, opt, val)
    return cp


def catalog_to_dict(catalog):
    """Write a SiteCatalog to YAML and read it back as a dictionary"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'sites.yml')
        catalog.write(path)
        with open(path) as fp:
            return yaml.safe_load(fp)


def get_site(catdict, name):
    """Pull the site entry called name out of a site catalog dictionary"""
    for site in catdict['sites']:
        if site['name'] == name:
            return site
    raise KeyError(name)


FAKE_PEGASUS_PLAN = '/opt/pegasus/bin/pegasus-plan'


class PegasusSitesTestClass(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        patcher = mock.patch(
            'pycbc.workflow.pegasus_sites.which',
            return_value=FAKE_PEGASUS_PLAN
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_make_catalog_contains_known_sites(self):
        cp = make_config()
        catdict = catalog_to_dict(
            pegasus_sites.make_catalog(cp, self.tmpdir)
        )
        names = [site['name'] for site in catdict['sites']]
        for site in pegasus_sites.KNOWN_SITES:
            self.assertIn(site, names)

    def test_slurm_site_defaults(self):
        cp = make_config()
        catalog = SiteCatalog()
        pegasus_sites.add_site(catalog, 'slurm', cp, out_dir=self.tmpdir)
        site = get_site(catalog_to_dict(catalog), 'slurm')

        # Jobs must be submitted through the batch/glite route with a
        # shared filesystem: condorio does not work with glite style
        profiles = site['profiles']
        self.assertEqual(profiles['pegasus']['style'], 'glite')
        self.assertEqual(profiles['pegasus']['data.configuration'],
                         'sharedfs')
        self.assertEqual(profiles['pegasus']['auxillary.local'], 'true')
        self.assertEqual(profiles['env']['PEGASUS_HOME'], '/opt/pegasus/')

        # The grid gateway is what makes the planner generate
        # 'grid_resource = batch slurm' submit files
        grid_types = [(grid['type'], grid['scheduler'])
                      for grid in site['grids']]
        self.assertIn(('batch', 'slurm'), grid_types)

        # A shared scratch directory must be defined for sharedfs mode
        dir_types = [d['type'] for d in site['directories']]
        self.assertIn('sharedScratch', dir_types)

    def test_slurm_site_partition_and_account(self):
        cp = make_config(extra={
            'pegasus_profile-slurm': {
                'pycbc|partition': 'somenodes',
                'pycbc|account': 'someproject',
            },
        })
        catalog = SiteCatalog()
        pegasus_sites.add_site(catalog, 'slurm', cp, out_dir=self.tmpdir)
        site = get_site(catalog_to_dict(catalog), 'slurm')

        # queue becomes sbatch --partition, project becomes --account
        profiles = site['profiles']
        self.assertEqual(profiles['pegasus']['queue'], 'somenodes')
        self.assertEqual(profiles['pegasus']['project'], 'someproject')

    def test_slurm_site_profile_passthrough(self):
        cp = make_config(extra={
            'pegasus_profile-slurm': {
                'pegasus|glite.arguments': '--constraint=avx2 --exclusive',
                'env|LAL_DATA_PATH': '/shared/lal-data',
            },
        })
        catalog = SiteCatalog()
        pegasus_sites.add_site(catalog, 'slurm', cp, out_dir=self.tmpdir)
        site = get_site(catalog_to_dict(catalog), 'slurm')

        profiles = site['profiles']
        self.assertEqual(profiles['pegasus']['glite.arguments'],
                         '--constraint=avx2 --exclusive')
        self.assertEqual(profiles['env']['LAL_DATA_PATH'],
                         '/shared/lal-data')


suite = unittest.TestSuite()
suite.addTest(
    unittest.TestLoader().loadTestsFromTestCase(PegasusSitesTestClass)
)

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
