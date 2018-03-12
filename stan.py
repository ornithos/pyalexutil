"""
This is a thin wrapper around CmdStan for convenient access for Python use. Of course in
most instances PyStan would be preferred but there are currently some features that are not
exposed in Rstan or PyStan which are in CmdStan. In particular I want access to specifying
a dense matrix for the inverse mass matrix.
"""
import os
import numpy as np
import pandas as pd
import subprocess
import tempfile
import json
import hashlib
import copy
from multiprocessing.dummy import Pool
from pyalexutil.txt import print_banner

def dump_stan_data(dict_data, fname=None, precision=8):
    expcd = "{:." + "{:d}".format(precision) + "e}"

    if fname is not None:
        foldernm = os.path.dirname(fname)
        assert os.path.isdir(foldernm), "specified folder ({:s}) in fname foes not exist.".format(
            foldernm
        )

    for i, (k, v) in enumerate(dict_data.items()):
        if np.isscalar(v):
            tmptxt = "{:s} <- {:g}".format(k, v)
        elif isinstance(v, np.ndarray):
            if v.ndim == 1:
                tmptxt = '{:s} <- '.format(k)
            else:
                tmptxt = '{:s} <- structure(\n'.format(k)

            # R data format is column-wise rather than row-wise
            # (irrelevant for 1-dim array / vector)
            val = v.T.ravel()
            if issubclass(val.dtype.type, np.integer):
                tmptxt += "c({:s})".format(",".join("{:d}".format(x) for x in val))
            else:
                tmptxt +=  "c({:s})".format(",".join(expcd.format(x) for x in val))

            if v.ndim > 1:
                tmptxt += ",\n.Dim = c({:s}))".format(", ".join(str(x) for x in v.shape))

        else:
            raise NotImplementedError("Do not know how to deal with {:s}, {:s}".format(
                k, str(type(v))
            ))
        if i == 0:
            txt = tmptxt
        else:
            txt = '\n'.join([txt, tmptxt])

    if fname is None:
        return txt
    else:
        with open(fname, 'w') as f:
            f.writelines(txt)
        return None


def ordered_md5_hash(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    elif isinstance(x, dict):
        x = copy.deepcopy(x)
        for k, v in x.items():
            if isinstance(v, np.ndarray):
                x[k] = v.tolist()

    return hashlib.md5(json.dumps(x).encode('utf-8')).hexdigest()

class CmdStanInterface(object):

    def __init__(self, cmd_stan_directory):
        assert os.path.isdir(cmd_stan_directory), "invalid directory specified"
        dir_contents = os.listdir('/Users/alexbird/utils/cmdstan-2.17.1')
        assert not set(['stan', 'make', 'makefile', 'src', 'runCmdStanTests.py']) - set(dir_contents), \
            "This does not look like the root directory of CmdStan."
        assert os.name != 'nt', "This class has not been built for Windows."

        if not 'bin' in dir_contents or set(['stanc', 'stansummary']) - \
                set(os.listdir(os.path.join(cmd_stan_directory, "bin"))):
            raise UserError("Looks like you haven't called make on CmdStan yet. Please do this first." + \
                  "(see Reference Manual Section 2.2).")
        self.cmdstandir = os.path.abspath(cmd_stan_directory)
        self._model_exec = None

    def set_stan_object_manual(self, filename):
        """
        For attaching a pre-compiled Stan object to the class (no need
        to call `compile` if using this interface. Please ensure that the
        relevant filename is a Stan object, and is compiled for the current
        architecture.
        :param filename:
        :return: None
        """
        assert os.path.isfile(filename), "file not found"
        with open(os.devnull, 'w') as devnull:
            op = subprocess.call([filename,'sample', 'help'], stdout=devnull)
        assert op == 0, "specified file does not appear to be Stan Object. Try recompiling."
        self._model_exec = os.path.abspath(filename)

    def compile(self, stanfile, verbose=True):
        assert os.path.splitext(stanfile)[1] == ".stan", "stan files must end in extension .stan"
        assert os.path.isfile(stanfile), "file not found"
        stanfile = os.path.abspath(stanfile)
        if verbose:
            print_banner("COMPILING STAN OBJECT")
            subprocess.call(''.join(['make ', stanfile]), cwd=self.cmdstandir, shell=True)
        else:
            with open(os.devnull, 'w') as devnull:
                subprocess.call(''.join(['make ', stanfile]), cwd=self.cmdstandir, stdout=devnull, shell=True)
        self._model_exec = stanfile[:-5]

    def sample(self, data_dict, num_chains, num_samples, num_warmup, save_warmup, adapt_engaged,
               algorithm="NUTS", metric='diag_e', inv_mass_matrix=None, stepsize=1,
               init="random", diagnostic=None, refresh=100, size_pool=2):

        if True: # code folding
            assert isinstance(data_dict, dict)
            assert isinstance(num_chains, int) and num_chains > 0
            assert isinstance(num_samples, int) and num_samples > 0
            assert isinstance(num_warmup, int) and num_warmup > 0
            assert isinstance(save_warmup, bool)
            assert isinstance(adapt_engaged, bool)
            assert algorithm.lower() in ['hmc', 'nuts'], "Algorithm should be 'HMC' or 'NUTS'"
            assert metric.lower() in ['unit_e', 'diag_e', 'dense_e']
            assert inv_mass_matrix is None or isinstance(inv_mass_matrix, np.ndarray), \
                "inv_mass_matrix must be None or numpy array."
            assert np.isscalar(stepsize), "stepsize should be scalar numeric"
            assert (isinstance(init, str) and init.lower() == "random") or \
                isinstance(init, dict), "init should be 'random' or dict of params (note only single" + \
                " dict is currently supported, i.e. same initialisation)."
            assert diagnostic is None or os.path.isdir(os.path.dirname(diagnostic)), "diagnostic file " + \
                "directory does not exist."
            assert isinstance(refresh, int) and refresh > 0

        # Save 3 different temporary files: data_dict, init and inv_mass_matrix
        dir_tmp = tempfile.gettempdir()

        data_md5 = ordered_md5_hash(data_dict)[-16:]
        data_name = os.path.join(dir_tmp, 'cmdstandata_{:s}'.format(data_md5))
        if not os.path.isfile(data_name):
            dump_stan_data(data_dict, fname=data_name)

        if inv_mass_matrix is not None:
            mass_md5 = hashlib.md5(inv_mass_matrix).hexdigest()[-16:]
            mass_name = os.path.join(dir_tmp, 'cmdstanmass_{:s}'.format(mass_md5))
            if not os.path.isfile(mass_name):
                dump_stan_data(dict(inv_metric=inv_mass_matrix), fname=mass_name)
        else:
            mass_name is None

        if isinstance(init, dict):
            init_md5 = hashlib.md5(init).hexdigest()[-16:]
            init_name = os.path.join(dir_tmp, 'cmdinitmass_{:s}'.format(init_md5))
            if not os.path.isfile(init_name):
                dump_stan_data(init, fname=init_name)
        else:
            init_name = None

        # Deal with other things: initialisation and output file
        # if init.lower() == "random":
        #     init = "2"  # this is much worse than what rstan/pystan do :S

        op_file_id = [tempfile.NamedTemporaryFile(delete=False) for x in range(num_chains)]
        output_files = []
        for fid in op_file_id:
            output_files.append(fid.name)
            fid.close()

        print('rm ' + ' '.join(output_files))


        # Generate command string for running the sampler in CmdStan
        def get_cmd_str_for_thread(j):
            cmds = [self._model_exec, "sample",
                    "num_samples={:d}".format(num_samples),
                    "num_warmup={:d}".format(num_warmup),
                    "save_warmup={:d}".format(save_warmup),
                    "adapt",
                    "engaged={:d}".format(adapt_engaged),
                    "algorithm=hmc",
                    "engine={:s}".format(["static", "nuts"][algorithm.lower()=="nuts"]),
                    "metric={:s}".format(metric)]
            if inv_mass_matrix is not None:
                cmds.append("metric_file={:s}".format(mass_name))
            cmds += ["stepsize={:.8f}".format(stepsize),
                     "id={:d}".format(j+1),
                     "data",
                     "file={:s}".format(data_name)]
            if isinstance(init, dict):
                cmds.append("init={:s}".format(init_name))
            cmds +=["output",
                     "file={:s}".format(output_files[j])]
            if diagnostic is not None:
                cmds.appen("diagnostic_file={:s}".format(diagnostic))
            cmds.append("refresh={:d}".format(refresh))
            return cmds

        # Run the sampler (in multithread pool - with a little help from https://stackoverflow.com/a/14533902)
        pool = Pool(size_pool)
        cmds = [get_cmd_str_for_thread(x) for x in range(num_chains)]
        for i, returncode in enumerate(pool.imap(subprocess.call, cmds)):
            if returncode != 0:
                print("chain %d failed: %d" % (i, returncode))


        # Collect results from various files
        out = [pd.read_csv(f, comment='#') for f in output_files]
        out = pd.concat(out)

        # Call stansummary()
        with tempfile.NamedTemporaryFile(mode='a') as f:
            subprocess.call(['grep', 'lp__', output_files[0]], stdout=f)
            for i in range(num_chains):
                subprocess.call(['sed', '/^[#l]/d', output_files[i]], stdout=f)
            summ = subprocess.check_output([os.path.join(self.cmdstandir, 'bin', 'stansummary'), f.name])

        # Delete output files
        for i in range(num_chains):
            op_file_id[i].close()
            os.remove(output_files[i])

        return out, summ.decode('utf-8')