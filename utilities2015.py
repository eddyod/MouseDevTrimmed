import matplotlib
matplotlib.use('Agg')

import os
import sys
from subprocess import check_output, call
import numpy as np
import pandas
import configparser

def load_hdf(fn, key='data'):
    return pandas.read_hdf(fn, key)


def load_ini(fp, split_newline=True, convert_none_str=True, section='DEFAULT'):
    """
    Value of string None will be converted to Python None.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(fp):
        raise Exception("ini file %s does not exist." % fp)
    config.read(fp)
    input_spec = dict(config.items(section))
    input_spec = {k: v.split('\n') if '\n' in v else v for k, v in input_spec.items()}
    for k, v in input_spec.items():
        if not isinstance(v, list):
            if '.' not in v and v.isdigit():
                    input_spec[k] = int(v)
            elif v.replace('.','',1).isdigit():
                input_spec[k] = float(v)
            elif v == 'None':
                if convert_none_str:
                        input_spec[k] = None
    assert len(input_spec) > 0, "Failed to read data from ini file."
    return input_spec


def shell_escape(s):
    """
    Escape a string (treat it as a single complete string) in shell commands.
    """
    from tempfile import mkstemp
    fd, path = mkstemp()
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(s)
        cmd = r"""cat %s | sed -e "s/'/'\\\\''/g; 1s/^/'/; \$s/\$/'/" """ % path
        escaped_str = check_output(cmd, shell=True)
    finally:
        os.remove(path)

    return escaped_str

def one_liner_to_arr(line, func):
    return np.array(map(func, line.strip().split()))

def create_if_not_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            sys.stderr.write('%s\n' % e);

    return path

def execute_command(cmd, stdout=None, stderr=None):
    sys.stderr.write(cmd + '\n')

    # try:
#     from errand_boy.transports.unixsocket import UNIXSocketTransport
#     errand_boy_transport = UNIXSocketTransport()
#     stdout, stderr, retcode = errand_boy_transport.run_cmd(cmd)

#     print stdout
#     print stderr

    # import os
    # retcode = os.system(cmd)
    retcode = call(cmd, shell=True, stdout=stdout, stderr=stderr)
    sys.stderr.write('return code: %d\n' % retcode)

    # if retcode < 0:
    #     print >>sys.stderr, "Child was terminated by signal", -retcode
    # else:
    #     print >>sys.stderr, "Child returned", retcode
    # except OSError as e:
    #     print >>sys.stderr, "Execution failed:", e
    #     raise e
