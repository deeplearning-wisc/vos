import os


def top_dir():
    """
    returns project top most directory
    :return: (str) Project directory
    """
    return os.sep.join(
        os.path.dirname(
            os.path.realpath(__file__)).split(
            os.sep)[
                :-2])


def data_dir():
    """
    Returns data directory. Data directory should never be in src, especially when using IDEs.
    :return:(str) data directory
    """
    return top_dir() + '/detection/data'


def configs_dir():
    """
    Returns configs directory
    :return: (str) Configs directory
    """
    return os.sep.join([top_dir(), 'detection', 'configs'])
