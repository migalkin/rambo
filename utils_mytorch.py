from typing import List, Dict
from pathlib import Path
import warnings
import os

class ImproperCMDArguments(Exception): pass
class BadParameters(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "Unexpected value of parameter {0}".format(dErrorArguments))
        self.dErrorArguments = dErrorArguments


class FancyDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self

# def compute_mask(t: Union[torch.Tensor, np.array], padding_idx=0):
#     """
#     compute mask on given tensor t
#     :param t: either a tensor or a nparry
#     :param padding_idx: the ID used to represented padded data
#     :return: a mask of the same shape as t
#     """
#     if type(t) is np.ndarray:
#         mask = np.not_equal(t, padding_idx)*1.0
#     else:
#         mask = torch.ne(t, padding_idx).float()
#     return mask

# Transparent, and simple argument parsing FTW!
def convert_nicely(arg, possible_types=(bool, float, int, str)):
    """ Try and see what sticks. Possible types can be changed. """
    for data_type in possible_types:
        try:

            if data_type is bool:
                # Hard code this shit
                if arg in ['T', 'True', 'true']: return True
                if arg in ['F', 'False', 'false']: return False
                raise ValueError
            else:
                proper_arg = data_type(arg)
                return proper_arg
        except ValueError:
            continue
    # Here, i.e. no data type really stuck
    warnings.warn(f"None of the possible datatypes matched for {arg}. Returning as-is")
    return arg


def parse_args(raw_args: List[str], compulsory: List[str] = (), compulsory_msg: str = "",
               types: Dict[str, type] = None, discard_unspecified: bool = False):
    """
        I don't like argparse.
        Don't like specifying a complex two liner for each every config flag/macro.

        If you maintain a dict of default arguments, and want to just overwrite it based on command args,
        call this function, specify some stuff like

    :param raw_args: unparsed sys.argv[1:]
    :param compulsory: if some flags must be there
    :param compulsory_msg: what if some compulsory flags weren't there
    :param types: a dict of confignm: type(configvl)
    :param discard_unspecified: flag so that if something doesn't appear in config it is not returned.
    :return:
    """

    # parsed_args = _parse_args_(raw_args, compulsory=compulsory, compulsory_msg=compulsory_msg)
    #
    # # Change the "type" of arg, anyway

    parsed = {}

    while True:

        try:                                        # Get next value
            nm = raw_args.pop(0)
        except IndexError:                          # We emptied the list
            break

        # Get value
        try:
            vl = raw_args.pop(0)
        except IndexError:
            raise ImproperCMDArguments(f"A value was expected for {nm} parameter. Not found.")

        # Get type of value
        if types:
            try:
                parsed[nm] = types[nm](vl)
            except ValueError:
                raise ImproperCMDArguments(f"The value for {nm}: {vl} can not take the type {types[nm]}! ")
            except KeyError:                    # This name was not included in the types dict
                if not discard_unspecified:     # Add it nonetheless
                    parsed[nm] = convert_nicely(vl)
                else:                           # Discard it.
                    continue
        else:
            parsed[nm] = convert_nicely(vl)

    # Check if all the compulsory things are in here.
    for key in compulsory:
        try:
            assert key in parsed
        except AssertionError:
            raise ImproperCMDArguments(compulsory_msg + f"Found keys include {[k for k in parsed.keys()]}")

    # Finally check if something unwanted persists here
    return parsed


def mt_save_dir(parentdir: Path, _newdir: bool = False):
    """
            Function which returns the last filled/or newest unfilled folder in a particular dict.
            Eg.1
                parentdir is empty dir
                    -> mkdir 0
                    -> cd 0
                    -> return parentdir/0

            Eg.2
                ls savedir -> 0, 1, 2, ... 9, 10, 11
                    -> mkdir 12 && cd 12 (if newdir is True) else 11
                    -> return parentdir/11 (or 12)

            ** Usage **
            Get a path using this function like so:
                parentdir = Path('runs')
                savedir = save_dir(parentdir, _newdir=True)

        :param parentdir: pathlib.Path object of the parent directory
        :param _newdir: bool flag to save in the last dir or make a new one
        :return: None
    """

    # Check if the dir exits
    try:
        assert parentdir.exists(), f'{parentdir} does not exist. Making it'
    except AssertionError:
        parentdir.mkdir(parents=False)

    # List all folders within, and convert them to ints
    existing = sorted([int(x) for x in os.listdir(parentdir) if x.isdigit()], reverse=True)

    if not existing:
        # If no subfolder exists
        parentdir = parentdir / '0'
        parentdir.mkdir()
    elif _newdir:
        # If there are subfolders and we want to make a new dir
        parentdir = parentdir / str(existing[0] + 1)
        parentdir.mkdir()
    else:
        # There are other folders and we dont wanna make a new folder
        parentdir = parentdir / str(existing[0])

    return parentdir