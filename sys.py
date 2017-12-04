import sys
import os
import re
import warnings
import time


def isunix(allowUnknown=False):
    test = sys.platform
    if re.search('linux.*', test, re.IGNORECASE):
        return True
    elif re.search('darwin', test, re.IGNORECASE):
        return True
    elif re.search('win.*', test, re.IGNORECASE):
        return False
    elif re.search('cygwin.*', test, re.IGNORECASE):
        return False
    else:
        if ~allowUnknown:
            warnings.warn('Unable to identify platform of runtime. Assuming unix..', UserWarning)
            return True
        else:
            return '?'
    return True


def filesep():
    running_unix = isunix(allowUnknown=True)
    if type(running_unix) is bool:
        if running_unix:
            return '/'
        else:
            return '\\'
    else:
        warnings.warn('Unknown platform: assuming unix based system, using forwardslash for directories')
        return '/'


def fullfile(*args):
    fs = filesep()
    fswrong = '/'
    if fs == '/':
        fswrong = '\\'
    out = ''
    for str in args:
        if str[-1] == fs:
            out += str
            continue
        elif str[-1] == fswrong:
            str = str[:-1]

        str += fs
        out += str
    return os.path.expanduser(out[:-1])


def time_now(format="%H:%M:%S"):
    return time.strftime(format, time.localtime())


def text_from_file(filename, filepath=""):
    with open(os.path.join(filepath, filename)) as f:
        txt = f.read()
    return txt


def read_single_keypress():
    """Waits for a single keypress on stdin.

    This is a silly function to call if you need to do it a lot because it has
    to store stdin's current setup, setup stdin for reading single keystrokes
    then read the single keystroke then revert stdin back after reading the
    keystroke.

    Returns the character of the key that was pressed (zero on
    KeyboardInterrupt which can happen when a signal gets handled)

    """
    import termios, fcntl, sys, os
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save) # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK
                  | termios.ISTRIP | termios.INLCR | termios. IGNCR
                  | termios.ICRNL | termios.IXON )
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios. PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON
                  | termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1) # returns a single character
    except KeyboardInterrupt:
        ret = 0
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret
