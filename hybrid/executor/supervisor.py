"""PID 1 of every executor job container.

Runs the job command as a child, reaps every orphan, and turns the container
exit code into something the host can classify without trusting job output:

* the child's own exit code (128+N when killed by signal N), or
* PIDS_LIMIT_EXIT when the container cgroup's pids.events counted any hit on
  the pids limit, whatever the child then did, or
* SUPERVISOR_ERROR_EXIT when the supervisor itself could not do its work.

Child exit codes equal to either reserved value are rewritten to 1.
A job can kill this process (same uid), which only changes its own
classification from pids to candidate_error.
"""
import os
import signal
import sys

PIDS_LIMIT_EXIT = 254
SUPERVISOR_ERROR_EXIT = 253
PIDS_EVENTS = "/sys/fs/cgroup/pids.events"


def pids_limit_hits():
    with open(PIDS_EVENTS) as handle:
        for line in handle:
            key, value = line.split()
            if key == "max":
                return int(value)
    raise RuntimeError("pids.events has no 'max' counter")


def run(argv):
    child = os.fork()
    if child == 0:
        try:
            os.execvp(argv[0], argv)
        except OSError as error:
            sys.stderr.write("supervisor: cannot exec %r: %s\n" % (argv[0], error))
            os._exit(127)
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda number, _frame: os.kill(child, number))
    while True:
        try:
            pid, status = os.wait()
        except InterruptedError:
            continue
        if pid == child:
            break
    code = os.waitstatus_to_exitcode(status)
    code = 128 - code if code < 0 else code
    if pids_limit_hits() > 0:
        return PIDS_LIMIT_EXIT
    if code in (PIDS_LIMIT_EXIT, SUPERVISOR_ERROR_EXIT):
        return 1
    return code


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("supervisor: no command\n")
        return SUPERVISOR_ERROR_EXIT
    try:
        return run(sys.argv[1:])
    except Exception as error:  # reported to the host as a distinct exit code
        sys.stderr.write("supervisor error: %r\n" % (error,))
        return SUPERVISOR_ERROR_EXIT


if __name__ == "__main__":
    sys.exit(main())
