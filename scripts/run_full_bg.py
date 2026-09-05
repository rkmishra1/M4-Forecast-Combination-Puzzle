"""Detached runner: registers SIGUSR1 stack-dump, then runs the full study."""
import faulthandler, signal, runpy, sys
faulthandler.register(signal.SIGUSR1, all_threads=True)
sys.argv = ["run.py", "--config", "configs/full.yaml"]
runpy.run_path("scripts/run.py", run_name="__main__")
