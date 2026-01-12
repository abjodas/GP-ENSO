from pathlib import Path

PLOT_DIR = None 

def set_plot_dir(path: Path):
    global PLOT_DIR
    PLOT_DIR = Path(path)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
