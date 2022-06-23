from time import sleep

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
progress = Progress(
    TextColumn("[bold blue]运行中...", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
)

with progress:
    for n in progress.track(range(100)):
        print(n)
        sleep(0.5)