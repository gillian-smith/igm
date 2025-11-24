import tensorflow as tf
import json
import numpy as np
import sys
from tqdm import tqdm
import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
import io
import re


from .visualizers import _plot_memory_pie, _plot_computational_pie


def print_comp(state):
    ################################################################

    size_of_tensor = {}

    for m in state.__dict__.keys():
        try:
            size_gb = sys.getsizeof(getattr(state, m).numpy())
            if size_gb > 1024**1:
                size_of_tensor[m] = size_gb / (1024**3)
        except:
            pass

    # sort from highest to lowest
    size_of_tensor = dict(
        sorted(size_of_tensor.items(), key=lambda item: item[1], reverse=True)
    )

    print("Memory statistics report:")
    with open("memory-statistics.txt", "w") as f:
        for key, value in size_of_tensor.items():
            print("     %24s  |  size : %8.4f Gb " % (key, value), file=f)
            print("     %24s  |  size : %8.4f Gb  " % (key, value))

    _plot_memory_pie(state)

    ################################################################

    modules = list(state.tcomp.keys())

    print("Computational statistics report:")
    with open("computational-statistics.txt", "w") as f:
        for m in modules:
            CELA = (m, np.mean(state.tcomp[m]), np.sum(state.tcomp[m]))
            print(
                "     %14s  |  mean time per it : %8.4f  |  total : %8.4f" % CELA,
                file=f,
            )
            print("     %14s  |  mean time per it : %8.4f  |  total : %8.4f" % CELA)

    _plot_computational_pie(state)


def print_gpu_info() -> None:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(f"{'CUDA Enviroment':-^150}")
    tf.sysconfig.get_build_info().pop("cuda_compute_capabilities", None)
    print(f"{json.dumps(tf.sysconfig.get_build_info(), indent=2, default=str)}")
    print(f"{'Available GPU Devices':-^150}")
    for gpu in gpus:
        gpu_info = {"gpu_id": gpu.name, "device_type": gpu.device_type}
        device_details = tf.config.experimental.get_device_details(gpu)
        gpu_info.update(device_details)

        print(f"{json.dumps(gpu_info, indent=2, default=str)}")
    print(f"{'':-^150}")


def print_info(state):

    if state.it % 100 == 1:
        if hasattr(state, "pbar"):
            state.pbar.close()
        state.pbar = tqdm(
            desc=f"IGM", ascii=False, dynamic_ncols=True, bar_format="{desc} {postfix}"
        )

    if hasattr(state, "pbar"):
        dic_postfix = {
            "🕒": datetime.datetime.now().strftime("%H:%M:%S"),
            "🔄": f"{state.it:06.0f}",
            "⏱ Time": f"{state.t.numpy():09.1f} yr",
            "⏳ Step": f"{state.dt:04.2f} yr",
        }
        if hasattr(state, "dx"):
            dic_postfix["❄️  Volume"] = (
                f"{np.sum(state.thk) * (state.dx**2) / 10**9:108.2f} km³"
            )
        if hasattr(state, "particle"):
            dic_postfix["# Particles"] = str(state.particle["x"].shape[0])

        #        dic_postfix["💾 GPU Mem (MB)"] = tf.config.experimental.get_memory_info("GPU:0")['current'] / 1024**2

        state.pbar.set_postfix(dic_postfix)
        state.pbar.update(1)


def print_model_with_inputs(
    model,
    input_names,
    normalization_method="standardization",
    title="Model Architecture",
):
    """
    Print TensorFlow model summary alongside input variable information.

    Args:
        model: TensorFlow/Keras model
        input_names: List of variable names corresponding to channels
        normalization_method: String describing the normalization (e.g., "standardization", "min-max [0,1]", "channel-wise scaling")
        title: Title for the display
    """
    console = Console()

    # ===== MODEL TABLE =====
    string_buffer = io.StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + "\n"))
    summary_str = string_buffer.getvalue()
    lines = summary_str.split("\n")

    model_table = Table(
        title=f"[bold cyan]{model.name}[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
        title_style="bold cyan",
        expand=False,
    )

    model_table.add_column("Layer (type)", style="cyan", no_wrap=True, width=20)
    model_table.add_column("Output Shape", style="green", width=22)
    model_table.add_column("Params", style="yellow", justify="right", width=10)

    # Parse model layers
    in_layers = False
    for line in lines:
        if line.strip().startswith("=") or line.strip().startswith("_"):
            continue
        if "Layer (type)" in line:
            in_layers = True
            continue
        if "Total params:" in line:
            in_layers = False

        if in_layers and line.strip():
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 3:
                layer_name = parts[0]
                output_shape = parts[1].replace("None", "[bold red]None[/bold red]")
                param_count = parts[2]

                if param_count != "0":
                    param_count = f"[bold yellow]{param_count}[/bold yellow]"
                else:
                    param_count = f"[dim]{param_count}[/dim]"

                model_table.add_row(layer_name, output_shape, param_count)

    # Get parameter summary
    total_params = trainable_params = None
    for line in lines:
        if "Total params:" in line:
            total_params = line.split("Total params:")[1].strip()
        elif "Trainable params:" in line:
            trainable_params = line.split("Trainable params:")[1].strip()

    model_table.caption = f"[bold white]Total:[/bold white] [cyan]{total_params}[/cyan] | [bold white]Trainable:[/bold white] [green]{trainable_params}[/green]"

    # ===== INPUT VARIABLES TABLE =====
    input_table = Table(
        title="[bold cyan]Input Variables[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="green",
        title_style="bold cyan",
        expand=False,
    )

    input_table.add_column("#", style="dim", justify="right", width=4)
    input_table.add_column("Variable", style="cyan bold", width=18)

    # Add each variable
    for i, var_name in enumerate(input_names):
        input_table.add_row(str(i), var_name)

    # Add normalization info as caption
    input_table.caption = (
        f"[bold white]Channels:[/bold white] [cyan]{len(input_names)}[/cyan] | "
        f"[bold white]Normalization:[/bold white] [yellow]{normalization_method}[/yellow]"
    )

    # ===== DISPLAY SIDE BY SIDE =====
    console.print()
    console.print(
        Panel(
            Columns([model_table, input_table], equal=False, expand=True),
            title=f"[bold]{title}[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()
