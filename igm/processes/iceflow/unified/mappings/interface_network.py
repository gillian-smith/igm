#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import warnings
from omegaconf import DictConfig
from typing import Any, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import io
import re

import igm
from igm.common.core import State
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
    load_model_from_path,
)
from .interface import InterfaceMapping
from igm.processes.iceflow.emulate.utils.networks import StandardizationLayer, FixedAffineLayer, NormalizationLayer
from .utils import process_inputs_scales, process_inputs_variances

class InterfaceNetwork(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:

        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_physics = cfg.processes.iceflow.physics
        cfg_unified = cfg.processes.iceflow.unified

        if cfg_unified.pretrained:
            dir_path = get_pretrained_emulator_path(cfg, state)
            iceflow_model = load_model_from_path(dir_path, cfg_unified.inputs)
        else:
            warnings.warn("No pretrained emulator found. Starting from scratch.")

            nb_inputs = len(cfg_unified.inputs) + (cfg_physics.dim_arrhenius == 3) * (
                cfg_numerics.Nz - 1
            )

            nb_outputs = 2 * cfg_numerics.Nz
            
            # TODO: Work on refactoring as this is currently quite messy with the names and hierarchy
            if cfg_unified.scaling.method.lower() == "automatic_standardization":
                norm = StandardizationLayer()
            elif cfg_unified.scaling.method.lower() == "automatic_normalization":
                norm = NormalizationLayer()
            elif cfg_unified.scaling.method.lower() == "manual_standardization":
                scales = process_inputs_scales(
                    cfg_unified.scaling.manual.inputs_scales, cfg_unified.inputs
                )
                variances = process_inputs_variances(
                    cfg_unified.scaling.manual.inputs_variances, cfg_unified.inputs
                )
                norm = FixedAffineLayer(
                        scales=scales,
                        variances=variances
                    )
            else:
                raise ValueError(
                    f"Unknown scaling method: {cfg_unified.scaling.method}. "
                    f"Available methods: automatic_standardization, automatic_normalization, manual_standardization"
                )

            architecture_name = cfg_unified.network.architecture

            # Get the function from the networks module
            if hasattr(igm.processes.iceflow.emulate.utils.networks, architecture_name):
                architecture_class = getattr(
                    igm.processes.iceflow.emulate.utils.networks, architecture_name
                )

                iceflow_model = architecture_class(
                    cfg, nb_inputs, nb_outputs, input_normalizer=norm
                )

            else:
                raise ValueError(
                    f"Unknown network architecture: {architecture_name}. "
                    f"Available architectures: cnn, unet"
                )

        state.iceflow_model = iceflow_model
        state.iceflow_model.compile(jit_compile=False) # not all architectures support jit_compile=True
        
        if cfg.processes.iceflow.unified.network.print_summary:
            print_model_summary(state.iceflow_model) # TODO: make it into a table potentially...

        return {
            "bcs": cfg_unified.bcs,
            "vertical_discr": state.iceflow.vertical_discr,
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": cfg_unified.network.output_scale,
            "precision": cfg_numerics.precision,
        }

# ! This should be moved to a RICH display folder as we are accumulating quite a few of them...
def print_model_summary(model, title="Model Architecture"):
    """
    Print a TensorFlow model summary as a styled Rich table.
    
    Args:
        model: TensorFlow/Keras model
        title: Title for the panel
    """
    console = Console()
    
    # Capture the model summary as a string
    string_buffer = io.StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
    summary_str = string_buffer.getvalue()
    
    # Parse the summary
    lines = summary_str.split('\n')
    
    # Create the table
    table = Table(
        title=f"[bold cyan]{model.name}[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
        title_style="bold cyan"
    )
    
    # Add columns
    table.add_column("Layer (type)", style="cyan", no_wrap=True)
    table.add_column("Output Shape", style="green")
    table.add_column("Param #", style="yellow", justify="right")
    table.add_column("Connected to", style="dim white")
    
    # Parse and add rows
    in_layers = False
    for line in lines:
        # Skip separator lines
        if line.strip().startswith('=') or line.strip().startswith('_'):
            continue
            
        # Check if we're in the layer section
        if 'Layer (type)' in line:
            in_layers = True
            continue
            
        # Stop at the totals section
        if 'Total params:' in line or 'Trainable params:' in line or 'Non-trainable params:' in line:
            in_layers = False
            
        # Parse layer rows
        if in_layers and line.strip():
            # Split by multiple spaces
            parts = re.split(r'\s{2,}', line.strip())
            
            if len(parts) >= 3:
                layer_name = parts[0]
                output_shape = parts[1]
                param_count = parts[2]
                connected_to = parts[3] if len(parts) > 3 else ""
                
                # Format output shape to highlight None
                output_shape = output_shape.replace('None', '[bold red]None[/bold red]')
                
                # Format parameter count
                if param_count != '0':
                    param_count = f"[bold yellow]{param_count}[/bold yellow]"
                else:
                    param_count = f"[dim]{param_count}[/dim]"
                
                # Clean up connected_to
                connected_to = connected_to.replace('[', '').replace(']', '').replace("'", '')
                if len(connected_to) > 50:
                    connected_to = connected_to[:47] + "..."
                
                table.add_row(layer_name, output_shape, param_count, connected_to)
    
    # Create summary stats
    total_params = None
    trainable_params = None
    non_trainable_params = None
    
    for line in lines:
        if 'Total params:' in line:
            total_params = line.split('Total params:')[1].strip()
        elif 'Trainable params:' in line:
            trainable_params = line.split('Trainable params:')[1].strip()
        elif 'Non-trainable params:' in line:
            non_trainable_params = line.split('Non-trainable params:')[1].strip()
    
    # Create summary panel
    summary_text = Text()
    summary_text.append("Total params: ", style="bold white")
    summary_text.append(f"{total_params}\n", style="bold cyan")
    summary_text.append("Trainable params: ", style="bold white")
    summary_text.append(f"{trainable_params}\n", style="bold green")
    summary_text.append("Non-trainable params: ", style="bold white")
    summary_text.append(f"{non_trainable_params}", style="bold red")
    
    summary_panel = Panel(
        summary_text,
        title="[bold magenta]Parameter Summary[/bold magenta]",
        border_style="magenta",
        padding=(1, 2)
    )
    
    # Print everything
    console.print()
    console.print(Panel(table, title=f"[bold]{title}[/bold]", border_style="cyan", padding=(1, 2)))
    console.print(summary_panel)
    console.print()