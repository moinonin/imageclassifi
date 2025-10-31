from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
import pandas as pd
import numpy as np

def print_colorful_results(results):
    # Create DataFrame
    df = pd.DataFrame(results)
    df["is_ai"] = df["filename"].str.contains(r"_ai(?=\.[^.]+$)", regex=True)
    df[["outcome", "confidence", "features"]] = pd.DataFrame(df["result"].tolist(), index=df.index)
    df["detected"] = np.where(df["outcome"] == True, "AI-Generated", "Human-Created")
    df.drop(columns=["result", "outcome"], inplace=True)
    df = df[["filename", "is_ai", "detected", "confidence", "features"]]
    
    # Create Rich table
    console = Console()
    table = Table(
        title="🎨 AI Detection Results Summary",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        title_style="bold cyan"
    )
    
    # Add columns
    table.add_column("📁 Filename", style="cyan", no_wrap=True)
    table.add_column("🤖 Ground-Truth", style="yellow", justify="center")
    table.add_column("🔍 Detected", justify="center")
    table.add_column("📊 Confidence", justify="center")
    table.add_column("🎯 Features", style="green")
    
    # Define emojis for outstanding features
    feature_emojis = {
        'eigen_entropy': '📈',
        'eigen_decay_rate': '📉', 
        'eigen_condition_number': '🎛️',
        'radial_smoothness': '🌀',
        'high_freq_energy': '⚡',
        'channel_correlation': '🔗',
        'color_consistency': '🎨',
        'local_inconsistency': '🔍',
        'noise_std': '📏',
        'noise_skew': '📐',
        'noise_regularity': '🔄'
    }
    
    # Add rows
    for _, row in df.iterrows():
        filename = row['filename']
        is_ai = "✅" if row['is_ai'] else "❌"
        
        # Color detection result
        if row['detected'] == "AI-Generated":
            detected = Text("🤖 AI-Generated", style="bold red")
        else:
            detected = Text("👤 Human-Created", style="bold green")
        
        # Color confidence
        confidence = row['confidence']
        if confidence == 'high':
            confidence_text = Text(f"🔥 {confidence}", style="bold red")
        elif confidence == 'medium':
            confidence_text = Text(f"⚠️ {confidence}", style="bold yellow")
        else:
            confidence_text = Text(f"💧 {confidence}", style="bold blue")
        
        # Format features beautifully
        features = row['features']
        feature_text = Text()
        for i, (key, value) in enumerate(features.items()):
            emoji = feature_emojis.get(key, '📊')
            
            # Color code values based on their characteristics
            if 'entropy' in key or 'energy' in key:
                if value > 0.5:
                    value_style = "bright_red"
                else:
                    value_style = "bright_green"
            elif 'smoothness' in key or 'consistency' in key:
                if value > 0.5:
                    value_style = "bright_green" 
                else:
                    value_style = "bright_red"
            elif 'correlation' in key:
                if value > 0.9:
                    value_style = "bright_cyan"
                else:
                    value_style = "bright_yellow"
            else:
                value_style = "white"
            
            # Format the feature line
            if i > 0:
                feature_text.append("\n")
            feature_text.append(f"{emoji} ", style="bold white")
            feature_text.append(f"{key}: ", style="bright_white")
            feature_text.append(f"{value:.4f}", style=value_style)
        
        table.add_row(filename, is_ai, detected, confidence_text, feature_text)
    
    # Print the table
    console.print()
    console.print(table)
    
    # Print summary statistics
    console.print("\n[bold turquoise]📊 Summary Statistics:[/bold turquoise]")
    correct_detections = ((df['is_ai'] == True) & (df['detected'] == "AI-Generated")) | \
                        ((df['is_ai'] == False) & (df['detected'] == "Human-Created"))
    accuracy = correct_detections.mean()
    
    console.print(f"🎯 Accuracy: [bold green]{accuracy:.1%}[/bold green]")
    console.print(f"📁 Total Files: [bold cyan]{len(df)}[/bold cyan]")
    console.print(f"🤖 AI Detected: [bold red]{(df['detected'] == 'AI-Generated').sum()}[/bold red]")
    console.print(f"👤 Human Detected: [bold green]{(df['detected'] == 'Human-Created').sum()}[/bold green]")

# Usage - replace your current print code with:
# print_colorful_results(results)