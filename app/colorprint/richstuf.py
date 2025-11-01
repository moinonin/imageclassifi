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
    table.add_column("🤖 Ground Truth", style="yellow", justify="center")
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
    
    # Calculate confusion matrix
    true_positive = ((df['is_ai'] == True) & (df['detected'] == "AI-Generated")).sum()
    false_positive = ((df['is_ai'] == False) & (df['detected'] == "AI-Generated")).sum()
    true_negative = ((df['is_ai'] == False) & (df['detected'] == "Human-Created")).sum()
    false_negative = ((df['is_ai'] == True) & (df['detected'] == "Human-Created")).sum()
    
    total = len(df)
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print Confusion Matrix
    console.print("\n[bold turquoise]🎯 Confusion Matrix:[/bold turquoise]")
    
    cm_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    cm_table.add_column("", style="bold cyan", justify="center")
    cm_table.add_column("Predicted: AI", style="bold red", justify="center")
    cm_table.add_column("Predicted: Human", style="bold green", justify="center")
    
    # Add rows to confusion matrix
    cm_table.add_row(
        "Actual: AI",
        f"[bold green]✓ TP: {true_positive}[/bold green]",
        f"[bold red]✗ FN: {false_negative}[/bold red]"
    )
    cm_table.add_row(
        "Actual: Human", 
        f"[bold red]✗ FP: {false_positive}[/bold red]",
        f"[bold green]✓ TN: {true_negative}[/bold green]"
    )
    
    console.print(cm_table)
    
    # Print performance metrics
    console.print("\n[bold turquoise]📊 Performance Metrics:[/bold turquoise]")
    
    metrics_table = Table(show_header=False, box=box.SIMPLE, show_edge=False)
    metrics_table.add_column("Metric", style="bold cyan")
    metrics_table.add_column("Value", style="bold white")
    
    metrics_table.add_row("🎯 Accuracy", f"[bold green]{accuracy:.1%}[/bold green]")
    metrics_table.add_row("🎯 Precision", f"[bold yellow]{precision:.1%}[/bold yellow]")
    metrics_table.add_row("🎯 Recall", f"[bold yellow]{recall:.1%}[/bold yellow]")
    metrics_table.add_row("🎯 F1-Score", f"[bold cyan]{f1_score:.1%}[/bold cyan]")
    
    console.print(metrics_table)
    
    # Print summary statistics
    console.print("\n[bold turquoise]📈 Summary Statistics:[/bold turquoise]")
    
    stats_table = Table(show_header=False, box=box.SIMPLE, show_edge=False)
    stats_table.add_column("Statistic", style="bold cyan")
    stats_table.add_column("Value", style="bold white")
    
    stats_table.add_row("📁 Total Files", f"[bold white]{total}[/bold white]")
    stats_table.add_row("🤖 Actual AI Files", f"[bold red]{(df['is_ai'] == True).sum()}[/bold red]")
    stats_table.add_row("👤 Actual Human Files", f"[bold green]{(df['is_ai'] == False).sum()}[/bold green]")
    stats_table.add_row("🔍 Detected AI", f"[bold red]{(df['detected'] == 'AI-Generated').sum()}[/bold red]")
    stats_table.add_row("🔍 Detected Human", f"[bold green]{(df['detected'] == 'Human-Created').sum()}[/bold green]")
    
    console.print(stats_table)
