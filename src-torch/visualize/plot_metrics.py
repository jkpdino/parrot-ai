import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

class MetricsVisualizer:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.metrics = self.load_metrics()
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
    
    def load_metrics(self) -> Dict[str, Any]:
        """Load metrics from JSON file"""
        metrics_path = self.run_dir / 'metrics.json'
        if not metrics_path.exists():
            raise FileNotFoundError(f"No metrics file found at {metrics_path}")
            
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def plot_losses(self, save: bool = True):
        """Plot training and evaluation losses"""
        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])
        
        # Training loss
        steps = self.metrics['steps']
        train_losses = self.metrics['train_losses']
        eval_steps = steps[::len(train_losses)//len(self.metrics['eval_losses'])]
        
        # Main loss plot
        ax1.plot(steps, train_losses, label='Training Loss', alpha=0.6)
        ax1.plot(eval_steps, self.metrics['eval_losses'], 
                label='Evaluation Loss', linewidth=2)
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Steps')
        ax1.legend()
        ax1.set_title('Training Progress')
        
        # Learning rate plot
        ax2.plot(steps, self.metrics['learning_rates'], 
                label='Learning Rate', color='green')
        ax2.set_ylabel('Learning Rate')
        ax2.set_xlabel('Steps')
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.run_dir / 'loss_plot.png')
        else:
            plt.show()
        plt.close()
    
    def plot_time_metrics(self, save: bool = True):
        """Plot metrics over wall clock time"""
        timestamps = np.array(self.metrics['timestamps'])
        start_time = timestamps[0]
        hours_elapsed = (timestamps - start_time) / 3600
        
        plt.figure()
        plt.plot(hours_elapsed, self.metrics['train_losses'], 
                label='Training Loss', alpha=0.6)
        
        # Calculate evaluation points in hours
        eval_indices = np.linspace(0, len(hours_elapsed)-1, 
                                 len(self.metrics['eval_losses'])).astype(int)
        plt.plot(hours_elapsed[eval_indices], self.metrics['eval_losses'],
                label='Evaluation Loss', linewidth=2)
        
        plt.xlabel('Hours Elapsed')
        plt.ylabel('Loss')
        plt.title('Training Progress Over Time')
        plt.legend()
        
        if save:
            plt.savefig(self.run_dir / 'time_plot.png')
        else:
            plt.show()
        plt.close()
    
    def generate_training_summary(self) -> str:
        """Generate a text summary of the training run"""
        total_steps = len(self.metrics['steps'])
        best_eval = self.metrics['best_eval_loss']
        final_train = self.metrics['train_losses'][-1]
        
        timestamps = np.array(self.metrics['timestamps'])
        total_hours = (timestamps[-1] - timestamps[0]) / 3600
        
        summary = [
            "Training Summary",
            "================",
            f"Total Steps: {total_steps}",
            f"Best Evaluation Loss: {best_eval:.4f}",
            f"Final Training Loss: {final_train:.4f}",
            f"Total Training Time: {total_hours:.2f} hours",
            f"Average Steps/Hour: {total_steps/total_hours:.1f}"
        ]
        
        return "\n".join(summary)
    
    def save_summary(self):
        """Save training summary to file"""
        summary = self.generate_training_summary()
        with open(self.run_dir / 'training_summary.txt', 'w') as f:
            f.write(summary)

def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('run_dir', type=str, help='Path to training run directory')
    parser.add_argument('--no-save', action='store_true', 
                       help='Display plots instead of saving')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    visualizer = MetricsVisualizer(run_dir)
    visualizer.plot_losses(save=not args.no_save)
    visualizer.plot_time_metrics(save=not args.no_save)
    visualizer.save_summary()

if __name__ == "__main__":
    main()