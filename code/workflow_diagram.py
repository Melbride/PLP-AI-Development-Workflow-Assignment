"""
AI Development Workflow Diagram Generator
Creates a visual representation of the AI development process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_workflow_diagram():
    """
    Create a visual workflow diagram for AI development
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define workflow stages
    stages = [
        ("Problem Definition", 0.1, 0.9),
        ("Data Collection", 0.1, 0.75),
        ("Data Preprocessing", 0.1, 0.6),
        ("Model Development", 0.1, 0.45),
        ("Model Evaluation", 0.1, 0.3),
        ("Deployment", 0.1, 0.15),
        ("Monitoring", 0.5, 0.9),
        ("Stakeholder Analysis", 0.5, 0.75),
        ("Feature Engineering", 0.5, 0.6),
        ("Hyperparameter Tuning", 0.5, 0.45),
        ("Bias Assessment", 0.5, 0.3),
        ("Integration Testing", 0.5, 0.15),
        ("Success Metrics", 0.9, 0.9),
        ("Ethical Review", 0.9, 0.75),
        ("Model Selection", 0.9, 0.6),
        ("Performance Testing", 0.9, 0.45),
        ("Continuous Improvement", 0.9, 0.3),
        ("Production Deployment", 0.9, 0.15)
    ]
    
    # Color scheme for different phases
    colors = {
        'planning': '#E8F4FD',
        'data': '#B8E6B8', 
        'modeling': '#FFE4B5',
        'evaluation': '#FFB6C1',
        'deployment': '#DDA0DD',
        'monitoring': '#F0E68C'
    }
    
    # Stage categories
    stage_colors = [
        colors['planning'], colors['data'], colors['data'], 
        colors['modeling'], colors['evaluation'], colors['deployment'],
        colors['monitoring'], colors['planning'], colors['data'],
        colors['modeling'], colors['evaluation'], colors['deployment'],
        colors['planning'], colors['planning'], colors['modeling'],
        colors['evaluation'], colors['monitoring'], colors['deployment']
    ]
    
    # Draw boxes for each stage
    boxes = []
    for i, (stage, x, y) in enumerate(stages):
        # Create fancy box
        box = FancyBboxPatch(
            (x-0.08, y-0.04), 0.16, 0.08,
            boxstyle="round,pad=0.01",
            facecolor=stage_colors[i],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(box)
        boxes.append((x, y))
        
        # Add text
        ax.text(x, y, stage, ha='center', va='center', fontsize=8, weight='bold')
    
    # Draw arrows to show workflow
    arrow_props = dict(arrowstyle='->', lw=1.5, color='darkblue')
    
    # Vertical arrows (main flow)
    for i in range(5):
        ax.annotate('', xy=(0.1, 0.75-i*0.15), xytext=(0.1, 0.9-i*0.15), arrowprops=arrow_props)
    
    # Horizontal arrows (parallel processes)
    for i in range(6):
        y_pos = 0.9 - i*0.15
        ax.annotate('', xy=(0.5, y_pos), xytext=(0.18, y_pos), arrowprops=arrow_props)
        ax.annotate('', xy=(0.9, y_pos), xytext=(0.58, y_pos), arrowprops=arrow_props)
    
    # Add title and labels
    ax.set_title('AI Development Workflow', fontsize=16, weight='bold', pad=20)
    
    # Add phase labels
    ax.text(0.1, 0.98, 'Core Workflow', ha='center', fontsize=12, weight='bold', color='darkblue')
    ax.text(0.5, 0.98, 'Parallel Activities', ha='center', fontsize=12, weight='bold', color='darkgreen')
    ax.text(0.9, 0.98, 'Quality Assurance', ha='center', fontsize=12, weight='bold', color='darkred')
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['planning'], label='Planning'),
        patches.Patch(color=colors['data'], label='Data Processing'),
        patches.Patch(color=colors['modeling'], label='Modeling'),
        patches.Patch(color=colors['evaluation'], label='Evaluation'),
        patches.Patch(color=colors['deployment'], label='Deployment'),
        patches.Patch(color=colors['monitoring'], label='Monitoring')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.1))
    
    # Set axis properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('C:/AI_Development_Workflow_Assignment/docs/workflow_diagram.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main execution function
    """
    print("Generating AI Development Workflow Diagram...")
    create_workflow_diagram()
    print("Diagram saved as 'workflow_diagram.png'")

if __name__ == "__main__":
    main()