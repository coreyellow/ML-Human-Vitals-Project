import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

save_dir = 'Dataset_visuals'

if __name__ == "__main__":
    df_male = pd.read_csv('Datasets/male_dataset.csv')
    df_female = pd.read_csv('Datasets/female_dataset.csv')
    
    features = df_male.columns.tolist()
    print(df_male[features].describe())
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        # Bins for discrete features        
        if feature in ['Oxygen Saturation', 'Respiratory Rate', 'Heart Rate', 'Age']:
            low = int(min(df_male[feature].min(), df_female[feature].min()))
            high = int(max(df_male[feature].max(), df_female[feature].max()))
            bins = range(low, high + 1)
        else:
            bins = 'auto'
        
        axes[i].hist(
            [df_male[feature], df_female[feature]],
            bins=bins,
            stacked=True,
            label=['Male', 'Female']
        )
        
        axes[i].set_title(f'Histogram of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_histograms.png')
    plt.show()
    
    
    
    