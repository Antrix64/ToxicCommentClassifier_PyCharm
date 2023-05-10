import matplotlib.pyplot as plot
import numpy as np
import dataHandler
import predictor


# Shows a barchart of the data distribution by number.
def createBarChart():
    # Count the number of comments for each toxicity category.
    data = dataHandler.loadCSVFromZipFile('toxic_subset_10901.zip', 'toxic_subset_10901.csv')
    dataColumns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    counts = [sum(data[column] == 1) for column in dataColumns]
    categories = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate', 'Non-toxic']

    # Total the number of comments that do not have toxic traits
    toxicCommentCount = 0
    for index, row in data.iterrows():
        if any(row[column] == 1 for column in dataColumns):
            toxicCommentCount += 1

    counts.append(len(data) - toxicCommentCount)

    # Set up the bar chart.
    plot.figure()
    x = np.arange(len(categories))
    plot.bar(x, counts)
    plot.xlabel('Toxicity Category')
    plot.ylabel('Count')
    plot.title('Toxic Comment Distribution by Number\nNote: Some comments fall into multiple categories.')

    # Adjust the subplots so all data is visible
    plot.subplots_adjust(bottom=0.3)

    # Rotate the x-axis labels for improved readability.
    plot.xticks(x, categories, rotation=45)

    plot.show()


# Shows a pie chart of the data distribution by percentages.
def createPieChart():
    # Define the toxic categories and their corresponding percentages
    data = dataHandler.loadCSVFromZipFile('toxic_subset_10901.zip', 'toxic_subset_10901.csv')
    dataColumns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # categories = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
    categories = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate', 'Non-toxic']
    counts = [sum(data[column] == 1) for column in dataColumns]
    totalData = len(data)

    # Total the number of comments that do not have toxic traits
    toxicCommentCount = 0
    for index, row in data.iterrows():
        if any(row[column] == 1 for column in dataColumns):
            toxicCommentCount += 1

    # Percentages of toxic and non-toxic comments
    percentToxic = toxicCommentCount / totalData
    percentNonToxic = 1 - percentToxic

    # Calculate the percentages for toxic comment distribution
    percentages = [((i / sum(counts)) * percentToxic) for i in counts]
    percentages.append(percentNonToxic)

    # Create the pie chart
    plot.figure()
    plot.pie(percentages, labels=None)
    plot.title('Toxic Comment Distribution by Percentage')

    # Format the text for the legend
    labels = [f'{category} ({(percentage * 100):.2f}%)' for category, percentage in zip(categories, percentages)]

    # Adjust the legend, so it is not on top of the pie chart.
    plot.legend(labels, loc='lower left', bbox_to_anchor=(0.86, 0.0))
    plot.tight_layout()

    plot.show()


# Create a heatmap of the classification report data produced by sklearn metrics.
def createHeatMap():
    report_data = predictor.getClassificationReport()
    # Get precision, recall, and F1-score data for each category
    categories = list(report_data.keys())
    metrics = ['precision', 'recall', 'f1-score']
    data = np.zeros((len(categories), len(metrics)))

    for i, category in enumerate(categories):
        for j, metric in enumerate(metrics):
            data[i, j] = report_data[category][metric]

    # Create a heatmap using Matplotlib
    figure, axis = plot.subplots()
    heatmap = axis.imshow(data, cmap='coolwarm', vmin=0, vmax=1)

    # Set the ticks on the value bar
    axis.set_xticks(np.arange(len(metrics)))
    axis.set_yticks(np.arange(len(categories)))

    # Set the correct labels
    labelY = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate', 'Micro Average', 'Macro Average',
              'Weighted Average', 'Samples Average']
    labelX = ['Precision', 'Recall', 'F1-Score']
    axis.set_xticklabels(labelX)
    axis.set_yticklabels(labelY)

    # Rotate the tick labels and set alignment for better readability
    plot.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plot.colorbar(heatmap)
    cbar.set_label('Value')

    # Add values to the heatmap
    for i in range(len(categories)):
        for j in range(len(metrics)):
            text = axis.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black')

    plot.title('Classification Report Heatmap')
    plot.xlabel('Metrics')
    plot.ylabel('Categories')
    plot.tight_layout()
    plot.subplots_adjust(bottom=0.18, top=0.94, right=0.75)

    plot.show()
