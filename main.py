import tkinter as tk
from tkinter import scrolledtext

import predictor
import visualizations

def onEnterKey(event):
    onSubmitButtonClicked()


def onSubmitButtonClicked():
    inputText = inputEntry.get()
    statusLabel.config(text="Determining if the comment is toxic...")
    window.update()
    prediction = predictor.predict(inputText)
    statusLabel.config(text="Ready.")
    outputText.insert(tk.END, f"{prediction}\n")
    outputText.see(tk.END)


def onBarChartButtonClicked():
    visualizations.createBarChart()


def onShowPieChartButtonClicked():
    visualizations.createPieChart()


def onShowHeatmapButtonClicked():
    statusLabel.config(text="Heatmap is being generated. Please be patient...")
    window.update()
    visualizations.createHeatMap()


# Create the main window
window = tk.Tk()
window.title("Toxic Comment Classifier")
window.geometry("800x600")

# Create the text input box and submit button
inputLabel = tk.Label(window, text="Enter Comment: ")
inputEntry = tk.Entry(window)
inputEntry.bind("<Return>", onEnterKey)
submitButton = tk.Button(window, text="Submit", command=onSubmitButtonClicked)

# Create the status label
statusLabel = tk.Label(window, text="Ready", anchor="w")

# Create the text output area
outputText = scrolledtext.ScrolledText(window, width=90, height=20)

# Create the buttons
barChartButton = tk.Button(window, text="Show Bar Chart", command=onBarChartButtonClicked)
pieChartButton = tk.Button(window, text="Show Pie Chart", command=onShowPieChartButtonClicked)
heatmapButton = tk.Button(window, text="Show Heatmap", command=onShowHeatmapButtonClicked)

# Configure the input label, input entry, and submit button to span the width of the window
inputLabel.pack(side="top", anchor="w", padx=10, pady=10)
inputEntry.pack(side="top", fill="x", padx=10, pady=10)
submitButton.pack(side="top", anchor="e", padx=10, pady=(0, 10))

# Position the text output area
outputText.pack(side="top", fill="both", padx=10, pady=10)

# Position the buttons
buttonFrame = tk.Frame(window)
buttonFrame.pack(side="bottom", pady=10)
barChartButton.pack(side="left", padx=(10, 5))
pieChartButton.pack(side="left", padx=5)
heatmapButton.pack(side="left", padx=(5, 10))

# Position the status label
statusLabel.pack(side="left", fill="x", padx=10, pady=(0, 10))

# Start the main event loop
window.mainloop()
