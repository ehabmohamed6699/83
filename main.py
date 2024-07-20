import tkinter as tk
from tkinter import ttk
import BackProbagation as bp
import pandas as pd
import helper


def submit_click():
    data = pd.read_csv("./Dry_Bean_Dataset.csv")

    data["MinorAxisLength"] = data["MinorAxisLength"].fillna(data["MinorAxisLength"].mean())
    norm_data = data.drop(columns=['Class'])
    norm_data = (norm_data - norm_data.mean()) / norm_data.std()
    norm_data["Class"] = data["Class"]
    data = norm_data
    train_data = pd.concat([data.iloc[0:30], data.iloc[50:80], data.iloc[100:130]])
    test_data = pd.concat([data.iloc[30:50], data.iloc[80:100], data.iloc[130:150]])

    inputs = None
    test = None
    activate = None
    inv_activate = None
    num_hidden_layers = entry_hidden_layers.get()
    neurons_per_layer = entry_neurons_per_layer.get()
    learning_rate = float(entry_learning_rate.get())
    activation_function = combo_var.get()
    bias = chk_var.get()

    input_size = 5
    hidden_layers = []
    for i in range(int(num_hidden_layers)):
        try:
            hidden_layers.append(int(neurons_per_layer.split(",")[i]))
        except:
            break
    output_size = 3
    epochs = int(entry_num_epochs.get())
    threshold = 0.001
    sigmoid = activation_function == "Sigmoid"
    print(train_data.columns)
    if sigmoid:
        inputs = [[[x[0], x[1], x[2], x[3], x[4]],
                [1, 0, 0] if x[5] == "BOMBAY" else [0, 1, 0] if x[5] == "CALI" else [0, 0, 1]] for x in
                train_data.values]
        test = [[[x[0], x[1], x[2], x[3], x[4]],
                [1, 0, 0] if x[5] == "BOMBAY" else [0, 1, 0] if x[5] == "CALI" else [0, 0, 1]] for x in
                test_data.values]
        activate = helper.sigmoid
        inv_activate = helper.dervative_sigmoid
    else:
        inputs = [[[x[0], x[1], x[2], x[3], x[4]],
                [1, -1, -1] if x[5] == "BOMBAY" else [-1, 1, -1] if x[5] == "CALI" else [-1, -1, 1]] for x in
                train_data.values]
        test = [[[x[0], x[1], x[2], x[3], x[4]],
                [1, -1, -1] if x[5] == "BOMBAY" else [-1, 1, -1] if x[5] == "CALI" else [-1, -1, 1]] for x in
                test_data.values]
        activate = helper.tanh
        inv_activate = helper.dervative_tanh

    model = bp.MLPClassifier(input_size, hidden_layers, output_size, activate, inv_activate, bias)
    model.learn(inputs,epochs,threshold,learning_rate)
    print("##################################################")
    print("#################TRAIN EVALUATION#################")
    print("##################################################")
    model.evaluate(inputs)
    print("##################################################")
    print("#################TEST EVALUATION##################")
    print("##################################################")
    model.evaluate(test)


root = tk.Tk()
root.title("Back Propagation")


labels = ["Number of hidden layers:",
          "Number of neurons in each hidden layer (please enter them comma separated):",
          "Learning rate:",
          "Number of epochs:"]

entry_vars = [tk.StringVar() for _ in range(4)]
entry_hidden_layers = tk.Entry(root, textvariable=entry_vars[0])
entry_neurons_per_layer = tk.Entry(root, textvariable=entry_vars[1])
entry_learning_rate = tk.Entry(root, textvariable=entry_vars[2])
entry_num_epochs = tk.Entry(root, textvariable=entry_vars[3])

for i, label_text in enumerate(labels):
    label = tk.Label(root, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=5)

entry_hidden_layers.grid(row=0, column=1, padx=10, pady=5)
entry_neurons_per_layer.grid(row=1, column=1, padx=10, pady=5)
entry_learning_rate.grid(row=2, column=1, padx=10, pady=5)
entry_num_epochs.grid(row=3, column=1, padx=10, pady=5)

# Create ComboBox
activation_functions = ["Sigmoid", "Hyperbolic Tangent Sigmoid"]
combo_var = tk.StringVar()
combo_box = ttk.Combobox(root, textvariable=combo_var, values=activation_functions, state="readonly")
combo_box.set("Activation function")
combo_box.grid(row=4, column=0, columnspan=2, padx=10, pady=5)

# Create checkbox
chk_var = tk.BooleanVar()
chk_box = tk.Checkbutton(root, text="Add bias", variable=chk_var)
chk_box.grid(row=5, column=0, columnspan=2, padx=10, pady=5)

# Create submit button
submit_btn = tk.Button(root, text="Submit", command=submit_click)
submit_btn.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

# Start the Window
root.config(bg="#ADD8E6")
root.title
root.mainloop()