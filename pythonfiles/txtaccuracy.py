import re

# Codex mapping
codex = {
    0: "Fixing",
    1: "Driving",
    2: "Falling",
    3: "Walking",
    4: "Crouching",
    5: "Jumping",
    6: "Pack_Help",
    7: "POV",
    8: "Digging",
    9: "Carrying",
    10: "Picking_Up",
    11: "Raking",
    12: "Waving",
    13: "Pointing",
    14: "Dropping"
}


# Reverse the codex for easy lookup
reverse_codex = {v: k for k, v in codex.items()}

# Read the terminal output from a log file
with open('python_files/txt/apollomixed6.txt', 'r') as file:
    input_text = file.read()

# Extract labels
labels = re.findall(r'Labels: \[(.*?)\]', input_text)
labels = [int(label) for label_list in labels for label in label_list.split(',')]

# Extract predictions
predictions = re.findall(r'Final prediction for video #\d+: (\w+)', input_text)

# Check if the extraction was successful
if len(labels) != len(predictions):
    print(f"Error: The number of labels ({len(labels)}) does not match the number of predictions ({len(predictions)}).")
    print(f"Labels: {labels}")
    print(f"Predictions: {predictions}")
else:
    # Initialize counters
    correct_predictions = 0
    total_predictions = len(labels)

    # Initialize class-specific counters
    class_correct = {class_name: 0 for class_name in codex.values()}
    class_total = {class_name: 0 for class_name in codex.values()}

    # Compare labels and predictions
    for i in range(len(labels)):
        label = labels[i]
        expected_class = codex[label]
        predicted_class = predictions[i]
        
        # Count overall accuracy
        if predicted_class == expected_class:
            correct_predictions += 1
        
        # Count class-specific accuracy
        class_total[expected_class] += 1
        if predicted_class == expected_class:
            class_correct[expected_class] += 1

    # Calculate overall accuracy
    overall_accuracy = (correct_predictions / total_predictions) * 100

    # Print overall accuracy
    print(f"Overall Accuracy: {overall_accuracy:.2f}%\n")

    # Calculate and print accuracy for each class
    for class_name in codex.values():
        if class_total[class_name] > 0:
            class_accuracy = (class_correct[class_name] / class_total[class_name]) * 100
            print(f"Accuracy for {class_name}: {class_accuracy:.2f}%")
        else:
            print(f"No instances for class {class_name}")
