import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Build a node class as a data type for the data in the tree
class node:
    parent = None   # Parent node
    left_child = None   # left child node
    right_child = None  # right child node
    threshold = 0   # A best break point to minimize the cost
    start = 0   # The start index of the input examples
    end = 0     # The end index of the input examples
    predictor = [0, 0]  # Predictor of left and right part
    cost = 0    # hinge cost of the node

    # Set the value of start and end
    def boundary_setting(self, start, end):
        self.start = start
        self.end = end

    # Set the value of threshold, predictor and cost
    def value_setting(self, threshold, predictor, cost):
        self.threshold = threshold
        self.predictor = predictor
        self.cost = cost

    # Set the parent node
    def parent_node_setting(self, parent):
        self.parent = parent

    # Set the child node
    def child_node_setting(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child


# This function read a csv file and return is with data.frame format
def csv_reader(path, flag):
    if flag == 'x':
        str_x = path + "features.csv"
        data = pd.read_csv(str_x).values.astype(np.float64)
    elif flag == 'y':
        str_y = path + "targets.csv"
        data = pd.read_csv(str_y).values.astype(np.float64)
    else:
        str_fold = path + "folds.csv"
        data = pd.read_csv(str_fold).values.astype(np.float64)
    return data


# The function of this part is find the hinge loss and find the best mean value to minimize the loss
def optimal_finding(y_lower, y_upper, sigma):
    max_lower = y_lower[0]
    min_upper = y_upper[0]
    for lower in y_lower:
        if not lower == np.inf:
            if lower > max_lower:
                max_lower = lower
    for upper in y_upper:
        if not upper == -np.inf:
            if upper < min_upper:
                min_upper = upper
    # Set mean as the mean of the max lower boundary and min upper boundary
    mean = (max_lower + min_upper)/2
    cost = 0
    # Calaulate the hinge loss point by point
    for y1 in y_lower:
        if not y1 == np.inf and not y1 == -np.inf:
            b1 = y1 + sigma
            if mean < b1:
                cost += np.abs(b1 - mean)
    for y2 in y_upper:
        if not y2 == np.inf and not y2 == -np.inf:
            b2 = y2 - sigma
            if mean > b2:
                cost += np.abs(b2 + mean)
    return [mean, cost]


# This function will find the best threshold which can minimize the total hinge loss of left and right part
def split(y_lower, y_upper, sigma):
    min_cost = np.inf   # Min hinge loss totally
    best_left_predictor = 0     # The predictor of left part
    best_right_predictor = 0    # The predictor of right part
    threshold = 0
    index = 0
    left_upper = []
    right_upper = []
    right_lower = []
    left_lower = []
    # If there is only one group input data, we think the cost is zero
    if len(y_lower) <= 1:
        min_cost = 0
    # Divide the input data into two sets, calculate the hinge loss of these two parts
    else:
        while index < len(y_lower)-1:
            # Divide the input data according to the index
            looper = 0
            while looper < len(y_lower):
                if looper <= index:
                    left_upper.append(y_upper[looper])
                    left_lower.append(y_lower[looper])
                else:
                    right_upper.append(y_upper[looper])
                    right_lower.append(y_lower[looper])
                looper += 1
            # Calculate total_cost and predictors
            left_cost = optimal_finding(left_lower, left_upper, sigma)[1]
            right_cost = optimal_finding(right_lower, right_upper, sigma)[1]
            left_predictor = optimal_finding(left_lower, left_upper, sigma)[0]
            right_predictor = optimal_finding(right_lower, right_upper, sigma)[0]
            total_cost = left_cost + right_cost
            # Update optimal if lower total cost is found.
            if total_cost < min_cost:
                min_cost = total_cost
                best_left_predictor = left_predictor
                best_right_predictor = right_predictor
                threshold = index+1
            # Clean the input data list for next loop round
            left_upper.clear()
            left_lower.clear()
            right_upper.clear()
            right_lower.clear()
            index += 1
    # Put the information of optimal into a node
    new_node = node()
    if min_cost == 0:
        best_right_predictor = optimal_finding(y_lower, y_upper, sigma)[0]
        best_left_predictor = optimal_finding(y_lower, y_upper, sigma)[0]
    new_node.value_setting(threshold, [best_left_predictor, best_right_predictor], min_cost)
    return new_node


def MMIT(y_lower, y_upper, sigma):
    # Initialize the tree and insert the root node
    node_list = []
    work_list = []
    root_node = split(y_lower, y_upper, sigma)
    root_node.boundary_setting(0, len(y_lower))
    work_list.append(root_node)
    node_list.append(root_node)
    index = 0
    # Keep developing the tree until there is no nodes have child node.
    while not len(work_list) == 0:
        # If the cost of the node is zero, we do need generate new child node
        if work_list[0].cost == 0:
            del work_list[0]
        # If the cost of the node is not zero, build two child node
        else:
            left_node = node()
            right_node = node()
            # Set the relationship between nodes
            left_node.parent_node_setting(node_list[index])
            right_node.parent_node_setting(node_list[index])
            node_list[index].child_node_setting(left_node, right_node)
            # Set examples range of nodes
            right_node.boundary_setting(right_node.parent.threshold, right_node.parent.end)
            left_node.boundary_setting(left_node.parent.start, left_node.parent.threshold)
            # Set the data that nodes will use for running
            left_examples_lower = y_lower[left_node.parent.start: left_node.parent.threshold]
            left_examples_upper = y_upper[left_node.parent.start: left_node.parent.threshold]
            right_examples_lower = y_lower[left_node.parent.threshold: left_node.parent.end]
            right_examples_upper = y_upper[left_node.parent.threshold: left_node.parent.end]
            # Run split function to find the optimal mean and calculate cost
            temp_left = split(left_examples_lower, left_examples_upper, sigma)
            temp_right = split(right_examples_lower, right_examples_upper, sigma)
            left_node.value_setting(temp_left.threshold + left_node.parent.start, temp_left.predictor, temp_left.cost)
            right_node.value_setting(temp_right.threshold + right_node.parent.threshold, temp_right.predictor, temp_right.cost)
            # Add two new nodes in work list and node list
            work_list.append(left_node)
            work_list.append(right_node)
            node_list.append(left_node)
            node_list.append(right_node)
            # Remove the parent node from work list
            del work_list[0]
        index += 1
    return node_list


# This function is used for predicting the value of y using MMIT
def predictor(node_list, X_examples):
    index = 0
    predictors = []
    while index < X_examples:
        # Set initial node as root node
        current_node = node_list[0]
        # Keep running until reach the deepest layer
        while not current_node.left_child == None or not current_node.right_child == None:
            # Move to left child node if less than threshold
            if index <= current_node.threshold:
                current_node = current_node.left_child
            # Move to right if larger than threshold
            else:
                current_node = current_node.right_child
        predictor = current_node.predictor[0]
        predictors.append(predictor)
        index += 1
    return predictors


# Read features, targets, L1-regression parameters
abs_x = csv_reader("abs_", 'x')
abs_y = csv_reader("abs_", 'y')
abs_predict_model = pd.read_csv("abs_data.csv", usecols=['parameters']).values.astype(np.float64)

linear_x = csv_reader("linear_", 'x')
linear_y = csv_reader("linear_", 'y')
linear_predict_model = pd.read_csv("linear_data.csv", usecols=['parameters']).values.astype(np.float64)

sin_x = csv_reader("sin_", 'x')
sin_y = csv_reader("sin_", 'y')
sin_predict_model = pd.read_csv("sin_data.csv", usecols=['parameters']).values.astype(np.float64)

# Get data that finish this figure need
x1 = abs_x[:, 0]
upper_y1 = abs_y[:, 1]
lower_y1 = abs_y[:, 0]

x2 = sin_x[:, 0]
upper_y2 = sin_y[:, 1]
lower_y2 = sin_y[:, 0]

x3 = linear_x[:, 0]
upper_y3 = linear_y[:, 1]
lower_y3 = linear_y[:, 0]

# Sort data for drawing figure
x1_sort = np.argsort(x1)
x2_sort = np.argsort(x2)
x3_sort = np.argsort(x3)

abs_x_sort = abs_x[x1_sort]
sin_x_sort = sin_x[x2_sort]
linear_x_sort = linear_x[x3_sort]
x1_sorted = x1[x1_sort]
x2_sorted = x2[x2_sort]
x3_sorted = x3[x3_sort]

y1_lower_sorted = lower_y1[x1_sort]
y1_upper_sorted = upper_y1[x1_sort]
y2_lower_sorted = lower_y2[x2_sort]
y2_upper_sorted = upper_y2[x2_sort]
y3_lower_sorted = lower_y3[x3_sort]
y3_upper_sorted = upper_y3[x3_sort]

# Generate target function
fun_abs = np.abs(x1_sorted-5)
fun_sin = np.sin(x2_sorted)
fun_linear = x3_sorted/5

abs_Interval_predict = []
linear_Interval_predict = []
sin_Interval_predict = []

# Generate predictors with L1-regression
i = 0
while i < 200:
    abs_Interval_predict.append(abs_predict_model[0][0])
    sin_Interval_predict.append(sin_predict_model[0][0])
    linear_Interval_predict.append(linear_predict_model[0][0])
    j = 1
    while j < 21:
        abs_Interval_predict[i] += abs_x_sort[i, j - 1] * abs_predict_model[j][0]
        sin_Interval_predict[i] += sin_x_sort[i, j - 1] * sin_predict_model[j][0]
        linear_Interval_predict[i] += linear_x_sort[i, j - 1] * linear_predict_model[j][0]
        j += 1
    i += 1

# Run MMIT function to generate MMIT predictor
nodes1 = MMIT(y1_lower_sorted, y1_upper_sorted, 0.4)
predictors1 = predictor(nodes1, 200)
nodes2 = MMIT(y2_lower_sorted, y2_upper_sorted, 0.4)
predictors2 = predictor(nodes2, 200)
nodes3 = MMIT(y3_lower_sorted, y3_upper_sorted, 0.4)
predictors3 = predictor(nodes3, 200)

# Figure 1: abs data
plt.figure(figsize=(7, 2))
plt.subplot(131)
plt.title("f(x) = |x|")
plt.scatter(x1, upper_y1, edgecolor='grey', facecolor='grey', linewidth=1.0, s=10, label="Upper limit")
plt.scatter(x1, lower_y1, edgecolor='grey', facecolor='none', linewidth=1.0, s=10, label="Lower limit")
plt.plot(x1_sorted, abs_Interval_predict, color="blue", linewidth=2, linestyle='-', label="L1-regression")
plt.plot(x1_sorted, predictors1, color="red", linewidth=2, linestyle='-', label="MMIT")
plt.plot(x1_sorted, fun_abs, color="black", linewidth=2, linestyle='-', zorder=1, label="Target function")

# Figure 2: sin data
plt.subplot(132)
plt.title("f(x) = sin(x)")
plt.scatter(x2, upper_y2, edgecolor='grey', facecolor='grey', linewidth=1.0, s=10, label="Upper limit")
plt.scatter(x2, lower_y2, edgecolor='grey', facecolor='none', linewidth=1.0, s=10, label="Lower Limit")
plt.plot(x2_sorted, predictors2, color="red", linewidth=2, linestyle='-', label="MMIT")
plt.plot(x2_sorted, sin_Interval_predict, color="blue", linewidth=2, linestyle='-', label="L1-regression")
plt.plot(x2_sorted, fun_sin, color="black", linewidth=2, linestyle='-', zorder=1, label="Target function")

# Figure 3: linear data
plt.subplot(133)
plt.title("f(x) = x/5")
plt.scatter(x3, upper_y3, edgecolor='grey', facecolor='grey', linewidth=1.0, s=10, label="Upper limit")
plt.scatter(x3, lower_y3, edgecolor='grey', facecolor='none', linewidth=1.0, s=10, label="Lower limit")
plt.plot(x3_sorted, predictors3, color="red", linewidth=2, linestyle='-', label="MMIT")
plt.plot(x3_sorted, linear_Interval_predict, color="blue", linewidth=2, linestyle='-', label="L1-regression")
plt.plot(x3_sorted, fun_linear, color="black", linewidth=2, linestyle='-', zorder=1, label="Target function")
plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
plt.show()
