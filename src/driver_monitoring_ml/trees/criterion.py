# given the parent, left and right 1D label arrays, calculate the information gain
def information_gain(parent_1D_label_array, left_child_1D_label_array, right_child_1D_label_array) -> float:
    # possible labels are 0 or 1

    num_0s_parent = 0
    num_1s_parent = 0

    for i in range(0, len(parent_1D_label_array)):
        if parent_1D_label_array[i] == 0:
            num_0s_parent += 1
        else:
            num_1s_parent += 1

    I_parent = 1 - (num_0s_parent / len(parent_1D_label_array)) ** 2 - (num_1s_parent / len(parent_1D_label_array)) ** 2 

    num_0s_left = 0
    num_1s_left = 0

    for i in range(0, len(left_child_1D_label_array)):
        if left_child_1D_label_array[i] == 0:
            num_0s_left += 1
        else:
            num_1s_left += 1

    I_left = 1 - ((num_0s_left / len(left_child_1D_label_array)) ** 2 + (num_1s_left / len(left_child_1D_label_array)) ** 2)

    num_0s_right = 0
    num_1s_right = 0

    for i in range(0, len(right_child_1D_label_array)):
        if right_child_1D_label_array[i] == 0:
            num_0s_right += 1
        else:
            num_1s_right += 1

    I_right = 1 - ((num_0s_right / len(right_child_1D_label_array)) ** 2 + (num_1s_right / len(right_child_1D_label_array)) ** 2)

    # calculate the information gain
    prop_left = len(left_child_1D_label_array) / len(parent_1D_label_array)
    prop_right = len(right_child_1D_label_array) / len(parent_1D_label_array)

    info_gain = I_parent - (prop_left * I_left + prop_right * I_right)
    return info_gain # the higher the information gain, the more decisive the split is