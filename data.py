import numpy as np

digit0_label = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
digit1_label = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
digit2_label = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
digit3_label = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
digit4_label = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
digit5_label = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
digit6_label = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
digit7_label = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
digit8_label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
digit9_label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

digitZero = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])

digitOne = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
])

digitTwo = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
])

digitThree = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])

digitFour = np.array([
    [0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
])

digitFive = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])

digitSix = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])

digitSeven = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
])

digitEight = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])

digitNine = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])

# The input and output values need to be in order. So first, item in
# the array is zero, so the output array item number 1 should be zero
training_input = [
    digitZero.flatten(),
    digitOne.flatten(),
    digitTwo.flatten(),
    digitThree.flatten(),
    digitFour.flatten(),
    digitFive.flatten(),
    digitSix.flatten(),
    digitSeven.flatten(),
    digitEight.flatten(),
    digitNine.flatten()
]
training_expected_output = [
    digit0_label, digit1_label, digit2_label, digit3_label, digit4_label,
    digit5_label, digit6_label, digit7_label, digit8_label, digit9_label
]
