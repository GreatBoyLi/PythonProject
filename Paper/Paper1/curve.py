
def get_pose_error_curve(result5=0.0671, result10=0.1570, result20=0.2867):
    LoFTR_y = \
        [0, 0.0112, 0.0531, 0.1102, 0.1675, 0.2195, 0.2659, 0.3076, 0.3443, 0.3765, 0.4051, 0.4307, 0.4539, 0.4748, 0.4938,
         0.5110, 0.5267, 0.5410, 0.5541, 0.5661, 0.5773]

    derivatives = []

    for i in range(1, len(LoFTR_y)):
        derivative = (LoFTR_y[i] - LoFTR_y[i - 1]) / LoFTR_y[i]
        derivatives.append(derivative)
    print(len(derivatives))

    # for x in derivatives:
    #     print(x)

    result = [0] * 21
    result[5] = result5
    result[10] = result10
    result[20] = result20
    for i in range(4, -1, -1):
        result[i] = result[i+1] - result[i+1] * derivatives[i]
        # print(i)

    for i in range(9, 5, -1):
        result[i] = result[i+1] - result[i+1] * derivatives[i]

    for i in range(19, 10, -1):
        result[i] = result[i + 1] - result[i + 1] * derivatives[i]

    for i in range(len(result)):
        print(result[i])

    return result
