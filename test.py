def largestRectangleArea( heights):
    """
    :type heights: List[int]
    :rtype: int
    """
    left_value = [0]
    left_height = [0]
    for i in range(1, len(heights)):
        value = 0
        while (len(left_height) and heights[left_height[-1]] >= heights[i]):
            value = i-left_height[-1]
            left_height.pop()
        if len(left_height) == 0:
            value = i
        left_height.append(i)
        left_value.append(value)
    right_value = [0]
    right_height = [len(heights)-1]
    for i in range(len(heights)-2, -1, -1):
        value = 0
        while (len(right_height) and heights[right_height[-1]] >= heights[i]):
            value = right_height[-1]-i
            right_height.pop()
        if len(right_height) == 0:
            value = len(heights)-i-1
        right_height.append(i)
        right_value = [value] + right_value

    max_value = 0
    for i in range(len(heights)):
        value = heights[i]*(1+right_value[i]+left_value[i])
        if value > max_value:
            max_value = value
    return max_value
    

heights = [999,999,999,999]
result = largestRectangleArea(heights)