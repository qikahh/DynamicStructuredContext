import numpy as np

def bubble_sort(arr: np.array)->np.array:
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def find_subarray(arr: np.array, value: int)-> np.array:
    """
    将输入数组按是否大于value切分成两个子数组并分别升序排序
    """
    subarray1 = arr[arr <= value]
    subarray2 = arr[arr > value]

    return bubble_sort(subarray1), bubble_sort(subarray2)
    
    
    
    
    