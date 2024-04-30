def sort_it(array):
    count = 0
    i = 0;
    while i < len(array):
        j = i
        while j > 0 and array[j-1] > array[j]:
            array[j-1], array[j] = array[j], array[j-1]
            j -= 1
            count += 1
        i += 1
    return count

array = [4,3,5,1,6]
print(sort_it(array))
# 4
