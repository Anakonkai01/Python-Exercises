# Define the original matrix
matrix = [
    [1, 2, 3, 5, 8],
    [5, 7, 7, 8,9,],
    [0, 2,4, 6, 8,],
    [8, 2,9, 3,7]
]

# Function to find the length of the longest contiguous odd numbers sequence in a list
def longest_odd_sequence(lst):
    max_length = 0
    current_length = 0
    for num in lst:
        if num % 2 != 0:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0
    return max_length

# Find the length of the longest contiguous odd numbers sequence in each row
max_odd_sequence_lengths = [longest_odd_sequence(row) for row in matrix]

# Find the maximum length of the contiguous odd numbers sequence among all rows
max_length = max(max_odd_sequence_lengths)

# Find the rows with the maximum length of the contiguous odd numbers sequence
rows_with_max_length = [row for row, length in zip(matrix, max_odd_sequence_lengths) if length == max_length]

# Print the rows with the longest contiguous odd numbers sequence
print("Rows with the longest contiguous odd numbers sequence:")
for row in rows_with_max_length:
    print(row)
