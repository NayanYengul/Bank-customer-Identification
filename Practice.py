binary = bin(17)[2:]



#########################
count_z = 0
count_o = 0
length_binary = len(binary)
for i in range(0, length_binary + 1):
    for j in range(i + 1, length_binary + 1):

        substring = binary[i:j]
        count_zero = substring.count('0')

        count_one = substring.count('1')

        mod_zero = count_zero % 2
        mod_one = count_one % 2
        if mod_zero != 0:
            count_z = count_z + 1
        if mod_one != 0:
            count_o = count_o + 1

print(str(count_z) + " " + str(count_o))


