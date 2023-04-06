num = 0
result = list()
digits = [1, 2, 3]
for i in range(len(digits)-1, -1, -1):
    a = digits[i]
    num += a  * 10 ** (len(digits)-1-i)
num += 1
while(num != 0):
    result.append(num % 10)
    num = num // 10

print(result)