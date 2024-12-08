# 给定一个非负整数 num，反复将各个位上的数字相加，直到结果为一位数
num = 0
while 1:
    # input a number
    try:
        num = int(input("Enter an integer number: "))
        break
    except ValueError:
        print("Please input integer only...")
        continue
sum = 0
sum1 = 0
while 1:
    if num > 10 or sum > 10:
        while num != 0:
            sum += (num % 10)
            num = num // 10
        while sum != 0:
            sum1 += sum % 10
            sum = sum // 10
        sum = sum1
    else:
        print(sum)
        break


