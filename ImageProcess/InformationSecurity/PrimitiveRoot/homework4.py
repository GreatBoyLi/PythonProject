
def ex_gcd(dividend, divisor):
    x1, x2, x3 = 1, 0, divisor
    y1, y2, y3 = 0, 1, dividend
    count = 0
    print(f"初始值    *     {x1}      {x2}      {x3}    |     {y1}      {y2}      {y3}")
    while y3 != 0 and y3 != 1:
        q = int(x3 / y3)
        t1, t2, t3 = x1 - q * y1, x2 - q * y2, x3 - q * y3
        x1, x2, x3 = y1, y2, y3
        y1, y2, y3 = t1, t2, t3
        count += 1
        print(f"{count}        {q}     {x1}      {x2}      {x3}    |     {y1}      {y2}      {y3}")
    if y3 == 0:
        return -1, x3   #第一个值是逆元
    if y3 == 1:
        return y2 % divisor, y3


# 求逆元和最大公约数
if __name__ == '__main__':
    x, y = ex_gcd(11, 20)   # 550 % 1769
    print(f"逆元是{x}")
    print(f"最大公约数是{y}")
