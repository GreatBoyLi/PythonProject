import math
from homework1 import gcd, is_comprise, euler_fun


# 获得质因子
# 时间复度为 O(logN)
def prime_factor(n: int) -> set[int]:
    result = set()
    while n % 2 == 0:
        result.add(2)
        n /= 2

    i = 3
    flag = math.sqrt(n)
    while i <= flag:
        while n % i == 0:
            result.add(i)
            n /= i
        i += 2

    if n > 2:
        result.add(int(n))

    return result


# 计算本原根
# 时间复杂度为 O(N*logN)
def primitive_root(num: int) -> (list[int], list[int]):
    result = []
    # 时间复杂度为 O(N*logN)
    iterm = euler_fun(num)
    exponent = num - 1
    preset = prime_factor(exponent)

    for x in iterm:
        flag = True
        for y in preset:
            aa = x ** (exponent / y)
            if (aa % num == 1):
                flag = False
                break
        if flag:
            result.append(x)

    return result, iterm


if __name__ == '__main__':

    print(prime_factor(307))

    a = input("输入要计算的本原根的数字：")
    result_end, iterm_end = primitive_root(int(a))
    print(f"与{a}互素的数有\"{len(iterm_end)}\"个，分别是：{iterm_end}")
    print(f"{a}的本原根有\"{len(result_end)}\"个，分别是：{result_end}")

