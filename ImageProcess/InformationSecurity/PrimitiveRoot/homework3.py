from homework2 import *

def is_primitive_root(num: int, test: int) -> bool:
    result = []
    # 时间复杂度为 O(N*logN)
    # iterm = euler_fun(num)
    exponent = num - 1
    preset = prime_factor(exponent)

    flag = True
    for y in preset:
        a = (exponent / y)
        aa = test ** a
        print(f"{test}的({num}-1)/{y}次方是{aa},mode{num}的值是{aa % num}")
        if (aa % num == 1):
            flag = False
    return flag


if __name__ == '__main__':
    a = input("输入要计算的本原根的数字：")
    b = input("输入要测试的本原根：")
    flag = is_primitive_root(int(a), int(b))
    if flag:
        print(f"{b}是{a}的本原根。")
    else:
        print(f"{b}不是{a}的本原根。")
