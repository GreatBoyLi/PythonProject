# 求最大公约数
# 时间复杂度为 O(logN)
def gcd(a: int, b: int) -> int:
    while b != 0:
        a, b = b, a % b
    return a


# 判断是否互素
def is_comprise(a: int, b: int) -> bool:
    if gcd(a, b) == 1:
        return True
    return False


# 获取互素的整数
# 时间复杂度为 O(N*logN)
def euler_fun(num: int) -> list[int]:
    iterm = []
    for x in range(1, num):
        if is_comprise(num, x):
            iterm.append(x)
    return iterm


# 计算本原根
# 时间复杂度为 O(N*logN)
def primitive_root(num: int) -> (list[int], list[int]):
    result = []
    iterm = euler_fun(num)
    for x in iterm:
        for i in range(1, len(iterm) + 1):
            if x ** i % num == 1:
                print(f"{x}的{i}次方mode{num}是1")
                if i == len(iterm):
                    result.append(x)
                else:
                    break
    return result, iterm


if __name__ == '__main__':
    c = input("输入要计算的本原根的数字：")
    result_end, iterm_end = primitive_root(int(c))
    print(f"与{c}互素的数有\"{len(iterm_end)}\"个，分别是：{iterm_end}")
    print(f"{c}的本原根有\"{len(result_end)}\"个，分别是：{result_end}")