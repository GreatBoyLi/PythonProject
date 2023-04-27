def mod_fraction(a, b, p):
    """
    使用费马小定理求 a/b 模 p 的值，其中 p 为质数
    """
    # 计算 b^(p-2) 模 p 的值，即 b 在模 p 意义下的逆元
    b_inv = pow(b, p-2, p)
    # 计算 a * b_inv 的值，并对 p 取模
    return (a * b_inv) % p


# a = 2 / 3
# b = 23
# c = (2 * 3 ** (b - 2)) % b
# print(c)

print(mod_fraction(1,2,23))
