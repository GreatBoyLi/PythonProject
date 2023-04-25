import gmpy2
from cryptography.hazmat.primitives.asymmetric import ec


# 定义椭圆曲线参数
p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
a = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2C
b = 0x0000000000000000000000000000000000000000000000000000000000000030
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
curve = ec.SECP256K1()  # 使用SECP256K1曲线


def inverse_mod(a, m):
    """使用扩展欧几里得算法求a在模m意义下的乘法逆元"""
    _, x, _ = gmpy2.gcdext(a, m)
    return x % m


def scalar_mult(scalar, point):
    """使用椭圆曲线倍乘算法计算标量乘"""
    x, y = point.public_numbers().x, point.public_numbers().y
    return ec.EllipticCurvePublicNumbers(
        x=(scalar * x) % p, y=(scalar * y) % p, curve=curve
    ).public_key()


def mod_inv(n, p):
    """使用费马小定理计算n在模p意义下的乘法逆元"""
    return pow(n, p - 2, p)


def fractional_mod(n, d, p):
    """使用椭圆曲线和费马小定理进行分数求模"""
    d_inv = inverse_mod(d, p)
    point = ec.EllipticCurvePublicNumbers(x=n, y=1, curve=curve).public_key()
    scalar = (d * d_inv) % p
    result = scalar_mult(scalar, point)
    return result.public_numbers().x


# 测试
n = 123456789
d = 987654321
p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
result = fractional_mod(n, d, p)
print(result)
