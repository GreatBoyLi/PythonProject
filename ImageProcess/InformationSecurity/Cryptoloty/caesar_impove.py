# 初实密钥
secretKey = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# 加密
def encryption(string):
    res = ''
    for x in string:
        if 97 <= ord(x) <= 122:
            y = (secretKey.index(x.upper()) + 3) % 26
            res += secretKey[y]
        else:
            res += x
    return res


# 解密
def decryption(string):
    res = ''
    for x in string:
        if 65 <= ord(x) <= 90:
            y = (secretKey.index(x) - 3) % 26
            res += secretKey[y].lower()
        else:
            res += x
    return res


# 设置加密密钥
def secret_key(string):
    global secretKey
    secretKey1 = []
    for x in string:
        if x != ' ' and x.upper() not in secretKey1:
            secretKey1.append(x.upper())
            secretKey.remove(x.upper())
    secretKey1.extend(secretKey)
    secretKey = secretKey1
    print(secretKey)


if __name__ == '__main__':
    secret_key('the message was transmitted an hour ago')
    over = False
    while not over:
        print('Casesar密码')
        message = input('请输入明文：')
        ciphertext = encryption(message)
        print(f'加密后的密文是：{ciphertext}')
        flag = input('是否需要解密？（Y：解密；N：不解密）')
        if flag == 'Y' or flag == 'y':
            plaintext = decryption(ciphertext)
            print(f'解密后的明文是：{plaintext}')
        input_a = input('是否结束？（Y：结束；N；不结束）')
        if input_a == 'Y' or input_a == 'y':
            over = True
        else:
            over = False
