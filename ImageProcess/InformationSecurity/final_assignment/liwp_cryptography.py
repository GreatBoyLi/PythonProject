import rsa
from binascii import b2a_hex, a2b_hex


class rsacrypt():
    def __init__(self, pubkey, prikey):
        self.pubkey = pubkey
        self.prikey = prikey

    def encrypt(self, text):
        self.ciphertext = rsa.encrypt(text.encode(), self.pubkey)
        # 因为rsa加密时候得到的字符串不一定是ascii字符集的，输出到终端或者保存时候可能存在问题
        # 所以这里统一把加密后的字符串转化为16进制字符串
        return b2a_hex(self.ciphertext)

    def decrypt(self, text):
        decrypt_text = rsa.decrypt(a2b_hex(text), prikey)
        return decrypt_text.decode()


if __name__ == '__main__':
    pubkey, prikey = rsa.newkeys(256)
    rs_obj = rsacrypt(pubkey,prikey)
    text='hello中国'
    ency_text = rs_obj.encrypt(text)
    print(ency_text)
    print(rs_obj.decrypt(ency_text))
    print(pubkey)
    print(prikey)

"""
b'0be7e588fd8d093de48e1ff16f177273d771e0eb6b475458e31b879aebef18f6'
hello中国
PublicKey(73203632342415782735861635389738854293696902283956451011523986747143440837753, 65537)
PrivateKey(73203632342415782735861635389738854293696902283956451011523986747143440837753, 65537, 56404226961792114085368646753370507079134793768842910110408058154189894119761, 55914002491444170148907949791677664855437, 1309218247318589494605835772663699869)

"""
