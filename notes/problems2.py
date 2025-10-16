test_string = "hello!坦克没有后视镜！"
utf8_encoded = test_string.encode("utf-8")
utf16_encoded = test_string.encode("utf-16")
utf32_encoded = test_string.encode("utf-32")

list8 = list(utf8_encoded)
list16 = list(utf16_encoded)
list32 = list(utf32_encoded)

def decode_uft8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([ bytes([b]).decode("utf-8") for b in bytestring])

def decode_right(bytestring: bytes):
    return bytestring.decode("utf-8")

decode_right("hello".encode('utf-8'))

[ ([b]).decode("utf-8") for b in utf8_encoded]