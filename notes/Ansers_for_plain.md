# Problems 1
1.a chr(0) returns '\x00';
1.b 

# Problems Unicode2: Unicode Encodings
a. UTF-8 encoding is space-efficient.
b. The return 
    1. break the unicode byte series into separated list elements which could potentially break some non-English characters, which takes more then one byte to encode.
    2. Then this function tries to decode on each separated elements. When the decoder runs into non-english tokens, it will raise error.
c.  
    ```python
    a = b'\xe5x9d' 
    a.decode()
    ```