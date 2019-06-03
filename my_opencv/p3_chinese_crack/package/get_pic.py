"""
author: wiken
Date:2019/6/3
"""
from requests_html import HTMLSession
import random
import time

s = HTMLSession()


for i in range(100):
    r = s.get(f"https://3u.travelsky.com/3ub2b/VerificationCode.do?agentType=domestic&{random.random()}")
    content = r.content
    # print(content)
    name = f"./pic/{str(int(time.time()*1000))}.png"
    with open(name, "wb") as f:
        f.write(content)
