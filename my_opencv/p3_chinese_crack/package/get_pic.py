"""
author: wiken
Date:2019/6/3
"""
from requests_html import HTMLSession
import random
import time

s = HTMLSession()
header = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3", "Accept-Encoding": "gzip, deflate, br", "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8", "Cache-Control": "no-cache", "Connection": "keep-alive", "Cookie": "PaaS-Router-SessionID=18487d5f0e4d30ae7314e43ed1271fb4; HttpOnly; PaaS-Router-SessionID=18487d5f0e4d30ae7314e43ed1271fb4; HttpOnly; JSESSIONID=qRw-ft5PCOlJGwKtJhoSP5Yf; Webtrends=222.178.116.97.1553572277513087; 18487d5f0e4d30ae7314e43ed1271fb4=70306dc7fff6ec4777f7ccfa04bbff8d; TUB2BCOOKIE_CAS_SSO_SID=E6764E1166544AB1EA64D8A66D21ED7C5ACB0187A3EEA24EAD122C5CE58D4ADF8D4F0DA7476A89EFBC4FC283558C05ED05CD22D6775655421AF08773944734B00DF7F0B34068F7A5916B95DBF4D6CFE3; TUB2BCDE_AGENT_TOKEN=""; TUB2BCOOKIE_AGENT_LOGIN_LANGUAGE=zh_CN", "Host": "3u.travelsky.com", "Pragma": "no-cache", "Upgrade-Insecure-Requests": "1", "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"}
print(random.random())
for i in range(10):
    r = s.get(f"https://3u.travelsky.com/3ub2b/VerificationCode.do?agentType=domestic&{random.random()}",headers=header, verify=False)
    content = r.content
    # print(content)
    name = f"./pic/{str(int(time.time()*1000))}.png"
    with open(name, "wb") as f:
        f.write(content)
