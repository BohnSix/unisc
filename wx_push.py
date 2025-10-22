import os
import requests
import re


def sc_send(sendkey, title, desp="", options=None):
    if options is None:
        options = {}
    # 判断 sendkey 是否以 'sctp' 开头，并提取数字构造 URL
    if sendkey.startswith("sctp"):
        match = re.match(r"sctp(\d+)t", sendkey)
        if match:
            num = match.group(1)
            url = f"https://{num}.push.ft07.com/send/{sendkey}.send"
        else:
            raise ValueError("Invalid sendkey format for sctp")
    else:
        url = f"https://sctapi.ftqq.com/{sendkey}.send"
    params = {"title": title, "desp": desp, **options}
    headers = {"Content-Type": "application/json;charset=utf-8"}
    response = requests.post(url, json=params, headers=headers)
    result = response.json()
    return result


def tell_me(title="主人服务器宕机了 via python", content="第一行\n\n第二行"):
    data = {}
    print(os.path.join(os.path.dirname(__file__), "..", ".env"))
    # with open(os.path.join(os.path.dirname(__file__), "..", ".env"), "r") as f:
    #     for line in f:
    #         key, value = line.strip().split("=")
    #         data[key] = value
    # key = data["SENDKEY"]
    # ret = sc_send(key, title, content)
    # return ret


if __name__ == "__main__":
    ret = tell_me(title="Cityscapes Inference Finished")
    print(ret)
