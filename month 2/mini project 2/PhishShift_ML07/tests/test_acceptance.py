"""Lightweight API acceptance tests. Run after starting the app in another terminal."""
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE="http://127.0.0.1:5000"
def post(payload):
    req=Request(BASE+"/api/predict",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    try:
        with urlopen(req,timeout=5) as r: return r.status,json.loads(r.read())
    except HTTPError as e: return e.code,json.loads(e.read())

if __name__=="__main__":
    print("AT-01 legitimate urgent password-reset notice")
    print(post({"message":"Urgent: reset your password using the official company portal you normally visit."}))
    print("AT-02 unfamiliar phishing wording")
    print(post({"message":"A quick note from support: share your login details to keep mailbox active."}))
    print("AT-03 punctuation/defanged wording; no URL is opened")
    print(post({"message":"SECURITY alert!!! verify-account now [dot] [slash] shortened link"}))
    print("Invalid empty input")
    print(post({"message":"   "}))
