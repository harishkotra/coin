"""Populate the wall with real reads. Every number here comes out of the model."""
import json
import urllib.request

Q = [
    ("Should I quit my job to build this full time?",
     "I have 11 months of savings. The product has 40 paying users at 12 dollars a month. I am the only person working on it, about 3 hours a night. My co-founder left last month. I have not talked to a customer in six weeks."),
    ("Should I move to Bangalore next month?",
     "Rent would go from 18000 to 32000 a month. My take home is 95000. Both sets of parents and my entire friend group are in Bangalore. My current lease ends in five weeks and the landlord wants an answer now. I work remote and keep the same job."),
    ("Will this launch land on time?",
     "Two of four milestones are done. The other two are blocked on a vendor who responds but is slow. The team is split on whether it slips. No marketing is booked yet."),
    ("Should I take the offer?",
     "It is 20 percent more money. The commute doubles from 30 to 60 minutes. The team looks fine. I am neither excited nor unhappy where I am."),
    ("Should I raise the price from 12 to 29 dollars?",
     "Forty paying users, none churned in three months, average usage up 30 percent month on month. Support is one email a week. The product replaces a 200 dollar a month tool. I have never tested a price change."),
    ("Should I hire a designer before the launch?",
     "I have 6000 dollars set aside. The site converts at 1.8 percent from 4000 visits a month. I can ship the current design next week or wait three weeks for someone else."),
    ("Should I get a dog?", ""),
    ("Should I text my ex?", ""),
    ("Is it worth learning Rust in 2026?", ""),
    ("Will my startup be profitable in two years?", ""),
]


def post(payload):
    req = urllib.request.Request(
        "http://127.0.0.1:8077/api/decide",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


for q, ctx in Q:
    try:
        d = post({"question": q, "context": ctx})
        print(f"{d['p']:.3f}  ev {d['evidence']:.2f}  spread {d['spread']:.2f}  {q}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("FAILED", q, exc, flush=True)
