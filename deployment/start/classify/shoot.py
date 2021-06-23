import requests

# ipt_text = [" Gino Daniel Rea (born 18 September 1989 inTooting London) is an English motorcycle racer ofItalian descent.2010 was his first season in theSupersport World Championship riding for theIntermoto Czech Honda team. He won the 2009European Superstock 600 championship on a GeorgeWhite-backed Ten Kate Honda clinching the title bya single point in the final round at Portimão.Reawas a successful Motocross and Supermoto riderbefore switching to circuit racing in 2007."]

url = f"http://localhost:7071/api/classify"

print(url)

resp = requests.get(url)

params = dict()
params["text"] = [" Gino Daniel Rea (born 18 September 1989 inTooting London) is an English motorcycle racer ofItalian descent.2010 was his first season in theSupersport World Championship riding for theIntermoto Czech Honda team. He won the 2009European Superstock 600 championship on a GeorgeWhite-backed Ten Kate Honda clinching the title bya single point in the final round at Portimão.Reawas a successful Motocross and Supermoto riderbefore switching to circuit racing in 2007."]


r = requests.get(url, params=params)
print(r.url)



# print(resp.text)