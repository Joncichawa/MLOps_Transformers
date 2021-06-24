import requests

requests.adapters.DEFAULT_RETRIES = 1

url = f"http://localhost:7071/api/classify"
url2 = "https://mlopsapp.azurewebsites.net/api/classify"

resp = requests.get(url)
txt1 = "The Other Place subtitled And Other Stories of the Same Sort is a collection of science fiction and fantasy stories by J. B. Priestley published in hardcover by Harper & Brothers and Heinemann in 1953. The title story original to the collection was adapted as an episode of the television series Westinghouse Studio One in 1958 starring Cedric Hardwicke as a sorcerer with chin whiskers"
# txt2 = "Artist Profile is an international contemporary art magazine published in Sydney Australia."
txt2 = "Robert Lewandowski(born 21 August 1988) is a Polish professional footballer who plays as a striker for Bundesliga club Bayern Munich and is the captain of the Poland national team. Recognized for his positioning, technique and finishing, Lewandowski is considered one of the best strikers of all time, as well as one of the most successful players in Bundesliga history. He has scored over 500 senior career goals for club and country. "
params = dict()
params["text"] = f"['{txt1}', '{txt2}']"
# params["code"] = "FA6mmG61MKHOs5nuxYKX6YlVvDsUpa0M2Or55joJuGuTzPBIQWxVSA=="
# params['header'] = "'Content-Type': 'application/x-www-form-urlencoded'"

# r = requests.get(url, params=params)
r = requests.get(url, params=params)
print(r.text)
