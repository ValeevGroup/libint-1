import requests, os

def load(basis):
  elements = None
  try:
    elements = basis['elements']
  except:
    import json
    elements = json.load(basis)['elements']
  basis = {}
  for Z in map(int,elements):
    basis[Z] = []
    # print(Z)
    for f in (elements[str(Z)]['electron_shells']):
      angular_momentum = f['angular_momentum']
      exponents = list(map(float, f['exponents']))
      coefficients = [list(map(float,c)) for c in f['coefficients']]
      for i,L in enumerate(angular_momentum):
        primitives = list(zip(exponents, coefficients[i]))
        basis[Z].append((L, primitives))
  return basis

def load_from_bse(basis_name, bse="http://basissetexchange.org"):
  # This allows for overriding the URL via an environment variable
  # Feel free to just use the base_url below
  base_url = os.environ.get('BSE_API_URL', bse)
  headers = {}
  #params = {'elements': ['H','C','O']}
  r = requests.get(
    base_url + '/api/basis/%s/format/json'% (basis_name),
    headers=headers,
    #params=params
  )
  return load(r.json())
