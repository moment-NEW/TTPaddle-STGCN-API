import json

import requests

response = requests.post(
    'http://localhost:5006/infer',
    json={'video_path': 'person_crop.mp4',"jump": True}
)

print(json.dumps(response.json(), indent=2))
