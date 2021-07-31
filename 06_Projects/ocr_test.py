import requests

# Azure OCR URL --> octet-stream test

subscription_key = 'subscription_key'
vision_base_url = 'endpoint_url'
ocr_url = vision_base_url + 'ocr'

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-Type': 'application/octet-stream'
}
params = {
    'language': 'ko',
    'detectOrientation': 'true'
}

read_image = open("test2.jpg", "rb")
# with open("test2.jpg", "rb") as f:
#     read_image = f.read()

response = requests.post(ocr_url, headers=headers, params=params, data=read_image)
result = response.json()
print(result)
