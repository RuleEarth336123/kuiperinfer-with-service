import os
import json
import requests

def yolo1():
    url = 'http://localhost:12124/yolo/v1'
    
    data = {
        "images" : ["/home/chunyu123/github/KuiperInfer/imgs/bus.jpg",
                    "/home/chunyu123/github/KuiperInfer/imgs/bus.jpg",
                    "/home/chunyu123/github/KuiperInfer/imgs/bus.jpg",
                    "/home/chunyu123/github/KuiperInfer/imgs/bus.jpg",
                    "/home/chunyu123/github/KuiperInfer/imgs/bus.jpg",
                    "/home/chunyu123/github/KuiperInfer/imgs/bus.jpg",
                    "/home/chunyu123/github/KuiperInfer/imgs/bus.jpg",
                    "/home/chunyu123/github/KuiperInfer/imgs/bus.jpg"],
        "batch_size" : "8",
        "param_path" : "/mnt/d/linux/models/yolov5s_batch8.pnnx.param",
        "bin_path" : "/mnt/d/linux/models/yolov5s_batch8.pnnx.bin"
    }
    
    json_data = json.dumps(data, indent=4)
    
    print(json_data)
        
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, data=json_data, headers=headers)

    print('Status Code:', response.status_code)

    if response.status_code == 200:
        response_json = json.loads(response.text)
        print(response_json)
               
    else:
        print('Failed to get a valid response from the server.')
    
    return

yolo1()
print(1)