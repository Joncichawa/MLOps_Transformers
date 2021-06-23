import logging

import azure.functions as func
import json

# Import helper script
from .predictonnx import predict_class_from_text

def main(req: func.HttpRequest) -> func.HttpResponse:
       input_text = req.params.get('text')
       logging.info('Input text to classify received: ' + input_text)

       results = predict_class_from_text(input_text)

       headers = {
                     "Content-type": "application/json",
                     "Access-Control-Allow-Origin": "*"
       }

       return func.HttpResponse(json.dumps(results, indent=4), headers = headers)

