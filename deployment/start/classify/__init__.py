import logging
import azure.functions as func
import json
import ast
# Import helper script
from .predictonnx import predict_class_from_text

headers = {
              "Content-type": "application/json",
              "Access-Control-Allow-Origin": "*"
}

def main(req: func.HttpRequest) -> func.HttpResponse:
       if req.params == {}: return func.HttpResponse(json.dumps({"3":"2"}, indent=4), headers = headers)
       input_text = req.params.get('text')

       logging.info(input_text)

       input_text = ast.literal_eval(input_text)
       results = predict_class_from_text(input_text)

       return func.HttpResponse(json.dumps(results, indent=4), headers = headers)

