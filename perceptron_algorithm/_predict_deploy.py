# -*- coding: utf-8 -*-
"""
Reads in the model artifacts, then makes a prediction
I also have no idea what I'm doing
"""

import argparse
from flask import Flask, request
from flask_restx import Api, reqparse, Namespace, fields, Resource

import pathlib

from statsmodels.iolib.smpickle import load_pickle

parser = argparse.ArgumentParser(description='Load & deploy a linear regression model.')
parser.add_argument('--model', help='fpath of the pickle file')
args = parser.parse_args()

results_model = load_pickle(pathlib.Path(args.model))

app = Flask(__name__)
api = Api(app)

# Namespace
ns_test = Namespace('predict_model', description='Model predictions')

# Models
dict_params = {}
for key in results_model.params.keys():
    dict_params[key] = fields.Float(required=True, location='json')

custom_model = ns_test.model('Custom', dict_params)

# Parser - Sets up Api doc
parser_requests = reqparse.RequestParser()
for key in results_model.params.keys():
    parser_requests.add_argument(key, type=float, required=True)

@ns_test.route('/predict_perceptron')
class Custom(Resource):

    @ns_test.expect(parser_requests)
    @ns_test.marshal_with(custom_model)
    def get(self):
        # args_request = parser_requests.parse_args()

        dict_features = {}
        for key in results_model.params.keys():
            dict_features[key] = request.args.get(key)

        # make prediction
        list_terms = [
            results_model.params[key]*value for key, value in dict_features.items()
        ]
        result = sum(list_terms)

        return { 'prediction': result }

api.add_namespace(ns_test)

if __name__ == "__main__":
    app.run()

# Scratch ========================================================
    # """
    # ---
    # parameters:
    #     - name: Area
    #       in: query
    #       #type: number
    #       required: True
    # responses:
    #     200:
    #         description: OK
    # """

# class ModelFeatures(Resource):
# def get(self):
#     parser = reqparse.RequestParser()
#     for key in results_model.params.keys():
#         parser.add_argument(key, type=float, required=True)
#     args_request = parser.parse_args()
#     return args_request

# app_model.add_resource(ModelFeatures, '/features')


# @app_model.route('/predict_model', methods=['GET'])
# def predict_model():
#     # get entries
#     dict_features = {}
#     for key in results_model.params.keys():
#         dict_features[key] = request.args.get(key)

#     # make prediction
#     list_terms = [
#         results_model.params[key]*value for key, value in dict_features.items()
#     ]
#     result = sum(list_terms)

#     return str(result)